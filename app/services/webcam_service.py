import contextlib
import io
import uuid
import cv2
import torch
import time
import os
import boto3
import numpy as np
import queue
from collections import deque
from ultralytics import YOLO
from fastapi.responses import JSONResponse
from config import config
from app.services.cnn3d import Deeper3DCNN_Gray

# ✅ 실시간 또는 파일 실행 모드
USE_VIDEO_FILE = True  # 🔄 영상 파일 기반 실행으로 설정
VIDEO_FILE_PATH = "app/services/test/pulling.mp4"  # 🔄 실제 테스트용 mp4 파일 경로

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 실행 장치: {device}")
print("🎮 현재 모드: 파일 재생 기반" if USE_VIDEO_FILE else "📸 현재 모드: 실시간 카메라")

# ✅ 모델 로드
model = YOLO("yolov8n.pt", verbose=False).to(device)

from ultralytics.utils import LOGGER # yolo 모델 로그 제거 위치 고정 
LOGGER.setLevel("WARNING")  # 또는 "ERROR" 또는 "CRITICAL"



cnn_model = Deeper3DCNN_Gray().to(device)
cnn_model.load_state_dict(torch.load("app/services/deeper3dcnn_gray3.pth", map_location=device))
cnn_model.eval()

# ✅ 상태
last_person_detected = {}
buffer_full_printed = {}
frame_buffers = {}
buffer_start_times = {}
video_paths = {}
latest_video_url = None
alerts = set()
active_cameras = {}
last_detection_time = {}
last_upload_time = {} # 전역에 저장 간격 제한 변수 추가
UPLOAD_COOLDOWN = 5  # 초 단위, 예: 10초에 한 번만 업로드 허용


if not os.path.exists("recordings"):
    os.makedirs("recordings")

# ✅ 분석 토와
frame_analysis_queue = queue.Queue()

s3_client = boto3.client(
    "s3",
    aws_access_key_id=config.ACCESS_KEY,
    aws_secret_access_key=config.SECRET_KEY,
    region_name=config.BUCKET_REGION,
)

def upload_to_s3(file_path, camera_index):
    print("📤 upload_to_s3() 실행중!")
    global latest_video_url
    try:
        filename = os.path.basename(file_path)
        s3_key = f"{config.BUCKET_DIRECTORY}/{filename}"
        with open(file_path, "rb") as file:
            s3_client.upload_fileobj(
                file, config.BUCKET_NAME, s3_key,
                ExtraArgs={
                    'ACL': 'public-read',
                    'ContentDisposition': 'inline',
                    'ContentType': 'video/mp4'
                }
            )
        s3_url = f"https://{config.BUCKET_NAME}.s3.{config.BUCKET_REGION}.amazonaws.com/{s3_key}"
        video_paths[camera_index] = s3_url
        latest_video_url = s3_url
        print(f"✅ S3 업로드 성공: {s3_url}")
        os.remove(file_path)
        return s3_url
    except Exception as e:
        print(f"❌ S3 업로드 실패: {e}")
        return None

def get_next_video_filename():
    print("📜 get_next_video_filename() 실행중!")
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    return os.path.join("recordings", f"{timestamp}_{unique_id}.mp4")


def sample_16_frames(frames):
    print("🎟 sample_16_frames() 실행중!")
    if len(frames) < 16:
        return []
    indices = np.linspace(0, len(frames) - 1, 16, dtype=int)
    return [frames[i] for i in indices]

def preprocess_frames(frames):
    print("🧪 preprocess_frames() 실행중!")
    gray_frames = [cv2.cvtColor(cv2.resize(f, (112, 112)), cv2.COLOR_BGR2GRAY) for f in frames]
    gray_frames = np.stack(gray_frames, axis=0)          # (16, 112, 112)
    gray_frames = np.expand_dims(gray_frames, axis=0)    # (1, 16, 112, 112)
    gray_frames = np.expand_dims(gray_frames, axis=1)    # ✅ (1, 1, 16, 112, 112) ← 여기 추가!
    return torch.FloatTensor(gray_frames).to(device) / 255.0

def detect_people(frame, camera_index):
    global last_person_detected
    person_detected = False

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        results = model(frame)

    for result in results:
        for box in result.boxes.data:
            class_id = int(box[5])
            confidence = float(box[4])
            if model.names[class_id] == "person" and confidence > 0.3:
                person_detected = True
                break

    # 🔄 상태 바뀌었을 때만 출력
    if camera_index not in last_person_detected or last_person_detected[camera_index] != person_detected:
        last_person_detected[camera_index] = person_detected
        if person_detected:
            print(f"🧍‍♂️ 사람 감지됨! (CAM{camera_index+1})")
        else:
            print(f"🙈 사람 안 보임! (CAM{camera_index+1})")

    return person_detected


def save_video(frames, filename):
    print("📏 save_video() 실행중!")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ✅ 안정적 코덱
    if len(frames) == 0:
        print("❌ 저장할 프레임이 없습니다.")
        return

    h, w = frames[0].shape[:2]
    print(f"🖼 저장할 영상 크기: {w}x{h}, 총 {len(frames)} 프레임")

    out = cv2.VideoWriter(filename, fourcc, 15, (w, h))
    for i, frame in enumerate(frames):
        out.write(frame)
    out.release()

def analyze_buffers(original_fps, camera_index):
    global buffer_full_printed, last_upload_time

    now = time.time()

    last_upload = last_upload_time.get(camera_index, 0)
    if now - last_upload < UPLOAD_COOLDOWN:
        return

    for buffer_id in [0, 1, 2]:
        key = (camera_index, buffer_id)
        buffer_frames = list(frame_buffers.get(key, []))

        # ✅ CNN 분석 조건: 프레임이 45개 다 차야 함
        if len(buffer_frames) < 45:
            continue

        # ✅ 디버깅용: 버퍼 꽉 찼을 때 1회 출력
        if not buffer_full_printed.get(key, False):
            print(f"✅ {key} 버퍼가 다 찼어요! CNN 분석 시작")
            buffer_full_printed[key] = True
        else:
            continue  # 이미 처리된 버퍼는 스킵

        # 🎯 CNN 분석용 16프레임 샘플링
        clip_for_cnn = sample_16_frames(buffer_frames)
        if len(clip_for_cnn) < 16:
            continue

        input_tensor = preprocess_frames(clip_for_cnn)
        with torch.no_grad():
            output = cnn_model(input_tensor)
        pred = torch.argmax(output, dim=1).item()

        print(f"🧠 CNN 결과: buffer{buffer_id} → pred={pred}")
        if pred == 1:
            print(f"🚨 폭력 감지! buffer{buffer_id} 전체 저장")

            filename = get_next_video_filename()
            save_video(buffer_frames, filename)
            upload_to_s3(filename, camera_index)

            alerts.add(f"CAM{camera_index+1}")
            last_detection_time[camera_index] = now
            last_upload_time[camera_index] = now

def background_analyzer():
    print("🧵 background_analyzer() 백그라운드 분석 시작!")
    step = 2
    buffer_duration = 5
    fps_assumed = 15
    buffer_ids = [0, 1, 2]
    for cam in range(5):
        for bid in buffer_ids:
            frame_buffers[(cam, bid)] = deque(maxlen=fps_assumed * buffer_duration)
            buffer_start_times[(cam, bid)] = None

    while True:
        if frame_analysis_queue.empty():
            time.sleep(0.01)
            continue

        frame, fps, camera_index = frame_analysis_queue.get()
        now = time.time()

        if not detect_people(frame, camera_index):
            for bid in buffer_ids:
                buffer_start_times[(camera_index, bid)] = None
            continue

        for bid in buffer_ids:
            key = (camera_index, bid)
            offset = bid * step
            start_time = buffer_start_times.get(key)
            if start_time is None or now - start_time > buffer_duration:
                buffer_start_times[key] = now
                frame_buffers[key].clear()
            frame_buffers[key].append(frame)

        analyze_buffers(fps, camera_index)

def process_video_stream(camera_index=0):
    print("🎩 process_video_stream() 실행중!")
    if USE_VIDEO_FILE:
        source = VIDEO_FILE_PATH
        cap = cv2.VideoCapture(source)
    else:
        source = camera_index
        cap = cv2.VideoCapture(source, cv2.CAP_ANY)

    if not cap.isOpened():
        print("❌ 비디오 열기 실패!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 15

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_analysis_queue.put((frame.copy(), fps, camera_index))

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

        # ✅ 영상 파일일 때는 빠르게 처리 (sleep 제거)
        if not USE_VIDEO_FILE:
            time.sleep(1 / fps if fps > 0 else 1 / 15)

    cap.release()


def get_latest_video_url():
    print("🔗 get_latest_video_url() 실행중!")
    global latest_video_url
    return JSONResponse({"video_url": latest_video_url if latest_video_url else None})

def get_available_cameras():
    print("📷 get_available_cameras() 실행중!")
    available = []
    index = 0
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if not cap.isOpened():
            break
        available.append(f"CAM{index+1}")
        cap.release()
        index += 1
    return available

def get_alerts():
    print("🚨 get_alerts() 실행중!")
    return {"alerts": list(alerts)}
