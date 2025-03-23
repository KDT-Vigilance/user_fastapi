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
import threading
from datetime import datetime
from collections import deque
from ultralytics import YOLO
from fastapi.responses import JSONResponse
from config import config
from app.services.cnn3d import Deeper3DCNN_Gray

USE_VIDEO_FILE = True
VIDEO_FILE_PATH = "app/services/test/pulling.mp4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 실행 장치: {device}")
print("[INFO] 현재 모드: 파일 재생 기반" if USE_VIDEO_FILE else "[INFO] 현재 모드: 실시간 카메라")

model = YOLO("yolov8n.pt", verbose=False).to(device)
from ultralytics.utils import LOGGER
LOGGER.setLevel("WARNING")

cnn_model = Deeper3DCNN_Gray().to(device)
cnn_model.load_state_dict(torch.load("app/services/deeper3dcnn_gray3.pth", map_location=device))
cnn_model.eval()

buffer_pred_time = {}      # CNN 분석 중복 방지
PRED_COOLDOWN = 10         # 같은 buffer 재분석 제한 시간
ALERT_COOLDOWN = 5         # 같은 상황 중복 녹화 제한 시간

last_person_detected = {}
buffer_full_printed = {}
frame_buffers = {}
buffer_start_times = {}
video_paths = {}
latest_video_url = None
alerts = {}
active_cameras = {}
last_detection_time = {}
last_upload_time = {}
UPLOAD_COOLDOWN = 5
camera_mapping = {}  


if not os.path.exists("recordings"):
    os.makedirs("recordings")

frame_analysis_queue = queue.Queue()

s3_client = boto3.client(
    "s3",
    aws_access_key_id=config.ACCESS_KEY,
    aws_secret_access_key=config.SECRET_KEY,
    region_name=config.BUCKET_REGION,
)

def upload_to_s3(file_path, camera_index):
    print("[INFO] upload_to_s3() 실행중!")
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
        print(f"[SUCCESS] S3 업로드 성공: {s3_url}")
        os.remove(file_path)
        return s3_url
    except Exception as e:
        print(f"[ERROR] S3 업로드 실패: {e}")
        return None

def get_next_video_filename():
    print("[INFO] get_next_video_filename() 실행중!")
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    return os.path.join("recordings", f"{timestamp}_{unique_id}.mp4")

def sample_16_frames(frames):
    print("[INFO] sample_16_frames() 실행중!")
    if len(frames) < 16:
        return []
    indices = np.linspace(0, len(frames) - 1, 16, dtype=int)
    return [frames[i] for i in indices]

def preprocess_frames(frames):
    print("[INFO] preprocess_frames() 실행중!")
    gray_frames = [cv2.cvtColor(cv2.resize(f, (112, 112)), cv2.COLOR_BGR2GRAY) for f in frames]
    gray_frames = np.stack(gray_frames, axis=0)
    gray_frames = np.expand_dims(gray_frames, axis=0)
    gray_frames = np.expand_dims(gray_frames, axis=1)
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
            if model.names[class_id] == "person" and confidence > 0.2:
                person_detected = True
                break

    if camera_index not in last_person_detected or last_person_detected[camera_index] != person_detected:
        last_person_detected[camera_index] = person_detected
        if person_detected:
            print(f"[DETECT] 사람 감지됨! (CAM{camera_index+1})")
        else:
            print(f"[DETECT] 사람 없음! (CAM{camera_index+1})")

    return person_detected

def save_video(frames, filename):
    print("[INFO] save_video() 실행중!")
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    if len(frames) == 0:
        print("[ERROR] 저장할 프레임이 없습니다.")
        return
    h, w = frames[0].shape[:2]
    print(f"[INFO] 저장할 영상 크기: {w}x{h}, 총 {len(frames)} 프레임")
    out = cv2.VideoWriter(filename, fourcc, 15, (w, h))
    for i, frame in enumerate(frames):
        out.write(frame)
    out.release()

def analyze_buffers(original_fps, camera_index):
    global last_upload_time, buffer_pred_time, last_detection_time

    now = time.time()
    last_upload = last_upload_time.get(camera_index, 0)
    if now - last_upload < UPLOAD_COOLDOWN:
        return

    for buffer_id in [0, 1, 2]:
        key = (camera_index, buffer_id)
        buffer_frames = list(frame_buffers.get(key, []))
        if len(buffer_frames) < 45:
            continue

        # CNN 재분석 제한 시간 체크
        last_pred_time = buffer_pred_time.get(key, 0)
        if now - last_pred_time < PRED_COOLDOWN:
            continue

        print(f"[INFO] {key} 버퍼가 다 찼어요! CNN 분석 시작")
        clip_for_cnn = sample_16_frames(buffer_frames)
        if len(clip_for_cnn) < 16:
            continue

        input_tensor = preprocess_frames(clip_for_cnn)
        with torch.no_grad():
            output = cnn_model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        print(f"[CNN] buffer{buffer_id} → pred={pred}")
        buffer_pred_time[key] = now

        if pred == 1:
            # 중복 감지 방지 (ALERT_COOLDOWN)
            last_detected = last_detection_time.get(camera_index, 0)
            if now - last_detected < ALERT_COOLDOWN:
                print(f"[SKIP] 중복 감지 무시 (buffer{buffer_id}, {now - last_detected:.2f}s 이내)")
                continue

            print(f"[ALERT] 폭력 감지! buffer{buffer_id} 감지됨")
            alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            alerts[f"CAM{camera_index+1}"] = alert_time

            # 인접 버퍼도 함께 저장
            buffers_to_save = [buffer_id]
            if buffer_id == 0:
                buffers_to_save.append(1)
            elif buffer_id == 1:
                buffers_to_save.append(2)

            combined_frames = []
            for bid in buffers_to_save:
                k = (camera_index, bid)
                frames = list(frame_buffers.get(k, []))
                combined_frames.extend(frames)

            print(f"[SAVE] 저장할 buffer들: {buffers_to_save}, 총 프레임 수: {len(combined_frames)}")
            filename = get_next_video_filename()
            save_video(combined_frames, filename)
            upload_to_s3(filename, camera_index)

            # 마지막 감지 시각 업데이트
            last_detection_time[camera_index] = now
            last_upload_time[camera_index] = now

        else:
            # pred=0이면 해당 buffer 전체 삭제
            frame_buffers[key].clear()
            print(f"[CLEAN] pred=0 → buffer{buffer_id} 전체 삭제 완료")


def background_analyzer():
    print("[THREAD] background_analyzer() 백그라운드 분석 시작!")
    step = 2
    buffer_duration = 5
    fps_assumed = 15
    buffer_ids = [0, 1, 2]
    for cam in range(10):
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
    print(f"[STREAM] process_video_stream({camera_index}) 시작!")

    if USE_VIDEO_FILE:
        source = VIDEO_FILE_PATH
        fps = 15  # 영상 파일 기본 FPS (원하면 동적으로 cap.get으로도 가능)
        
        while True:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"[ERROR] 영상 파일 열기 실패: {source}")
                break

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] 영상 재생 완료 → 처음부터 다시 재생")
                    break  # 내부 루프를 빠져나가 외부 while True에서 재생 재시작

                frame_analysis_queue.put((frame.copy(), fps, camera_index))
                _, buffer = cv2.imencode(".jpg", frame)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
                time.sleep(1 / fps)

            cap.release()

    else:
        source = camera_index
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[ERROR] CAM{camera_index} 열기 실패!")
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
            time.sleep(1 / fps if fps > 0 else 1 / 15)
        cap.release()


def get_available_camera_indexes(max_check=10):
    print("[INFO] get_available_camera_indexes() 실행중!")
    available = []
    for index in range(max_check):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"[INFO] CAM{len(available)+1} (index={index}) 사용 가능")
            available.append(index)
        cap.release()
    return available

def start_all_cameras():
    global camera_mapping
    print("[START] start_all_cameras() 시작!")
    if USE_VIDEO_FILE:
        print("[INFO] 파일 모드이므로 카메라 시작 생략")
        return

    indexes = get_available_camera_indexes()
    camera_mapping = {f"CAM{i+1}": idx for i, idx in enumerate(indexes)}
    print(f"[INFO] 감지된 카메라 매핑: {camera_mapping}")
    for cam_name, idx in camera_mapping.items():
        threading.Thread(target=process_video_stream, args=(idx,), daemon=True).start()
        print(f"[INFO] {cam_name} 스트리밍 시작")

def get_available_cameras():
    global camera_mapping
    available = []
    camera_mapping = {}
    for index in range(5):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cam_name = f"CAM{index+1}"
            available.append(cam_name)
            camera_mapping[cam_name] = index
        cap.release()
    return available

def get_latest_video_url():
    global latest_video_url
    return JSONResponse({"video_url": latest_video_url if latest_video_url else None})

def get_alerts():
    return {"alerts": alerts}