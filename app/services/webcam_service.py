import uuid
import cv2
import torch
import time
import threading
import os
import boto3
import numpy as np
from ultralytics import YOLO
from fastapi.responses import JSONResponse
from config import config
from cnn3d import Deeper3DCNN_Gray  # 🔹 3D CNN 모델 임포트

# 🔹 YOLOv5 모델 로드
model = YOLO("yolov5su.pt")

# 🔹 3D CNN 모델 로드
cnn_model = Deeper3DCNN_Gray()
cnn_model.load_state_dict(torch.load("user_fastapi/app/services/deeper3dcnn_gray3.pth", map_location='cpu'))
cnn_model.eval()

recording = {}
video_paths = {}
latest_video_url = None
alerts = set()
lock = threading.Lock()
active_cameras = {}

def get_alerts():
    return {"alerts": list(alerts)}

s3_client = boto3.client(
    "s3",
    aws_access_key_id=config.ACCESS_KEY,
    aws_secret_access_key=config.SECRET_KEY,
    region_name=config.BUCKET_REGION,
)

def upload_to_s3(file_path, camera_index):
    global latest_video_url
    try:
        filename = os.path.basename(file_path)
        s3_key = f"{config.BUCKET_DIRECTORY}/{filename}"
        with open(file_path, "rb") as file:
            s3_client.upload_fileobj(
                file,
                config.BUCKET_NAME,
                s3_key,
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
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}.mp4"
    return os.path.join("recordings", filename)

def start_recording(cap, camera_index):
    global recording, latest_video_url
    if recording.get(camera_index, False):
        return
    recording[camera_index] = True
    video_filename = get_next_video_filename()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

    start_time = time.time()
    frames = []
    while time.time() - start_time < 5:
        success, frame = cap.read()
        if not success:
            break
        out.write(frame)
        frames.append(frame)

    out.release()
    recording[camera_index] = False
    print(f"🎥 녹화 완료: {video_filename}")

    # 3D CNN 추가: 행동 인식
    if len(frames) >= 16:
        clip = preprocess_frames(frames[:16])
        with torch.no_grad():
            output = cnn_model(clip)
            pred = torch.argmax(output, dim=1).item()
            if pred == 1:
                s3_url = upload_to_s3(video_filename, camera_index)
                latest_video_url = s3_url
                alerts.add(f"CAM{camera_index+1}")
                print(f"🚨 폭력 감지됨: {s3_url}")
            else:
                print("❌ 폭력 아님으로 판단")
    else:
        print("❌ 충분한 프레임이 없음")

def preprocess_frames(frames):
    frames = [cv2.resize(f, (112, 112)) for f in frames]
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    gray_frames = np.stack(gray_frames, axis=0)  # (16, 112, 112)
    gray_frames = np.expand_dims(gray_frames, axis=0)  # (1, 16, 112, 112)
    gray_frames = torch.FloatTensor(gray_frames) / 255.0  # Normalize
    return gray_frames

def detect_people(frame):
    results = model(frame)
    detected = False
    for result in results:
        for box in result.boxes.data:
            class_id = int(box[5])
            confidence = float(box[4])
            if model.names[class_id] == "person" and confidence > 0.3:
                detected = True
    return detected

def generate_frames(camera_index=0):
    global active_cameras
    if camera_index in active_cameras:
        active_cameras[camera_index].release()
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"❌ 카메라를 열 수 없음: CAM{camera_index+1}")
        return
    active_cameras[camera_index] = cap

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            try:
                if detect_people(frame):
                    threading.Thread(target=start_recording, args=(cap, camera_index)).start()
            except Exception as e:
                print(f"❌ YOLO 감지 오류: {e}")
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    finally:
        cap.release()
        active_cameras.pop(camera_index, None)

def get_latest_video_url():
    global latest_video_url
    if not latest_video_url:
        return JSONResponse({"video_url": None})
    return JSONResponse({"video_url": latest_video_url})

def get_available_cameras():
    available_cameras = []
    index = 0
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if not cap.isOpened():
            break
        available_cameras.append(f"CAM{index+1}")
        cap.release()
        index += 1
    return available_cameras if available_cameras else ["CAM1"]

def cleanup_cameras():
    global active_cameras
    for index, cap in active_cameras.items():
        if cap.isOpened():
            cap.release()
    active_cameras = {}
