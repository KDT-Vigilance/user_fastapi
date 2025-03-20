import uuid
import cv2
import torch
import time
import threading
import os
import boto3
from ultralytics import YOLO
from fastapi.responses import JSONResponse
from config import config

# 🔹 YOLOv5 모델 로드
model = YOLO("yolov5su.pt")

recording = {}  # ✅ 카메라별 녹화 상태 관리
video_paths = {}  # ✅ 녹화된 비디오 경로 저장 (S3 URL)
latest_video_url = None  # ✅ 가장 최근 저장된 S3 URL
alerts = set()  # 🚨 감지된 카메라 목록
lock = threading.Lock()  # 🔒 스레드 동기화 (녹화 충돌 방지)
active_cameras = {}  # 🔹 현재 열린 카메라 저장

def get_alerts():
    """ 🚨 감지된 카메라 목록 반환 """
    return {"alerts": list(alerts)}

# 🔹 S3 클라이언트 설정
s3_client = boto3.client(
    "s3",
    aws_access_key_id=config.ACCESS_KEY,
    aws_secret_access_key=config.SECRET_KEY,
    region_name=config.BUCKET_REGION,
)

def upload_to_s3(file_path, camera_index):
    """🎥 녹화된 파일을 S3에 업로드 후 URL 반환"""
    global latest_video_url  # 🔹 최신 URL 변수 사용

    try:
        filename = os.path.basename(file_path)
        s3_key = f"{config.BUCKET_DIRECTORY}/{filename}"

        # 🔹 S3 업로드 (비동기 처리 X → 바로 실행)
        with open(file_path, "rb") as file:
            s3_client.upload_fileobj(
                file,
                config.BUCKET_NAME,
                s3_key,
                ExtraArgs={
                    'ACL': 'public-read',
                    'ContentDisposition': 'inline',  # ✅ 이 부분 추가 (자동 다운로드 방지)
                    'ContentType': 'video/mp4'  # ✅ 파일 타입 명시 (MP4 영상)
                }
            )

        # ✅ S3 URL 생성 후 저장
        s3_url = f"https://{config.BUCKET_NAME}.s3.{config.BUCKET_REGION}.amazonaws.com/{s3_key}"
        video_paths[camera_index] = s3_url  # 🔹 카메라별 S3 URL 저장
        latest_video_url = s3_url  # ✅ 최신 URL 저장
        print(f"✅ S3 업로드 성공: {s3_url}")

        # 🔹 업로드 후 로컬 파일 삭제
        os.remove(file_path)
        return s3_url
    except Exception as e:
        print(f"❌ S3 업로드 실패: {e}")
        return None


def get_next_video_filename():
    """🎥 타임스탬프 + UUID 앞 8자리 조합한 파일명 생성"""
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}.mp4"
    return os.path.join("recordings", filename)

def start_recording(cap, camera_index):
    """ ✅ 5초 동안 녹화 후 S3 업로드 """
    global recording, latest_video_url

    if recording.get(camera_index, False):
        return  # 이미 녹화 중이면 중복 실행 방지

    recording[camera_index] = True
    video_filename = get_next_video_filename()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

    start_time = time.time()
    while time.time() - start_time < 5:
        success, frame = cap.read()
        if not success:
            break
        out.write(frame)

    out.release()
    recording[camera_index] = False
    print(f"🎥 녹화 완료: {video_filename}")

    # 🔹 S3 업로드 후 URL 반환
    s3_url = upload_to_s3(video_filename, camera_index)
    if not s3_url:
        print(f"❌ S3 업로드 실패: {video_filename}는 로컬에 남겨둠.")
    else:
        latest_video_url = s3_url  # ✅ 가장 최근 저장된 URL 업데이트
        print(f"✅ 녹화된 영상의 S3 URL: {s3_url}")  # ✅ S3 URL 출력

def detect_people(frame):
    """ YOLOv5를 사용하여 사람 감지 """
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
    """ 📹 실시간 스트리밍 + YOLO 감지 + 녹화 """
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

            # ✅ YOLO 감지 후 녹화 시작
            try:
                if detect_people(frame):
                    alerts.add(f"CAM{camera_index+1}")
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
    """ ✅ 가장 최근에 저장된 S3 URL 반환 """
    global latest_video_url

    if not latest_video_url:
        return JSONResponse({"video_url": None})  # ✅ URL이 없으면 `null` 반환

    print(f"✅ 최신 비디오 URL 반환: {latest_video_url}")
    return JSONResponse({"video_url": latest_video_url})  # ✅ URL이 있으면 반환


def get_available_cameras():
    """ 사용 가능한 카메라 목록 반환 """
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
    """FastAPI 종료 시, 열린 카메라 리소스 해제"""
    global active_cameras
    for index, cap in active_cameras.items():
        if cap.isOpened():
            cap.release()
            print(f"🔴 카메라 해제 완료: CAM{index+1}")
    active_cameras = {}
