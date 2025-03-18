import cv2
import torch
import time
import threading
import os
from ultralytics import YOLO
from fastapi.responses import FileResponse, JSONResponse

# 🔹 YOLOv5 모델 로드
model = YOLO("yolov5s.pt")

recording = {}  # ✅ 카메라별 녹화 상태 관리
video_paths = {}  # ✅ 녹화된 비디오 경로 저장
alerts = set()  # 🚨 감지된 카메라 목록
lock = threading.Lock()  # 🔒 스레드 동기화 (녹화 충돌 방지)

# 🔹 저장 폴더 생성
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

def get_next_video_filename():
    """🎥 recordings 폴더에서 다음 저장할 파일 번호 찾기"""
    files = [f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".mp4")]
    numbers = sorted([int(f.split(".")[0]) for f in files if f.split(".")[0].isdigit()])
    next_number = numbers[-1] + 1 if numbers else 0
    return os.path.join(RECORDINGS_DIR, f"{next_number}.mp4")

def detect_people(frame):
    """ YOLOv5를 사용하여 사람 감지 """
    results = model(frame)  # YOLOv5 실행
    detected = False

    for result in results:
        for box in result.boxes.data:
            class_id = int(box[5])  # 클래스 ID (사람인지 확인)
            confidence = float(box[4])  # 신뢰도 값

            if model.names[class_id] == "person" and confidence > 0.5:
                detected = True

    return detected

def start_recording(cap, camera_index):
    """ ✅ 5초 동안 녹화된 영상 저장 """
    global recording, video_paths

    if recording.get(camera_index, False):
        return  # 이미 녹화 중이면 중복 실행 방지

    recording[camera_index] = True  # 녹화 시작
    video_filename = get_next_video_filename()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
    video_paths[camera_index] = video_filename  # 녹화된 영상 경로 저장

    start_time = time.time()
    while time.time() - start_time < 5:
        success, frame = cap.read()
        if not success:
            break
        out.write(frame)  # ✅ 지속적으로 프레임 추가

    out.release()
    recording[camera_index] = False  # 녹화 종료
    print(f"🎥 녹화 완료: {video_filename}")

def get_available_cameras():
    """ 사용 가능한 카메라 목록 반환 """
    available_cameras = []
    index = 0
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if not cap.isOpened():
            break
        available_cameras.append(f"CAM{index+1}")
        print(f"✅ 카메라 감지됨! {available_cameras[-1]}")
        cap.release()
        index += 1

    return available_cameras if available_cameras else ["CAM1"]  # ✅ 기본값 반환

def generate_frames(camera_index=0):
    """ 📹 실시간 스트리밍 (카메라 유지) + YOLOv5 감지 + 녹화 """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)

    if not cap.isOpened():
        print(f"❌ 스트리밍용 카메라를 열 수 없음: CAM{camera_index+1}")
        return

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print(f"❌ 프레임을 가져오지 못함: CAM{camera_index+1}")
                break

            # ✅ YOLOv5를 적용하여 사람 감지
            try:
                if detect_people(frame):
                    alerts.add(f"CAM{camera_index+1}")  # 🚨 감지된 카메라 추가
                    threading.Thread(target=start_recording, args=(cap, camera_index)).start()  # 🎥 녹화 시작
            except Exception as e:
                print(f"❌ YOLO 감지 중 오류 발생: {e}")

            # ✅ 프레임을 JPEG로 변환하여 스트리밍
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print(f"❌ 스트리밍 중 오류 발생: {e}")

    finally:
        cap.release()
        print(f"🔴 스트리밍 종료: CAM{camera_index+1}")

def get_video_path(camera_index):
    """ 🚨 녹화된 비디오 경로 반환 """
    if camera_index in video_paths and os.path.exists(video_paths[camera_index]):
        print(f"✅ 녹화된 영상 반환: {video_paths[camera_index]}")
        return FileResponse(video_paths[camera_index], media_type="video/mp4")
    else:
        print(f"❌ 녹화된 비디오 없음: CAM{camera_index+1}")
        return JSONResponse({"error": "녹화된 비디오 없음"}, status_code=404)

def get_alerts():
    """ 🚨 감지된 카메라 목록 반환 """
    return {"alerts": list(alerts)}
