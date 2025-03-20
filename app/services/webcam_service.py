import cv2
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

def get_available_cameras():
    """ 🔍 사용 가능한 카메라 목록 반환 """
    available_cameras = []
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        available_cameras.append(f"CAM{index+1}")
        cap.release()
        index += 1

    return available_cameras

def generate_video_filename(phone, zip_code, cam_name):
    """ 🎥 사용자 정보 기반 녹화 파일명 생성 """
    base_filename = f"{phone}_{zip_code}_{cam_name}"
    index = 0
    while True:
        video_filename = f"{base_filename}_{index}.mp4"
        video_path = os.path.join(RECORDINGS_DIR, video_filename)
        if not os.path.exists(video_path):
            return video_path
        index += 1  # 파일이 존재하면 다음 번호 사용

def detect_people(frame):
    """ 🎯 YOLOv5를 사용하여 사람 감지 """
    results = model(frame)
    detected = False

    for result in results:
        for box in result.boxes.data:
            class_id = int(box[5])  # 클래스 ID (사람인지 확인)
            confidence = float(box[4])  # 신뢰도 값

            if model.names[class_id] == "person" and confidence > 0.5:
                detected = True

    return detected

def start_recording_with_metadata(camera_index, phone, zip_code, cam_name):
    """✅ 사용자 정보 기반으로 5초 동안 녹화된 영상 저장"""
    global recording

    if recording.get(cam_name, False):
        print(f"⏳ 이미 녹화 중: {cam_name}")
        return

    # ✅ OpenCV로 카메라 열기 (webcam_service에서 관리)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ 카메라를 열 수 없음: {cam_name}")
        return

    recording[cam_name] = True  # ✅ 녹화 시작

    video_path = generate_video_filename(phone, zip_code, cam_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

    print(f"📌 녹화 정보: {phone}, {zip_code}, {cam_name}")
    print(f"📌 저장될 파일명: {video_path}")

    start_time = time.time()
    while time.time() - start_time < 5:
        success, frame = cap.read()
        if not success:
            break
        out.write(frame)

    out.release()
    cap.release()  # ✅ 카메라 종료
    recording[cam_name] = False  # ✅ 녹화 종료
    print(f"🎥 녹화 완료: {video_path}")


def generate_frames(camera="CAM1"):
    """ 📹 실시간 스트리밍 + YOLOv5 감지 + 녹화 """
    
    # 1️⃣ **카메라 유효성 검증 (get_available_cameras)**
    available_cameras = get_available_cameras()
    if not available_cameras:
        print("❌ 사용 가능한 카메라가 없습니다!")
        return

    # 2️⃣ **올바른 카메라 인덱스 변환**
    if not camera or camera in ["null", "None", ""]:
        camera = available_cameras[0]  # 기본값으로 첫 번째 카메라 선택

    try:
        if isinstance(camera, int):
            camera_index = camera
        elif isinstance(camera, str) and "CAM" in camera:
            camera_index = int(camera.replace("CAM", "")) - 1
        else:
            print(f"❌ 잘못된 카메라 값: {camera}")
            return

    except ValueError as e:
        print(f"❌ 카메라 인덱스 변환 오류: {e}")
        return

    # 3️⃣ **카메라 정상 작동 여부 확인**
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ 카메라를 열 수 없음: {camera}")
        return

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # ✅ YOLOv5를 적용하여 사람 감지
            try:
                if detect_people(frame):
                    alerts.add(camera)

                    # ✅ 현재 녹화 중이 아니라면 녹화 시작
                    if not recording.get(camera, False):
                        phone = "default"
                        zip_code = "00000"
                        cam_name = camera

                        # ✅ 녹화 시작 전 정보 출력
                        print(f"📌 녹화 정보: {phone}, {zip_code}, {cam_name}")

                        threading.Thread(target=start_recording_with_metadata, args=(camera_index, phone, zip_code, cam_name)).start()
            
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
        print(f"🔴 스트리밍 종료: {camera}")

def get_video_path(camera_index):
    """ 🚨 녹화된 비디오 경로 반환 """
    if camera_index in video_paths and os.path.exists(video_paths[camera_index]):
        return FileResponse(video_paths[camera_index], media_type="video/mp4")
    else:
        return JSONResponse({"error": "녹화된 비디오 없음"}, status_code=404)

def get_alerts():
    """ 🚨 감지된 카메라 목록 반환 """
    return {"alerts": list(alerts)}
