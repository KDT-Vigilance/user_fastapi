from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import os
from app.services.webcam_service import (
    get_available_cameras, 
    generate_frames, 
    get_video_path, 
    get_alerts, 
    start_recording_with_metadata
)

router = APIRouter()

# 📌 녹화 영상 저장 폴더
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

class CameraInfo(BaseModel):
    phone: str
    zip_code: str
    cam_name: str

@router.get("/cameras")
def list_cameras():
    """사용 가능한 카메라 목록 반환"""
    cameras = get_available_cameras()
    return {"cameras": cameras}

@router.get("/video_feed")
async def video_feed(camera: str = "CAM1"):  # 기본값 CAM1
    """웹캠 실시간 스트리밍"""
    if camera is None or camera == "null":
        camera = "CAM1"  # 🚀 기본값 설정
    try:
        camera_index = int(camera.replace("CAM", "")) - 1
        return StreamingResponse(generate_frames(camera_index), media_type="multipart/x-mixed-replace; boundary=frame")
    except ValueError:
        raise HTTPException(status_code=400, detail="잘못된 카메라 값입니다.")


@router.get("/recorded_video")
def recorded_video(camera: str = "CAM1"):
    """녹화된 비디오 제공"""
    camera_index = int(camera.replace("CAM", "")) - 1
    return get_video_path(camera_index)

@router.get("/alerts", include_in_schema=False)
@router.get("/alerts/")
def get_alert_list():
    """🚨 감지된 카메라 목록 반환"""
    return get_alerts()


# ✅ FastAPI에서 Node.js로부터 유저 정보 받아서 `services`로 녹화 요청
@router.post("/store_camera_info")
async def store_camera_info(data: CameraInfo, background_tasks: BackgroundTasks):
    """📌 Node.js에서 받은 데이터로 영상 저장 요청"""
    try:
        # ✅ 받은 데이터 확인
        print("📌 FastAPI에서 받은 데이터:", data.dict())

        # ✅ phone, zip_code, cam_name이 `None`이 아닌지 확인하고 문자열 변환
        phone = str(data.phone) if data.phone else "unknown"
        zip_code = str(data.zip_code) if data.zip_code else "00000"
        cam_name = str(data.cam_name) if data.cam_name else "CAM1"

        if not phone or not zip_code or not cam_name:
            print("❌ 필수 데이터가 누락됨:", phone, zip_code, cam_name)
            raise HTTPException(status_code=400, detail="phone, zip_code, cam_name이 필요합니다.")

        # ✅ `cam_name`을 `camera_index`로 변환
        camera_index = int(cam_name.replace("CAM", "")) - 1

        print(f"📌 녹화 요청 준비: {phone}, {zip_code}, {cam_name} (카메라 인덱스: {camera_index})")

        # ✅ `background_tasks.add_task()`에서 camera_index만 서비스로 전달
        background_tasks.add_task(start_recording_with_metadata, camera_index, phone, zip_code, cam_name)

        print("✅ 녹화 요청 성공:", phone, zip_code, cam_name)
        return {"message": "녹화 요청됨"}

    except Exception as e:
        print("❌ 서버 오류 발생:", str(e))
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")
