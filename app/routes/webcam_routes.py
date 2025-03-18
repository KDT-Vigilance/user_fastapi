from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.services.webcam_service import (
    get_available_cameras, 
    generate_frames, 
    get_video_path, 
    get_alerts
)

router = APIRouter()

@router.get("/cameras")
def list_cameras():
    """사용 가능한 카메라 목록 반환"""
    cameras = get_available_cameras()
    return {"cameras": cameras}

@router.get("/video_feed")
async def video_feed(camera: str = "CAM1"):
    """웹캠 실시간 스트리밍"""
    camera_index = int(camera.replace("CAM", "")) - 1
    return StreamingResponse(generate_frames(camera_index), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/recorded_video")
def recorded_video(camera: str = "CAM1"):
    """녹화된 비디오 제공"""
    camera_index = int(camera.replace("CAM", "")) - 1
    return get_video_path(camera_index)

@router.get("/alerts")
def get_alert_list():
    """🚨 감지된 카메라 목록 반환"""
    return get_alerts()

