from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.services.webcam_service import (
    get_available_cameras, 
    generate_frames, 
    get_latest_video_url, 
    get_alerts
)

router = APIRouter()

@router.get("/cameras")
def list_cameras():
    """✅ 사용 가능한 카메라 목록 반환"""
    return {"cameras": get_available_cameras()}

@router.get("/video_feed")
async def video_feed(camera: str = "CAM1"):
    """✅ 실시간 웹캠 스트리밍"""
    available_cameras = get_available_cameras()
    if camera not in available_cameras:
        print(f"⚠️ 카메라 {camera}가 없음 → 기본값 CAM1 사용")
        camera = "CAM1"

    camera_index = int(camera.replace("CAM", "")) - 1
    return StreamingResponse(generate_frames(camera_index), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/latest_video")
async def fetch_latest_video():
    """✅ 가장 최근 S3에 업로드된 비디오 URL 반환 (값 없으면 `null`)"""
    return get_latest_video_url()

@router.get("/alerts", include_in_schema=False)
def get_alert_list():
    """🚨 감지된 카메라 목록 반환"""
    return get_alerts()