from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.services.webcam_service import get_available_cameras, generate_frames

router = APIRouter()

@router.get("/cameras")
def list_cameras():
    """사용 가능한 카메라 목록 반환"""
    cameras = get_available_cameras()
    return {"cameras": cameras}

@router.get("/video_feed")
async def video_feed():
    """웹캠 실시간 스트리밍"""
    return StreamingResponse(generate_frames(0), media_type="multipart/x-mixed-replace; boundary=frame")
