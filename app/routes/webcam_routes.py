from fastapi import APIRouter
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from app.services.webcam_service import (
    get_available_cameras,
    process_video_stream as generate_frames,
    get_latest_video_url,
    get_alerts,
    USE_VIDEO_FILE,
    VIDEO_FILE_PATH  # 영상 경로도 가져와야 함
)

router = APIRouter()

@router.get("/cameras")
def list_cameras():
    """✅ 사용 가능한 카메라 목록 반환"""
    return {"cameras": get_available_cameras()}

@router.get("/video_feed")
async def video_feed(camera: str = "CAM1"):
    from app.services.webcam_service import camera_mapping

    if USE_VIDEO_FILE:
        return StreamingResponse(generate_frames(0), media_type="multipart/x-mixed-replace; boundary=frame")

    available_cameras = get_available_cameras()
    if camera not in available_cameras:
        print(f"⚠️ 카메라 {camera} 없음 → 기본값 CAM1 사용")
        camera = "CAM1"

    # ✅ camera_mapping 활용
    camera_index = camera_mapping.get(camera, 0)
    return StreamingResponse(generate_frames(camera_index), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/video_mode")
def get_video_mode():
    """✅ 현재 실행 모드 반환 (프론트에서 video / img 선택용)"""
    return {"USE_VIDEO_FILE": USE_VIDEO_FILE}

@router.get("/latest_video")
async def fetch_latest_video():
    """✅ 가장 최근 S3에 업로드된 비디오 URL 반환 (값 없으면 `null`)"""
    return get_latest_video_url()

@router.get("/alerts", include_in_schema=False)
def get_alert_list():
    """🚨 감지된 카메라 목록 반환"""
    return get_alerts()

@router.get("/mode")
def get_streaming_mode():
    """✅ 현재 스트리밍 모드 반환"""
    return {"use_video_file": USE_VIDEO_FILE}
