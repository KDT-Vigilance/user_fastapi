from app.services.webcam_service import get_available_cameras  
import logging

def get_alerts():
    """ 감지된 모든 카메라 반환 """
    detected_cameras = get_available_cameras()  # ✅ 현재 감지된 카메라 목록 가져오기
    logging.info(f"🔍 현재 감지된 카메라 개수: {len(detected_cameras)}개 → {detected_cameras}")  
    return detected_cameras

