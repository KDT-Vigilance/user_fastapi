from fastapi import APIRouter

router = APIRouter()

# ✅ 감지된 경보 목록 반환 (테스트용)
@router.get("/")
def get_alerts():
    """상황 발생한 카메라 목록 반환"""
    return {"alerts": []}  # 기본값 (나중에 상황 발생하면 변경)
