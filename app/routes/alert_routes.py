from fastapi import APIRouter
from app.services.alert_service import get_alerts

router = APIRouter()

@router.get("/")
def alerts():
    """감지된 카메라 목록 반환"""
    alerts = get_alerts()
    return {"alerts": alerts}
