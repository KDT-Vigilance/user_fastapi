from fastapi import FastAPI
from app.routes.webcam_routes import router as webcam_router
from app.routes.alert_routes import router as alert_router
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS 설정 (보안 강화 가능)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 보안이 필요하면 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🚀 라우터 등록 (경로 일관성 유지)
app.include_router(webcam_router, prefix="/video", tags=["Webcam"])
app.include_router(alert_router, prefix="/alerts", tags=["Alerts"])

# 🏠 기본 엔드포인트 (FastAPI 정상 실행 확인용)
@app.get("/")
def read_root():
    return {"message": "User FastAPI is running!"}
