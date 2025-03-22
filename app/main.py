from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.webcam_routes import router as webcam_router
from app.routes.alert_routes import router as alert_router
from contextlib import asynccontextmanager
import threading
import logging

from app.services.webcam_service import background_analyzer  # 백그라운드 함수 import 필요!

logging.basicConfig(level=logging.INFO)

# ✅ lifespan 먼저 정의!
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("💡 background_analyzer 실행 시작")
    threading.Thread(target=background_analyzer, daemon=True).start()
    yield

# ✅ 그 다음에 FastAPI 선언
app = FastAPI(lifespan=lifespan)

# 🔐 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🚀 라우터 등록
app.include_router(webcam_router, prefix="/video", tags=["Webcam"])
app.include_router(alert_router, prefix="/alerts", tags=["Alerts"])

# 🏠 기본 확인용 엔드포인트
@app.get("/")
def read_root():
    return {"message": "User FastAPI is running!"}
