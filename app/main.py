from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.webcam_routes import router as webcam_router
from app.routes.alert_routes import router as alert_router
from contextlib import asynccontextmanager
import threading
import logging

from app.services.webcam_service import background_analyzer, start_all_cameras, process_video_stream
from config import config

# ✅ True일 경우 파일 기반 실행으로 처리
USE_VIDEO_FILE = True  # 여기가 True여야 함

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("💡 background_analyzer 실행 시작")

    if not getattr(app.state, "initialized", False):
        app.state.initialized = True  # ✅ reload 시 중복 실행 방지

        # ✅ CNN 분석 스레드 실행
        threading.Thread(target=background_analyzer, daemon=True).start()

        if USE_VIDEO_FILE:
            print("[INFO] 파일 기반 실행: 영상 스트리밍 스레드 시작")
            # ✅ 파일 기반 실행 → process_video_stream을 리스트로 강제 실행
            threading.Thread(target=lambda: list(process_video_stream(0)), daemon=True).start()
        else:
            print("[INFO] 실시간 카메라 실행: start_all_cameras() 호출")
            start_all_cameras()

    yield


# ✅ FastAPI 앱 생성
app = FastAPI(lifespan=lifespan)

# 🔐 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 라우터 등록
app.include_router(webcam_router, prefix="/video", tags=["Webcam"])
app.include_router(alert_router, prefix="/alerts", tags=["Alerts"])

# ✅ 기본 헬스체크
@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}
