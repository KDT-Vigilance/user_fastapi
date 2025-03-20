from fastapi import FastAPI
from app.routes.webcam_routes import router as webcam_router
from app.routes.alert_routes import router as alert_router
from fastapi.middleware.cors import CORSMiddleware
import logging
import time

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 서버 준비 확인 플래그
server_ready = False

@app.on_event("startup")
async def startup_event():
    global server_ready
    logging.info("⏳ 서버 시작 중...")
    time.sleep(2)  # ✅ 서버가 안정적으로 실행될 시간을 주기 위해 대기
    server_ready = True
    logging.info("✅ 서버가 정상적으로 시작되었습니다.")

# 🚀 라우터 등록
app.include_router(webcam_router, prefix="/video", tags=["Webcam"])
app.include_router(alert_router, prefix="/alerts", tags=["Alerts"])

# 🏠 기본 엔드포인트 (FastAPI 정상 실행 확인용)
@app.get("/")
def read_root():
    if not server_ready:
        return {"message": "⏳ 서버가 아직 시작되지 않았습니다. 잠시 후 다시 시도해주세요."}
    return {"message": "✅ FastAPI 서버 실행 중!"}
