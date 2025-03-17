import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def get_env_value(key, default=None):
    """ 환경 변수 값을 가져오며, 리스트 형태도 지원 """
    value = os.getenv(key, default)
    if value and value.startswith("["):
        return [v.strip() for v in value.strip("[]").split(",")]
    return value

class Config:
    # 데이터베이스 설정
    DB_HOST = get_env_value("DB_HOST")

    # JWT 설정
    JWT_SECRET = get_env_value("JWT_SECRET")
    JWT_EXPIRES_SEC = int(get_env_value("JWT_EXPIRES_SEC", 259200))

    # Bcrypt 설정
    BCRYPT_SALT_ROUNDS = int(get_env_value("BCRYPT_SALT_ROUNDS", 12))

    # 서비스 포트 설정
    USER_FRONT = int(get_env_value("USER_FRONT", 3000))
    USER_BACK = int(get_env_value("USER_BACK", 8080))
    USER_FASTAPI = int(get_env_value("USER_FASTAPI", 8000))
    POLICE_FRONT = int(get_env_value("POLICE_FRONT", 3100))
    POLICE_BACK = int(get_env_value("POLICE_BACK", 9000))
    
    # 기본 webcam index
    WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX", 0))  

    # API 설정
    API_KEY = get_env_value("API_KEY")
    API_SECRET = get_env_value("API_SECRET")
    MY_NUMBER = get_env_value("MY_NUMBER")

config = Config()
