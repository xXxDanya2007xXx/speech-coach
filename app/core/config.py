from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Путь к ffmpeg (по умолчанию просто "ffmpeg" из PATH)
    ffmpeg_path: str = Field(default="ffmpeg", alias="FFMPEG_PATH")

    # Настройки локального Whisper (faster-whisper)
    whisper_model: str = Field(
        default="small", alias="WHISPER_MODEL"
    )  # варианты: tiny, base, small, medium, large-v3
    whisper_device: str = Field(
        default="cpu", alias="WHISPER_DEVICE"
    )  # cpu или cuda
    whisper_compute_type: str = Field(
        default="int8", alias="WHISPER_COMPUTE_TYPE"
    )  # int8, int8_float16, float16, float32

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
