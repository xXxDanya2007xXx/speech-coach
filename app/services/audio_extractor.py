import subprocess
import logging
from pathlib import Path
from typing import Protocol, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class AudioExtractor(Protocol):
    def extract(self, video_path: Path, audio_path: Path) -> None:
        ...


class FfmpegAudioExtractor:
    def __init__(self, ffmpeg_path: str | None = None):
        self.ffmpeg_path = ffmpeg_path or settings.ffmpeg_path

    def extract(self, video_path: Path, audio_path: Path) -> None:
        """
        Извлекает моно WAV 16kHz из видео-файла с помощью ffmpeg.
        """
        cmd = [
            self.ffmpeg_path,
            "-y",  # Перезаписать без подтверждения
            "-i", str(video_path),
            "-vn",  # Без видео
            "-acodec", "pcm_s16le",
            "-ar", "16000",  # Частота дискретизации
            "-ac", "1",  # Моно
            "-hide_banner",  # Скрыть баннер ffmpeg
            "-loglevel", "error",  # Только ошибки
            str(audio_path),
        ]

        logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")

        try:
            # Используем subprocess.run с правильными параметрами
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,  # Захватываем вывод
                text=True,
                timeout=60  # Таймаут 60 секунд
            )

            if result.stderr:
                logger.warning(f"FFmpeg warnings: {result.stderr}")

            logger.info(f"Audio extracted successfully: {audio_path}")

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed with code {e.returncode}: {e.stderr}")
            raise RuntimeError(f"Failed to extract audio: {e.stderr}")
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timeout (60 seconds)")
            raise RuntimeError("Audio extraction timeout")
