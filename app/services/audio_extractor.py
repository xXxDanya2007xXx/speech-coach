import subprocess
from pathlib import Path
from typing import Protocol

from app.core.config import settings


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
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ]
        subprocess.run(cmd, check=True)
