import logging
from pathlib import Path
from typing import Protocol, List

from faster_whisper import WhisperModel

from app.core.config import settings
from app.models.transcript import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> Transcript:
        ...


class LocalWhisperTranscriber:
    """
    Использует локальную модель Whisper через faster-whisper.
    Модель скачивается при первом запуске (несколько сотен МБ).
    """

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ):
        self.model_size = model_size or settings.whisper_model
        self.device = device or settings.whisper_device
        self.compute_type = compute_type or settings.whisper_compute_type

        logger.info(f"Loading Whisper model: {
                    self.model_size} on {self.device}")
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info(f"Whisper model loaded successfully")

    def transcribe(self, audio_path: Path) -> Transcript:
        logger.info(f"Transcribing audio: {audio_path}")

        # segments — генератор, info — объект с метаданными
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,  # Включить фильтрацию голосовой активности
        )

        segments: List[TranscriptSegment] = []
        texts: List[str] = []

        for seg in segments_iter:
            segments.append(
                TranscriptSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text,
                )
            )
            texts.append(seg.text)

        full_text = " ".join(texts).strip()

        logger.info(f"Transcription complete: {len(segments)} segments, {
                    len(full_text)} characters")

        return Transcript(text=full_text, segments=segments)
