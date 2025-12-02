import os
import shutil
import tempfile
from pathlib import Path

from fastapi import UploadFile

from app.services.audio_extractor import AudioExtractor
from app.services.transcriber import Transcriber
from app.services.analyzer import SpeechAnalyzer
from app.models.analysis import AnalysisResult


class SpeechAnalysisPipeline:
    """
    Координирует:
    - приём UploadFile (видео),
    - сохранение во временный файл,
    - извлечение аудио,
    - транскрибацию,
    - анализ.
    """

    def __init__(
        self,
        audio_extractor: AudioExtractor,
        transcriber: Transcriber,
        analyzer: SpeechAnalyzer,
    ):
        self.audio_extractor = audio_extractor
        self.transcriber = transcriber
        self.analyzer = analyzer

    async def analyze_upload(self, file: UploadFile) -> AnalysisResult:
        suffix = Path(file.filename or "video").suffix or ".mp4"

        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_video_path = Path(tmp_video.name)
        tmp_video.close()

        await self._save_upload_to_path(file, temp_video_path)

        temp_audio_path = temp_video_path.with_suffix(".wav")

        try:
            # 1) Извлекаем аудио из видео
            self.audio_extractor.extract(temp_video_path, temp_audio_path)
            # 2) Транскрибируем аудио
            transcript = self.transcriber.transcribe(temp_audio_path)
            # 3) Анализируем с учётом пути к аудио (для оценки пауз)
            result = self.analyzer.analyze(
                transcript, audio_path=temp_audio_path)
            return result
        finally:
            # Удаляем временные файлы
            for path in (temp_video_path, temp_audio_path):
                try:
                    if path.exists():
                        os.remove(path)
                except OSError:
                    pass

    @staticmethod
    async def _save_upload_to_path(upload: UploadFile, dst: Path) -> None:
        upload.file.seek(0)
        with dst.open("wb") as out_file:
            shutil.copyfileobj(upload.file, out_file)
        await upload.close()
