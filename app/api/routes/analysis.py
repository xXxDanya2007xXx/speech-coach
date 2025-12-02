from fastapi import APIRouter, UploadFile, File, Depends, HTTPException

from app.api.deps import get_speech_pipeline
from app.models.analysis import AnalysisResult
from app.services.pipeline import SpeechAnalysisPipeline

router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_video(
    file: UploadFile = File(...),
    pipeline: SpeechAnalysisPipeline = Depends(get_speech_pipeline),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    try:
        result = await pipeline.analyze_upload(file)
        return result
    except Exception as e:
        # TODO: логирование
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze video: {e}")
