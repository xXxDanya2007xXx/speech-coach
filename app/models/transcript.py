from pydantic import BaseModel
from typing import List


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    text: str
    segments: List[TranscriptSegment]
