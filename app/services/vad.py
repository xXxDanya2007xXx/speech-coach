import wave
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


def detect_speech_regions_webrtc(audio_path: Path, mode: int = 3) -> List[Tuple[float, float]]:
    """Detect speech regions using webrtcvad. Returns list of (start, end) seconds.
    If webrtcvad not installed or audio format unsupported, returns empty list.
    """
    try:
        import webrtcvad
    except Exception as e:
        logger.debug("webrtcvad not available: %s", e)
        return []

    try:
        with wave.open(str(audio_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()

            if n_channels != 1 or sampwidth != 2 or framerate not in (8000, 16000, 32000, 48000):
                logger.debug("webrtcvad: unsupported audio format: channels=%d sampwidth=%d framerate=%d", n_channels, sampwidth, framerate)
                return []

            frames = wf.readframes(n_frames)
    except Exception as e:
        logger.debug("webrtcvad: failed reading wave: %s", e)
        return []

    vad = webrtcvad.Vad(mode)

    frame_duration_ms = 30  # 10, 20, 30 supported
    bytes_per_sample = 2
    frame_length = int(framerate * (frame_duration_ms / 1000.0))
    frame_bytes = frame_length * bytes_per_sample

    regions = []
    is_speech = False
    speech_start = 0.0

    i = 0
    total_bytes = len(frames)
    while i + frame_bytes <= total_bytes:
        frame = frames[i: i + frame_bytes]
        t_start = (i // bytes_per_sample) / framerate
        try:
            vad_flag = vad.is_speech(frame, framerate)
        except Exception:
            vad_flag = False

        if vad_flag and not is_speech:
            is_speech = True
            speech_start = t_start
        elif not vad_flag and is_speech:
            is_speech = False
            regions.append((speech_start, t_start))

        i += frame_bytes

    # close any open region
    if is_speech:
        regions.append((speech_start, (i // bytes_per_sample) / framerate))

    # merge tiny regions
    merged = []
    for s, e in regions:
        if not merged or s - merged[-1][1] > 0.05:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], e)

    return merged


def detect_speech_regions_pyannote(audio_path: Path, model_name: str = "pyannote/voice-activity-detection") -> List[Tuple[float, float]]:
    """Detect speech regions using pyannote.audio Pipeline if available.
    Returns list of (start, end). If library or model not available returns empty.
    """
    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        logger.debug("pyannote.audio not available: %s", e)
        return []

    try:
        pipeline = Pipeline.from_pretrained(model_name)
        diarization = pipeline(str(audio_path))
        regions = []
        for segment in diarization.get_timeline().support():
            regions.append((segment.start, segment.end))
        return regions
    except Exception as e:
        logger.debug("pyannote detection failed: %s", e)
        return []


def detect_speech_regions(audio_path: Path, use_pyannote: bool, use_webrtc: bool, webrtc_mode: int = 3, pyannote_model: str = None) -> List[Tuple[float, float]]:
    if use_pyannote and pyannote_model:
        regions = detect_speech_regions_pyannote(audio_path, pyannote_model)
        if regions:
            return regions

    if use_webrtc:
        regions = detect_speech_regions_webrtc(audio_path, mode=webrtc_mode)
        if regions:
            return regions

    return []

