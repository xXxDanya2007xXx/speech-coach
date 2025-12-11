import wave
import struct
import math
from pathlib import Path
from app.services.analyzer import SpeechAnalyzer, MIN_PAUSE_GAP_SEC
from app.core.config import settings
from app.models.transcript import Transcript, TranscriptSegment, WordTiming


def create_test_wav(path: Path, sample_rate=16000):
    # duration 2 seconds
    duration = 2.0
    n_samples = int(sample_rate * duration)
    freq = 400
    amplitude = 3000

    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        frames = []
        for i in range(n_samples):
            t = i / sample_rate
            # tone from 0-0.3 and 0.8-1.0 seconds
            if (0.0 <= t <= 0.3) or (0.8 <= t <= 1.0):
                sample = int(amplitude * math.sin(2 * math.pi * freq * t))
            else:
                sample = 0
            frames.append(struct.pack('<h', sample))

        wf.writeframes(b''.join(frames))


def test_find_fillers_and_pauses(tmp_path):
    # Build transcript
    words = [
        WordTiming(word='Ээээ', start=0.0, end=0.3, confidence=0.95),
        WordTiming(word='привет', start=0.35, end=0.6, confidence=0.9),
        WordTiming(word='ну', start=0.8, end=1.0, confidence=0.9),
    ]

    segments = [
        TranscriptSegment(start=0.0, end=0.6, text='Ээээ привет', words=[words[0], words[1]]),
        TranscriptSegment(start=0.8, end=1.0, text='ну', words=[words[2]]),
    ]

    transcript = Transcript(text='Ээээ привет ну', segments=segments, word_timings=words)

    # create audio file with silence in pause between 0.6 and 0.8
    audio_file = tmp_path / 'test.wav'
    create_test_wav(audio_file)

    analyzer = SpeechAnalyzer()

    timed_data = analyzer._analyze_with_timings(transcript, audio_file)

    # Check fillers detected (Ээээ and ну)
    names = [f.word for f in timed_data.filler_words_detailed]
    assert 'э-э' in names or 'эм' in names or 'мм' in names
    assert 'ну' in names

    # Check pause detected between 0.6 and 0.8 (duration 0.2 < MIN_PAUSE_GAP_SEC may not be counted)
    # But pause between 0.6 and 0.8 is 0.2 sec, which is less than default; we expect at least one pause between 0.6 and 0.8 to be skipped
    # There should be a pause between 0.6 and 0.8 only if gap >= MIN_PAUSE_GAP_SEC
    pauses = timed_data.pauses_detailed
    for p in pauses:
        assert p.duration >= MIN_PAUSE_GAP_SEC

    # There should be at least one pause if a gap > MIN_PAUSE_GAP_SEC exists
    # Here words at 0.35 and 0.8 gap = 0.2 (short) -> maybe no pauses in this test


def test_pause_filtering_with_non_silent_gap(tmp_path):
    # Two words with a gap but the gap contains sound -> should be filtered out
    words = [
        WordTiming(word='привет', start=0.0, end=0.3, confidence=0.95),
        WordTiming(word='следующий', start=1.0, end=1.3, confidence=0.95),
    ]

    segments = [TranscriptSegment(start=0.0, end=0.3, text='привет', words=[words[0]]),
                TranscriptSegment(start=1.0, end=1.3, text='следующий', words=[words[1]])]
    transcript = Transcript(text='привет следующий', segments=segments, word_timings=words)

    audio_file = tmp_path / 'test_sound_in_gap.wav'
    # Build wav with tone in the gap between 0.3 and 1.0
    sample_rate = 16000
    duration = 2.0
    n_samples = int(sample_rate * duration)
    freq = 400
    amplitude = 3000

    import wave as wav
    import struct as st
    with wav.open(str(audio_file), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        frames = []
        for i in range(n_samples):
            t = i / sample_rate
            # tone from 0.3 to 1.0 (which is the gap)
            if 0.3 <= t <= 1.0:
                sample = int(amplitude * math.sin(2 * math.pi * freq * t))
            else:
                sample = 0
            frames.append(st.pack('<h', sample))
        wf.writeframes(b''.join(frames))

    analyzer = SpeechAnalyzer()
    timed_data = analyzer._analyze_with_timings(transcript, audio_file)

    # Because there's audio in gap, pauses_detailed should be empty (filtered out)
    assert len(timed_data.pauses_detailed) == 0


def test_filler_cluster_detection():
    from app.services.analyzer_advanced import AdvancedSpeechAnalyzer
    from app.models.timed_models import FillerWordDetail

    fillers = [
        FillerWordDetail(word='э-э', timestamp=1.0, exact_word='Ээ', duration=0.2, confidence=0.9),
        FillerWordDetail(word='э-э', timestamp=2.5, exact_word='Ээ', duration=0.2, confidence=0.9),
        FillerWordDetail(word='ну', timestamp=3.0, exact_word='ну', duration=0.2, confidence=0.9),
        FillerWordDetail(word='эм', timestamp=3.5, exact_word='эм', duration=0.2, confidence=0.9),
    ]

    analyzer = AdvancedSpeechAnalyzer()
    clusters = analyzer._find_filler_clusters(fillers)
    # After tightening gap to 2.0 seconds we expect last two belong to same cluster
    assert any(len(c) >= 2 for c in clusters)


import pytest
from app.services.pipeline import SpeechAnalysisPipeline
from app.services.analyzer import EnhancedAnalysisResult
from app.models.timed_analysis import TimedAnalysisData
from app.models.analysis import FillerWordsStats, PausesStats, PhraseStats
from app.models.transcript import Transcript, TranscriptSegment, WordTiming
from app.models.timed_analysis import TimedFillerWord


@pytest.mark.asyncio
async def test_llm_filler_classification_integration(tmp_path):
    class DummyAnalyzer:
        def analyze(self, transcript, audio_path, include_timings):
            result = EnhancedAnalysisResult(
                duration_sec=5.0,
                speaking_time_sec=4.0,
                speaking_ratio=0.8,
                words_total=3,
                words_per_minute=45.0,
                filler_words=FillerWordsStats(total=1, per_100_words=33.3, items=[{"word":"э-э","count":1}]),
                pauses=PausesStats(count=0, avg_sec=0.0, max_sec=0.0, long_pauses=[]),
                phrases=PhraseStats(count=0, avg_words=0.0, avg_duration_sec=0.0, min_words=0, max_words=0, min_duration_sec=0.0, max_duration_sec=0.0, length_classification="", rhythm_variation=""),
                advice=[],
                transcript="Ээ тест",
                timed_data=TimedAnalysisData(
                    filler_words_detailed=[TimedFillerWord(word="э-э", timestamp=1.0, exact_word="Ээ", confidence=0.9, duration=0.2)],
                    pauses_detailed=[],
                    speech_rate_windows=[],
                    word_timings_count=2,
                    speaking_activity=[]
                ),
                gigachat_analysis=None
            )
            return result

    class FakeGigaChatClient:
        async def classify_fillers_context(self, contexts, cache=None):
            return [dict(**c, is_filler=True, score=0.9) for c in contexts]

    # Build a transcript with word timings that include the filler at 1.0s
    words = [WordTiming(word='Ээ', start=1.0, end=1.2, confidence=0.95),
             WordTiming(word='привет', start=1.3, end=1.6, confidence=0.95)]
    segments = [TranscriptSegment(start=1.0, end=1.6, text='Ээ привет', words=words)]
    transcript = Transcript(text='Ээ привет', segments=segments, word_timings=words)

    pipeline = SpeechAnalysisPipeline(
        transcriber=None,
        analyzer=DummyAnalyzer(),
        gigachat_client=FakeGigaChatClient(),
        enable_cache=False,
        enable_metrics=False,
        include_timings=True,
    )

    settings.llm_fillers_enabled = True
    result = await pipeline._analyze_speech(transcript, tmp_path / "tmp.wav")

    assert result.timed_data.filler_words_detailed[0].context_score == 0.9
    assert result.timed_data.filler_words_detailed[0].is_context_filler is True


def test_vad_filters_pauses_monkeypatch(monkeypatch, tmp_path):
    # Build transcript with a pause between two words >= MIN_PAUSE_GAP_SEC
    words = [WordTiming(word='один', start=0.0, end=0.3, confidence=0.95),
             WordTiming(word='два', start=1.0, end=1.2, confidence=0.95)]
    segments = [TranscriptSegment(start=0.0, end=1.2, text='один два', words=words)]
    transcript = Transcript(text='один два', segments=segments, word_timings=words)

    # Force detect_speech_regions to report speech inside the gap
    def fake_detect(audio_path, use_pyannote, use_webrtc, webrtc_mode=None, pyannote_model=None):
        return [(0.5, 1.05)]

    from app.services import vad
    monkeypatch.setattr(vad, 'detect_speech_regions', fake_detect)

    analyzer = SpeechAnalyzer()
    timed = analyzer._analyze_with_timings(transcript, tmp_path / 'dummy.wav')

    # Because VAD reports speech within 0.5-1.05s overlapping the gap, pauses should be filtered out
    assert len(timed.pauses_detailed) == 0
