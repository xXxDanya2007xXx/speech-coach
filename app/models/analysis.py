from typing import List, Dict, Any, Literal
from pydantic import BaseModel


class FillerWordsStats(BaseModel):
    total: int
    per_100_words: float
    items: List[Dict[str, Any]]  # {"word": str, "count": int}


class PausesStats(BaseModel):
    count: int
    avg_sec: float
    max_sec: float
    # {"start": float, "end": float, "duration": float}
    long_pauses: List[Dict[str, float]]


class PhraseStats(BaseModel):
    """
    Статистика по фразам — последовательностям речи между паузами.
    """

    count: int
    avg_words: float
    avg_duration_sec: float
    min_words: int
    max_words: int
    min_duration_sec: float
    max_duration_sec: float

    # Качественная оценка длины фраз:
    # short_phrases  — в среднем очень короткие фразы;
    # balanced       — сбалансированная длина;
    # long_phrases   — в среднем очень длинные фразы.
    length_classification: str

    # Качественная оценка вариативности длительности фраз:
    # insufficient_data  — данных мало для оценки;
    # uniform            — фразы по длительности довольно равномерны;
    # moderately_variable — умеренная вариативность;
    # highly_variable    — сильно разная длительность фраз.
    rhythm_variation: str


class AdviceItem(BaseModel):
    """
    Структурированный совет по одной из категорий анализа.
    """

    # Категория, к которой относится совет
    category: Literal["speech_rate", "filler_words", "pauses", "phrasing"]

    # Условная "строгость" совета
    severity: Literal["info", "suggestion", "warning"]

    # Краткий заголовок (для UI)
    title: str

    # Наблюдение по текущей записи
    observation: str

    # Рекомендация по улучшению
    recommendation: str


class AnalysisResult(BaseModel):
    duration_sec: float

    # Время, когда спикер реально говорит (без "тишины" и шумных пауз)
    speaking_time_sec: float
    # Доля времени говорения: speaking_time_sec / duration_sec
    speaking_ratio: float

    words_total: int
    words_per_minute: float

    filler_words: FillerWordsStats
    pauses: PausesStats
    phrases: PhraseStats

    advice: List[AdviceItem]
    transcript: str
