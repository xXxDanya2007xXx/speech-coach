import json
import logging
from typing import Optional, Dict, Any, List
import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.models.gigachat import GigaChatAnalysis
from app.models.analysis import AnalysisResult

logger = logging.getLogger(__name__)


class GigaChatError(Exception):
    """Кастомная ошибка для GigaChat API"""
    pass


class GigaChatClient:
    """Клиент для работы с GigaChat API"""

    def __init__(self):
        self.api_key = settings.gigachat_api_key.get_secret_value(
        ) if settings.gigachat_api_key else None
        self.base_url = settings.gigachat_base_url
        self.model = settings.gigachat_model
        self.timeout = settings.gigachat_timeout
        self.max_tokens = settings.gigachat_max_tokens

        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[float] = None

        # Инициализируем HTTP-клиент
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def authenticate(self) -> None:
        """Аутентификация в GigaChat API и получение access token"""
        if not self.api_key:
            raise GigaChatError("GigaChat API key not configured")

        try:
            # Согласно документации GigaChat, для получения токена используется
            # POST запрос на /oauth/token с grant_type=client_credentials
            auth_url = f"{self.base_url}/oauth/token"
            auth_data = {
                "grant_type": "client_credentials",
                "scope": "GIGACHAT_API_PERS"
            }

            # Используем Basic Auth с client_id = api_key, client_secret = пустая строка
            auth_response = await self.client.post(
                auth_url,
                data=auth_data,
                auth=(self.api_key, "")
            )
            auth_response.raise_for_status()

            auth_result = auth_response.json()
            self._access_token = auth_result.get("access_token")

            if not self._access_token:
                raise GigaChatError("Failed to obtain access token")

            logger.info("GigaChat authentication successful")

        except httpx.RequestError as e:
            logger.error(f"GigaChat authentication request failed: {e}")
            raise GigaChatError(f"Authentication request failed: {e}")
        except Exception as e:
            logger.error(f"GigaChat authentication error: {e}")
            raise GigaChatError(f"Authentication error: {e}")

    async def analyze_speech(self, analysis_result: AnalysisResult) -> Optional[GigaChatAnalysis]:
        """
        Отправляет результаты анализа в GigaChat для получения
        расширенных персонализированных рекомендаций.
        """
        if not settings.gigachat_enabled:
            logger.info("GigaChat analysis is disabled")
            return None

        if not self._access_token:
            try:
                await self.authenticate()
            except GigaChatError as e:
                logger.warning(f"Failed to authenticate with GigaChat: {e}")
                return None

        try:
            # Создаем промпт для анализа
            prompt = self._create_analysis_prompt(analysis_result)

            # Формируем запрос
            chat_url = f"{self.base_url}/chat/completions"
            request_data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": """Ты эксперт по публичным выступлениям и ораторскому искусству. 
                        Анализируй речь по предоставленным метрикам и транскрипту. 
                        Дай развернутый, персонализированный анализ. 
                        Отвечай в формате JSON со следующей структурой:
                        {
                            "overall_assessment": "строка - общая оценка",
                            "strengths": ["массив строк - сильные стороны"],
                            "areas_for_improvement": ["массив строк - зоны роста"],
                            "detailed_recommendations": ["массив строк - конкретные рекомендации"],
                            "key_insights": ["массив строк - ключевые инсайты"],
                            "confidence_score": число от 0 до 1
                        }"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": self.max_tokens
            }

            headers = {
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json"
            }

            response = await self.client.post(chat_url, json=request_data, headers=headers)
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Парсим JSON ответ
            try:
                parsed_content = json.loads(content)
                return GigaChatAnalysis(**parsed_content)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Failed to parse GigaChat response: {e}")
                # Пробуем извлечь JSON из текста, если он есть
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    parsed_content = json.loads(json_str)
                    return GigaChatAnalysis(**parsed_content)
                else:
                    # Если JSON не найден, создаем структурированный ответ из текста
                    return GigaChatAnalysis(
                        overall_assessment=content[:500],
                        strengths=[],
                        areas_for_improvement=[],
                        detailed_recommendations=[],
                        key_insights=[content[:1000] if len(
                            content) > 1000 else content],
                        confidence_score=0.5
                    )

        except httpx.RequestError as e:
            logger.error(f"GigaChat API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing GigaChat response: {e}")
            return None

    def _create_analysis_prompt(self, analysis_result: AnalysisResult) -> str:
        """Создает промпт для анализа на основе результатов"""
        # Формируем строку со словами-паразитами
        filler_items = ""
        for item in analysis_result.filler_words.items:
            if item.get("count", 0) > 0:
                filler_items += f"- {item['word']}: {item['count']} раз\n"

        # Формируем строку с паузами
        pauses_info = ""
        if analysis_result.pauses.long_pauses:
            pauses_info = "Длинные паузы:\n"
            # Берем только топ-3
            for pause in analysis_result.pauses.long_pauses[:3]:
                pauses_info += f"- {pause['duration']:.1f} сек (с {pause['start']:.1f} по {
                    pause['end']:.1f})\n"

        # Формируем строку с рекомендациями
        advice_info = ""
        for advice in analysis_result.advice:
            advice_info += f"- {advice.title}: {advice.observation}\n"

        prompt = f"""Проанализируй это публичное выступление:

=== ТРАНСКРИПТ ===
{analysis_result.transcript[:3000]}{'... [текст сокращен]' if len(analysis_result.transcript) > 3000 else ''}

=== МЕТРИКИ ===
Длительность: {analysis_result.duration_sec:.1f} секунд
Время говорения: {analysis_result.speaking_time_sec:.1f} секунд
Коэффициент говорения: {analysis_result.speaking_ratio:.2%}
Темп речи: {analysis_result.words_per_minute:.1f} слов/минуту
Общее количество слов: {analysis_result.words_total}

Слова-паразиты: {analysis_result.filler_words.total} ({analysis_result.filler_words.per_100_words:.1f} на 100 слов)
{f'Наиболее частые:\n{filler_items}' if filler_items else ''}

Количество пауз: {analysis_result.pauses.count}
Средняя длина паузы: {analysis_result.pauses.avg_sec:.1f} секунд
Самая длинная пауза: {analysis_result.pauses.max_sec:.1f} секунд
{pauses_info if pauses_info else ''}

Количество фраз: {analysis_result.phrases.count}
Средняя длина фразы: {analysis_result.phrases.avg_words:.1f} слов
Классификация длины фраз: {analysis_result.phrases.length_classification}
Вариативность ритма: {analysis_result.phrases.rhythm_variation}

=== СТАНДАРТНЫЕ РЕКОМЕНДАЦИИ ===
{advice_info}

Дай развернутый анализ с учетом контекста публичного выступления.
Обрати внимание на:
1. Ясность и структурированность мысли
2. Эмоциональную окраску речи
3. Убедительность аргументации
4. Взаимодействие с аудиторией (на основе пауз и темпа)
5. Профессиональную лексику и терминологию
6. Общее впечатление от выступления

Верни ответ строго в формате JSON, как указано в system prompt."""

        return prompt

    async def close(self):
        """Закрывает HTTP-клиент"""
        await self.client.aclose()
