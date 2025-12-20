from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse
from typing import List, Optional
import logging
from pydantic import BaseModel
import os
from pathlib import Path
import httpx

from app.services.gigachat import GigaChatClient, GigaChatError
from app.services.cache import AnalysisCache
from app.core.config import settings
from pathlib import Path
from app.api.deps import get_gigachat_client

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Модель сообщения в чате"""
    role: str  # "user" или "assistant"
    content: str


class ChatRequest(BaseModel):
    """Модель запроса на чат с Gigachat"""
    message: Optional[str] = None  # Новое одиночное сообщение
    messages: Optional[List[ChatMessage]] = None  # Или список сообщений
    analysis_context: Optional[dict] = None  # Контекст анализа
    history: Optional[List[dict]] = None  # История чата
    analysis_id: Optional[str] = None  # ID анализа, если пользователь хочет обсудить конкретный анализ
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000


class ChatResponse(BaseModel):
    """Модель ответа от Gigachat"""
    response: str
    model: str
    tokens_used: Optional[int] = None


@router.post("/chat", response_model=ChatResponse)
async def chat_with_gigachat(
    request: ChatRequest,
    gigachat_client: GigaChatClient = Depends(get_gigachat_client)
):
    """
    Чат с GigaChat - позволяет задавать вопросы и получать ответы от GigaChat
    Поддерживает контекст анализа и историю чата
    """
    # Если GigaChat не настроен — попытаться вернуть уже сгенерированные сервером советы/резюме по analysis_id
    if gigachat_client is None:
        # Если указан analysis_id — попробуем получить сохраненные данные из кеша и вернуть их
        if request.analysis_id:
            try:
                cache_dir = Path(settings.cache_dir) / "analysis"
                cache = AnalysisCache(cache_dir, ttl_seconds=settings.cache_ttl)
                # Попробуем несколько вариантов ключа (как делает кеширование в pipeline)
                candidates = [request.analysis_id, f"{request.analysis_id}_gigachat_True", f"{request.analysis_id}_gigachat_False"]
                cached = None
                for key in candidates:
                    cached = cache.get_by_key(key)
                    if cached is not None:
                        break

                if cached is None:
                    raise HTTPException(status_code=503, detail="GigaChat unavailable and no cached analysis found for provided analysis_id")

                # Compose a response from cached data: prefer stored gigachat_analysis, then advice
                parts = []
                if getattr(cached, 'gigachat_analysis', None):
                    # gigachat_analysis may be a model or dict
                    ga = getattr(cached, 'gigachat_analysis')
                    try:
                        # if model-like, try to extract summary fields
                        overall = getattr(ga, 'overall_assessment', None) or (ga.get('overall_assessment') if isinstance(ga, dict) else None)
                    except Exception:
                        overall = None
                    if overall:
                        parts.append(str(overall))
                    else:
                        parts.append(str(ga))

                if getattr(cached, 'advice', None):
                    adv = getattr(cached, 'advice')
                    if isinstance(adv, list) and len(adv) > 0:
                        parts.append('\nРекомендации:')
                        for a in adv:
                            # advice items might be Pydantic models or dicts
                            title = getattr(a, 'title', None) or (a.get('title') if isinstance(a, dict) else None)
                            rec = getattr(a, 'recommendation', None) or (a.get('recommendation') if isinstance(a, dict) else None)
                            if title or rec:
                                parts.append(f"- {title or ''}: {rec or ''}")

                # transcript excerpt
                if getattr(cached, 'transcript', None):
                    excerpt = str(getattr(cached, 'transcript'))[:800]
                    parts.append('\nТранскрипт (фрагмент): ' + excerpt)

                response_text = '\n'.join(parts) if parts else 'Кэшированного анализа нет подробностей.'
                return ChatResponse(response=response_text, model='local-cache', tokens_used=None)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to load cached analysis for chat fallback: {e}")
                raise HTTPException(status_code=500, detail="Failed to load cached analysis")
        # Если analysis_id не указан — сообщаем, что служба недоступна
        raise HTTPException(status_code=503, detail="GigaChat service is not configured and no analysis_id provided")
    
    try:
        # Подготовим сообщения для отправки в GigaChat
        formatted_messages = []
        
        # Добавим системное сообщение с контекстом
        system_content = """Ты - помощник по развитию навыков публичных выступлений. 
Твоя задача - помогать людям улучшать свои ораторские способности, 
давать советы по структуре речи, стилю выступления, взаимодействию с аудиторией 
и другим аспектам ораторского мастерства."""
        
        # Добавим контекст анализа если он есть
        if request.analysis_context:
            context_lines = []
            ctx = request.analysis_context
            
            if isinstance(ctx, dict):
                if ctx.get('timeline'):
                    timeline = ctx['timeline']
                    if timeline.get('fillers'):
                        context_lines.append(f"Найдено {len(timeline['fillers'])} слов-паразитов")
                    if timeline.get('suspicious_moments'):
                        context_lines.append(f"Найдено {len(timeline['suspicious_moments'])} проблемных моментов")
                    if timeline.get('emphases'):
                        context_lines.append(f"Найдено {len(timeline['emphases'])} выделенных слов")
                
                if ctx.get('transcript'):
                    transcript_preview = str(ctx['transcript'])[:200]
                    context_lines.append(f"Транскрипт: {transcript_preview}...")
                
                if context_lines:
                    system_content += "\n\nКонтекст анализа пользователя:\n" + "\n".join(context_lines)
        
        system_message = {
            "role": "system",
            "content": system_content
        }
        formatted_messages.append(system_message)
        
        # Добавляем историю чата если она есть
        if request.history and isinstance(request.history, list):
            for msg in request.history:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    formatted_messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
        # Если история не передана, используем messages из старого формата
        elif request.messages:
            for msg in request.messages:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Добавляем текущее сообщение
        if request.message:
            formatted_messages.append({
                "role": "user",
                "content": request.message
            })
        elif not request.messages and not request.history:
            raise HTTPException(status_code=400, detail="Messages or message cannot be empty")
        
        # Убедимся, что токен аутентификации действителен
        try:
            await gigachat_client.authenticate()
        except GigaChatError as e:
            logger.error(f"Failed to authenticate with GigaChat API: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to authenticate with GigaChat API: {str(e)}")
        
        # Проверяем, что токен действительно установлен после аутентификации
        if not gigachat_client._access_token:
            logger.error("Failed to obtain access token for chat request")
            raise HTTPException(status_code=500, detail="Failed to authenticate with GigaChat API")
        
        # Отправляем запрос в GigaChat
        chat_url = f"{gigachat_client.api_url}/chat/completions"
        
        request_data = {
            "model": gigachat_client.model,
            "messages": formatted_messages,
            "temperature": request.temperature,
            "max_tokens": min(request.max_tokens, gigachat_client.max_tokens),
        }

        headers = {
            "Authorization": f"Bearer {gigachat_client._access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        logger.info(f"Sending chat request to GigaChat with {len(formatted_messages)} messages...")
        
        try:
            response = await gigachat_client.client.post(chat_url, json=request_data, headers=headers)
        except httpx.ConnectError as e:
            logger.error(f"Connection error when sending chat request to GigaChat: {e}")
            raise HTTPException(status_code=500, detail=f"Connection error when connecting to GigaChat API: {str(e)}")

        if response.status_code != 200:
            logger.error(f"GigaChat API error {response.status_code}: {response.text}")
            raise HTTPException(status_code=500, detail=f"GigaChat API error: {response.text}")

        result = response.json()

        if "choices" not in result or len(result["choices"]) == 0:
            logger.error("No choices in GigaChat response")
            raise HTTPException(status_code=500, detail="No response from GigaChat")

        assistant_response = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens")

        logger.info(f"GigaChat responded with {len(assistant_response)} characters")
        
        return ChatResponse(
            response=assistant_response,
            model=gigachat_client.model,
            tokens_used=tokens_used
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat_with_gigachat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during chat with GigaChat")


@router.post("/chat/analyze-followup", response_model=ChatResponse)
async def analyze_followup_chat(
    request: ChatRequest,
    gigachat_client: GigaChatClient = Depends(get_gigachat_client)
):
    """
    Чат с GigaChat по результатам анализа - позволяет задавать уточняющие вопросы 
    по конкретному анализу речи
    """
    # Если GigaChat не настроен — возвращаем локальную заглушку с кратким резюме, если есть кеш
    if gigachat_client is None:
        try:
            ctx = None
            if request.analysis_id:
                try:
                    cache_dir = Path(settings.cache_dir) / "analysis"
                    cache = AnalysisCache(cache_dir, ttl_seconds=settings.cache_ttl)
                    cached = cache.get_by_key(request.analysis_id)
                    if cached:
                        ctx = cached
                except Exception:
                    ctx = None

            lines = ["GigaChat недоступен — краткое резюме локально:"]
            if ctx:
                if getattr(ctx, 'gigachat_analysis', None):
                    lines.append(str(getattr(ctx, 'gigachat_analysis')))
                if getattr(ctx, 'advice', None):
                    adv = getattr(ctx, 'advice')
                    if isinstance(adv, list) and len(adv) > 0:
                        lines.append('Топ-советы: ' + ', '.join([a.title for a in adv[:3] if getattr(a, 'title', None)]))
            else:
                lines.append('Нет сохраненного контекста анализа. Задайте вопрос, и я постараюсь ответить локально.')

            return ChatResponse(response='\n'.join(lines), model='local-fallback', tokens_used=None)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")
        
        # Если указан analysis_id, попробовать получить контекст анализа из кеша
        # Ожидается, что frontend передаёт `analysis_id` равным ключу кеша (sha256)
        if request.analysis_id:
            try:
                cache_dir = Path(settings.cache_dir) / "analysis"
                cache = AnalysisCache(cache_dir, ttl_seconds=settings.cache_ttl)
                # try raw id and common decorated keys used by cache_analysis
                candidates = [request.analysis_id, f"{request.analysis_id}_gigachat_True", f"{request.analysis_id}_gigachat_False"]
                cached = None
                for key in candidates:
                    cached = cache.get_by_key(key)
                    if cached is not None:
                        break
                if cached:
                    # Построим краткое резюме анализа для передачи в систему
                    summary_lines = []
                    if getattr(cached, 'gigachat_analysis', None):
                        summary_lines.append('GigaChat summary: ' + str(getattr(cached, 'gigachat_analysis')))
                    if getattr(cached, 'transcript', None):
                        # include short excerpt of transcript
                        transcript_excerpt = str(getattr(cached, 'transcript'))[:1000]
                        summary_lines.append('Transcript excerpt: ' + transcript_excerpt)
                    if getattr(cached, 'advice', None):
                        adv = getattr(cached, 'advice')
                        if isinstance(adv, list) and len(adv) > 0:
                            top_advice = ', '.join([a.title for a in adv[:3] if getattr(a, 'title', None)])
                            if top_advice:
                                summary_lines.append('Top advice: ' + top_advice)

                    if summary_lines:
                        context_msg = 'Context for this follow-up analysis:\n' + '\n'.join(summary_lines)
                        # prepend as system message so assistant can use it
                        formatted_messages = [{
                            "role": "system",
                            "content": context_msg
                        }]
                    else:
                        formatted_messages = []
                else:
                    formatted_messages = []
            except Exception as e:
                logger.warning(f"Failed to load analysis context from cache: {e}")
                formatted_messages = []
        else:
            # Подготовим сообщения
            formatted_messages = []
        
        # Системное сообщение, дающее роль и поведение ассистента
        system_content = """Ты - опытный тренер по ораторскому искусству. 
        Пользователь хочет обсудить результаты анализа своего выступления. 
        Отвечай на вопросы пользователя, основываясь на его анализе речи и предоставляя 
        конкретные рекомендации по улучшению. Если пользователь спрашивает о чем-то, 
        что не отражено в анализе, используй экспертные знания в области ораторского мастерства."""

        # если formatted_messages уже содержит system message с контекстом, добавим дополнительный
        if formatted_messages and formatted_messages[0].get("role") == "system":
            # дополним существующий системный контекст
            formatted_messages[0]["content"] += "\n\n" + system_content
        else:
            formatted_messages.insert(0, {"role": "system", "content": system_content})
        
        # Добавляем сообщения пользователя
        for msg in request.messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Убедимся, что токен аутентификации действителен
        try:
            await gigachat_client.authenticate()
        except GigaChatError as e:
            logger.error(f"Failed to authenticate with GigaChat API: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to authenticate with GigaChat API: {str(e)}")
        
        # Проверяем, что токен действительно установлен после аутентификации
        if not gigachat_client._access_token:
            logger.error("Failed to obtain access token for analysis follow-up chat request")
            raise HTTPException(status_code=500, detail="Failed to authenticate with GigaChat API")
        
        # Отправляем запрос в GigaChat
        chat_url = f"{gigachat_client.api_url}/chat/completions"
        
        # Determine max_tokens: prefer request, fall back to client configured limit
        desired_max = request.max_tokens or gigachat_client.max_tokens
        max_tokens = int(min(desired_max, gigachat_client.max_tokens))

        request_data = {
            "model": gigachat_client.model,
            "messages": formatted_messages,
            "temperature": request.temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {gigachat_client._access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        logger.info(f"Sending analysis follow-up request to GigaChat...")
        
        try:
            response = await gigachat_client.client.post(chat_url, json=request_data, headers=headers)
        except httpx.ConnectError as e:
            logger.error(f"Connection error when sending analysis follow-up request to GigaChat: {e}")
            raise HTTPException(status_code=500, detail=f"Connection error when connecting to GigaChat API: {str(e)}")

        if response.status_code != 200:
            logger.error(f"GigaChat API error {response.status_code}: {response.text}")
            raise HTTPException(status_code=500, detail=f"GigaChat API error: {response.text}")

        result = response.json()

        if "choices" not in result or len(result["choices"]) == 0:
            logger.error("No choices in GigaChat response")
            raise HTTPException(status_code=500, detail="No response from GigaChat")

        assistant_response = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens")

        logger.info(f"GigaChat responded with {len(assistant_response)} characters")
        
        return ChatResponse(
            response=assistant_response,
            model=gigachat_client.model,
            tokens_used=tokens_used
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_followup_chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during analysis follow-up chat")


@router.get("/chat/ui", response_class=HTMLResponse)
async def get_chat_page():
    """
    Возвращает HTML-страницу для чата с GigaChat
    """
    try:
        # Определяем путь к файлу шаблона
        template_path = Path(__file__).parent.parent / "templates" / "chat.html"
        
        if not template_path.exists():
            # Если файл не найден, возвращаем простую HTML-страницу
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Чат с GigaChat</title>
                <meta charset="utf-8">
            </head>
            <body>
                <h1>Чат с GigaChat</h1>
                <p>Для полноценного использования чата, пожалуйста, используйте API-запросы к /chat или /chat/analyze-followup</p>
            </body>
            </html>
            """
        else:
            # Читаем содержимое файла шаблона
            with open(template_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
        
        return HTMLResponse(content=html_content)
    
    except Exception as e:
        logger.error(f"Error serving chat page: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error loading chat page")