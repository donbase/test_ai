from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status
import re
from ai_assistant.db.connection import get_session
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from ai_assistant.config import settings
from ai_assistant.schemas import Question, QuestionId
from ai_assistant.config import prompts
from ai_assistant.utils.history import get_history_id, get_history, save_history, save_history_id
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from ai_assistant.utils.closest_chunks import get_closest_chunks_id
from enum import Enum

api_router = APIRouter(tags=["ai_assistant"])


_rl = InMemoryRateLimiter(requests_per_second=9,
                          check_every_n_seconds=0.05)


_llm = ChatOpenAI(
    base_url="https://llm.api.cloud.yandex.net/v1",
    model=f"gpt://{settings.YC_FOLDER_ID}/gpt-oss-120b/latest",
    api_key=settings.YC_API_KEY,
    temperature=0.0,
    rate_limiter=_rl
)

emb = YandexGPTEmbeddings(
    api_key=settings.YC_API_KEY,
    folder_id=settings.YC_FOLDER_ID,
)



@api_router.get(
    "/get_quest_id",
    status_code=status.HTTP_200_OK,
)
async def get_quest_id(
    age: int,
):
    response = _llm.invoke(f"""Придумай квест для ребёнка {age} лет, направленный на развитие полезных привычек и навыков.
Каждый квест должен иметь короткое игровое название (2–3 слова) и краткое описание (до 10 слов), объясняющее, что нужно сделать. Выдавай по одному квесту за раз.
Примеры:
Час тишины
За 1 час до сна выключи экраны.

Пенная броня
Почисти зубы и прими душ тёплой водой.

История Снов
Прочитай или послушай сказку перед сном.

Гнездо Спокойствия
Собери игрушки и подготовь комнату ко сну.""")
    return {"response": re.sub(
        r"Thoughts:.*?Answer:",
        "",
        response.content,
        flags=re.S
    ).strip()
            }
