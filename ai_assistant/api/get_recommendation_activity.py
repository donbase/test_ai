import uuid
import httpx
from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status
import re
from ai_assistant.db.connection import get_session
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from ai_assistant.config import settings
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from ai_assistant.utils.closest_chunks import get_closest_chunks_id
from enum import Enum
from datetime import datetime, timedelta



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

class Sex(Enum):
    male = "male"
    female = "female"

BASE_URL = "http://158.160.44.235/api/v1/child/"


@api_router.get(
    "/get_recommendation_activity_id/{id}",
    status_code=status.HTTP_200_OK,
)
async def get_recommendation(
    id: uuid.UUID,
    sex: Sex,
    age: int,
):
    before_date_obj = datetime.now()
    before_date = before_date_obj.strftime('%Y-%m-%d')
    after_date_obj = before_date_obj - timedelta(days=7)
    after_date = after_date_obj.strftime('%Y-%m-%d')
    async with httpx.AsyncClient() as client:
        steps_response = await client.get(BASE_URL + str(id) + "/" + "steps", params={"after_date": after_date, "before_date": before_date})
        activity_response = await client.get(BASE_URL + str(id) + "/" + "active_minutes", params={"after_date": after_date, "before_date": before_date})

    prompt = (
        f"У меня есть следующая информация по ребенку с полом {sex} и возрастом {age}:\n"
        f"Информация по шагам:\n{steps_response.text}\n\n"
        f"Информация по активности:\n{activity_response.text}\n\n"
        f"Оценивая эти показатели, дай по ним отзыв, дай небольшую (2-3 предложения) рекомендацию по активности для этого ребенка."
    )

    response = _llm.invoke(prompt)
    return {"response": re.sub(
        r"Thoughts:.*?Answer:",
        "",
        response.content,
        flags=re.S
    ).strip()
            }
