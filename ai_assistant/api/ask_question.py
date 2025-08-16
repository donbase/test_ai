from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status


from ai_assistant.db.connection import get_session
from langchain_community.chat_models.yandex import ChatYandexGPT
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from ai_assistant.config import settings
from ai_assistant.schemas import Question, QuestionId

from ai_assistant.utils.history import get_history_id, get_history, save_history, save_history_id


api_router = APIRouter(tags=["ai_assistant"])


_rl = InMemoryRateLimiter(requests_per_second=9,
                          check_every_n_seconds=0.05)
_llm = ChatYandexGPT(
    model_uri=f"gpt://{settings.YC_FOLDER_ID}/yandexgpt-lite/latest",
    api_key=settings.YC_API_KEY,
    folder_id=settings.YC_FOLDER_ID,
    temperature=0.0,
    rate_limiter=_rl
)

_llm_new = ChatOpenAI(
    base_url="https://llm.api.cloud.yandex.net/v1",
    model=f"gpt://{settings.YC_FOLDER_ID}/gpt-oss-20b/latest",
    api_key=settings.YC_API_KEY,
    temperature=0.0,
    rate_limiter=_rl
)


@api_router.post(
    "/ask_question_id",
    status_code=status.HTTP_200_OK,
)
async def ask_question_id(
    model: QuestionId = Body(...),
    session: AsyncSession = Depends(get_session),
):
    history = await get_history_id(model.user_id, session)
    messages = [{"role": "user", "content": msg} for msg in history]
    messages.append({"role": "user", "content": "Вопрос: " + model.text})
    response = _llm_new.invoke(messages)
    if model.text[-1] != "?":
        await save_history_id(model.user_id, session, model.text)
    return {"response": response.content.strip()}



async def ask_question_uuid(
    model: Question = Body(...),
    session: AsyncSession = Depends(get_session),
):

    history = await get_history(model.user_id, session)
    messages = [{"role": "user", "content": msg} for msg in history]
    messages.append({"role": "user", "content": "Вопрос: " + model.text})

    response = _llm_new.invoke(messages)
    await save_history(model.user_id, session, model.text)

    return response.content.strip()

















