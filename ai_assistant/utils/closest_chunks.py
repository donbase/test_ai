
from sqlalchemy import select, insert, desc
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status
from ai_assistant.db.orm_models import VectorTable
from ai_assistant.db.connection import get_session


async def get_closest_chunks_id(
    id: int,
    vector: list,
    session: AsyncSession,
):
    result = await session.execute(
        select(VectorTable.text_chunk)
        .where(VectorTable.user_id == id)
        .order_by(VectorTable.embedding.op("<->")(vector))
        .limit(5)
    )

    closest_vectors = result.scalars().all()
    return closest_vectors