from sqlalchemy import select, insert, desc
from sqlalchemy.ext.asyncio import AsyncSession
from ai_assistant.db.orm_models import HistoryTable, HistoryTableId
import uuid


async def get_history_id(
    id: int,
    session: AsyncSession,
):
    result = await session.execute(
        select(HistoryTableId.text_chunk)
        .where(HistoryTableId.user_id == id)
        .order_by(desc(HistoryTableId.dt_created))
        .limit(10)
    )
    return result.scalars().all()

async def save_history_id(
    id: int,
    session: AsyncSession,
    message: str,
):
    session.add(HistoryTableId(user_id=id, text_chunk=message))
    await session.commit()



async def get_history(
    id: uuid.UUID,
    session: AsyncSession,
):
    result = await session.execute(
        select(HistoryTable.text_chunk)
        .where(HistoryTable.user_id == id)
        .order_by(desc(HistoryTable.dt_created))
        .limit(10)
    )

    return result.scalars().all()


async def save_history(
    id: uuid.UUID,
    session: AsyncSession,
    message: str,
):
    session.add(HistoryTable(user_id=id, text_chunk=message))
    await session.commit()

