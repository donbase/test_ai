from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import INTEGER, TEXT, TIMESTAMP, UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base
from ai_assistant.db.orm_models.base import DeclarativeBase


class HistoryTable(DeclarativeBase):
    __tablename__ = "history_table_uuid"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
        unique=True,
        doc="Unique id of the string in table",
    )

    user_id = Column(
        UUID(as_uuid=True),
        doc="User id",
        nullable=False,
    )

    dt_created = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),  # pylint: disable=not-callable
        nullable=False,
        doc="Date and time when string in table was created",
    )

    text_chunk = Column(
        TEXT,
        nullable=False,
    )

    def __repr__(self):
        columns = {column.name: getattr(self, column.name) for column in self.__table__.columns}
        return f'<{self.__tablename__}: {", ".join(map(lambda x: f"{x[0]}={x[1]}", columns.items()))}>'
