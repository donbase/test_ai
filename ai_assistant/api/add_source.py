from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status
from fastapi import APIRouter, UploadFile, File
from ai_assistant.db.connection import get_session
from ai_assistant.db.orm_models import VectorTable
import pdfplumber
from io import BytesIO
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from ai_assistant.config import settings
from docx import Document


api_router = APIRouter(tags=["ai_assistant"])


emb = YandexGPTEmbeddings(
    api_key=settings.YC_API_KEY,
    folder_id=settings.YC_FOLDER_ID,
)



@api_router.post(
    "/add_source",
    status_code=status.HTTP_200_OK,
)
async def add_source_handler(
    user_id: int,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    contents = await file.read()
    stream = BytesIO(contents)

    name = file.filename
    extension = name.split('.')[-1].lower()
    content_type = file.content_type


    try:
        if extension == 'pdf' or content_type == 'application/pdf':
            with pdfplumber.open(stream) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    txt = page.extract_text()
                    if txt:
                        vector = emb.embed_query(txt)
                        new_string = VectorTable()
                        new_string.user_id = user_id
                        new_string.embedding = vector
                        new_string.text_chunk = txt
                        session.add(new_string)

        elif extension in ('docx', 'doc') or content_type in (
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/msword',
        ):
            document = Document(stream)
            full_text = "\n".join([para.text for para in document.paragraphs])
            page_size_chars = 2000
            pages = [full_text[i:i + page_size_chars] for i in range(0, len(full_text), page_size_chars)]
            for page in pages:
                vector = emb.embed_query(page)
                new_string = VectorTable()
                new_string.user_id = user_id
                new_string.embedding = vector
                new_string.text_chunk = page
                session.add(new_string)


        await session.commit()


    except Exception as exc:
        return {"error": f"Ошибка при парсинге {name}: {exc}"}
