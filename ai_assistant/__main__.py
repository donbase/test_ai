from logging import getLogger

from fastapi import FastAPI
import uvicorn

from ai_assistant.config import settings, Settings
from ai_assistant.api import list_of_routers
from ai_assistant.utils import get_hostname


logger = getLogger(__name__)


def bind_routes(application: FastAPI, setting: Settings) -> None:
    for route in list_of_routers:
        application.include_router(route, prefix=setting.PATH_PREFIX)


def get_app() -> FastAPI:
    description = "Микросервис ИИ-ассистента"

    application = FastAPI(
        title="ai_assistant",
        description=description,
        docs_url="/docs",
        openapi_url="/openapi",
        version="1.0.0",
    )

    bind_routes(application, settings)
    application.state.settings = settings
    return application


app = get_app()


if __name__ == "__main__":  # pragma: no cover
    uvicorn.run(
        "ai_assistant.__main__:app",
        host=get_hostname(settings.APP_HOST),
        port=settings.APP_PORT,
        reload=True,
    )
