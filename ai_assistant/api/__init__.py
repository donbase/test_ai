from ai_assistant.api.ask_question import api_router as ask_question_router
from ai_assistant.api.add_source import api_router as add_source_router
from ai_assistant.api.get_recommendation import api_router as get_recommendation_router
from ai_assistant.api.get_quest import api_router as get_quest_router
from ai_assistant.api.get_recommendation import api_router as get_recommendation_router

list_of_routers = [
    ask_question_router,
    add_source_router,
    get_recommendation_router,
    get_quest_router,
    get_recommendation_router,
]


__all__ = [
    "list_of_routers",
]