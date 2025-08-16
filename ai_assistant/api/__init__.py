from ai_assistant.api.ask_question import api_router as ask_question_router

list_of_routers = [
    ask_question_router,
]


__all__ = [
    "list_of_routers",
]