from langchain_openai import ChatOpenAI
from .config import settings


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        openai_api_key="not-needed",
        openai_api_base=settings.llm_base_url,
    )
