import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_model: str = os.getenv("LLM_MODEL", "gemma-3-12b-it")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "text-embedding-bge-large-zh-v1.5"
    )

    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore/health_db")
    clean_sop_dir: str = os.getenv("CLEAN_SOP_DIR", "./data/sources/clean_sop")
    air_knowledge_dir: str = os.getenv(
        "AIR_KNOWLEDGE_DIR", "./data/sources/air_purifier_knowledge"
    )


settings = Settings()
