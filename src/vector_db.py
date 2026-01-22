from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from .config import settings


def get_vector_store(collection_name: str, embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def reset_collection(collection_name: str, embeddings: Embeddings) -> Chroma:
    vs = get_vector_store(collection_name, embeddings)
    vs.delete_collection()
    return get_vector_store(collection_name, embeddings)
