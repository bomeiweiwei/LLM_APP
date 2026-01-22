from openai import OpenAI
from langchain.embeddings.base import Embeddings
from .config import settings


class LmStudioEmbeddings(Embeddings):
    def __init__(self, model_name: str, url: str):
        self.model_name = model_name
        self.url = url
        self.client = OpenAI(base_url=url, api_key="lm-studio")

    def embed_query(self, text: str):
        resp = self.client.embeddings.create(input=text, model=self.model_name)
        return resp.data[0].embedding

    def embed_documents(self, texts: list[str]):
        resp = self.client.embeddings.create(input=texts, model=self.model_name)
        return [x.embedding for x in resp.data]


def get_embeddings() -> Embeddings:
    return LmStudioEmbeddings(
        model_name=settings.embedding_model, url=settings.llm_base_url
    )
