from typing import List
from langchain_chroma import Chroma


def parent_document_retrieval(
    vector_store: Chroma, question: str, n: int = 20, k: int = 1
):
    """
    n: 先取回 n 份相似子文件（或混合文件）
    k: 去重後回傳 k 份 parent 文件
    """
    docs = vector_store.similarity_search(question, k=n)
    seen_parent_ids = set()
    parents = []

    for doc in docs:
        pid = doc.metadata.get("parent_id")
        if not pid or pid in seen_parent_ids:
            continue
        seen_parent_ids.add(pid)

        parent_docs = vector_store.similarity_search(
            query="",
            k=1,
            filter={"id": pid},
        )
        if parent_docs:
            parents.append(parent_docs[0])

        if len(parents) >= k:
            break

    return parents
