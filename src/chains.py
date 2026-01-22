import uuid
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from .retrieval import parent_document_retrieval

SYSTEM_PROMPT = """
你是一位耐心且有清潔專業的 AI 助教，負責回答顧客的問題。

請根據「提供的參考資料（context）」來回答問題：
- 只能使用 context 中的資訊作答
- 不要自行補充未出現在 context 中的知識
- 回答時請使用清楚、白話、教學導向的說明方式
- 回答時不使用Markdown格式
- 問答跟參考資料無關時回答：此問題不屬於清潔SOP或空氣清淨機知識

參考資料：
{context}
""".strip()

USER_PROMPT = """
問題：
{question}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
)


def build_rag_chain(llm, vector_store: Chroma, n: int = 20, k: int = 1):
    parallel = RunnableParallel(
        question=RunnablePassthrough(),
        context=RunnableLambda(
            lambda input: parent_document_retrieval(
                vector_store=vector_store, question=input["question"], n=n, k=k
            )
        ),
    )
    return parallel | prompt | llm | StrOutputParser()


def build_parent_child_docs(documents, category: str, parent_splitter, child_splitter):
    total_docs = []
    parent_docs = parent_splitter.split_documents(documents)

    for pdoc in parent_docs:
        parent_id = str(uuid.uuid4())
        pdoc.metadata["id"] = parent_id
        pdoc.metadata["parent_id"] = parent_id
        pdoc.metadata["category"] = category
        # 也可加：pdoc.metadata["source"] = pdoc.metadata.get("source")

        child_docs = child_splitter.split_documents([pdoc])
        for cdoc in child_docs:
            cdoc.metadata["id"] = str(uuid.uuid4())
            cdoc.metadata["parent_id"] = parent_id
            cdoc.metadata["category"] = category

        total_docs.append(pdoc)
        total_docs.extend(child_docs)

    return total_docs
