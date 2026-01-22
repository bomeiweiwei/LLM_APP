from langchain_openai import ChatOpenAI
from langchain.embeddings.base import Embeddings
from openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import uuid

llm_model_name = "gemma-3-12b-it"
llm_base_url = "http://localhost:1234/v1"
text_embedding_model_name = "text-embedding-bge-large-zh-v1.5"

llm = ChatOpenAI(
    model=llm_model_name, openai_api_key="not-needed", openai_api_base=llm_base_url
)


class LmStudioEmbeddings(Embeddings):
    def __init__(self, model_name, url):
        self.model_name = model_name
        self.url = url
        self.client = OpenAI(base_url=url, api_key="lm-studio")

    def embed_query(self, text: str):
        response = self.client.embeddings.create(input=text, model=self.model_name)
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]):
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return [x.embedding for x in response.data]


embeddings = LmStudioEmbeddings(model_name=text_embedding_model_name, url=llm_base_url)

# ====================================================
sop_loader = DirectoryLoader(
    path="data/sources/clean_sop",
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)

sop_documents = sop_loader.load()
# print(f"總共有{len(documents)}筆資料")

air_loader = DirectoryLoader(
    path="data/sources/air_purifier_knowledge",
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)

air_documents = air_loader.load()
# ====================================================

# ====================================================
sop_parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
sop_child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

air_parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
air_child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

sop_total_docs = []
air_total_docs = []
sop_parent_docs = sop_parent_splitter.split_documents(sop_documents)
air_parent_docs = air_parent_splitter.split_documents(air_documents)

for doc in sop_parent_docs:
    doc.metadata["id"] = str(uuid.uuid4())
    doc.metadata["parent_id"] = doc.metadata["id"]
    # 分割子層文件
    split_docs = sop_child_splitter.split_documents([doc])
    for sdoc in split_docs:
        sdoc.metadata["id"] = str(uuid.uuid4())
        sdoc.metadata["parent_id"] = doc.metadata["id"]
    sop_total_docs.append(doc)
    sop_total_docs.extend(split_docs)

for doc in air_parent_docs:
    doc.metadata["id"] = str(uuid.uuid4())
    doc.metadata["parent_id"] = doc.metadata["id"]
    # 分割子層文件
    split_docs = air_child_splitter.split_documents([doc])
    for sdoc in split_docs:
        sdoc.metadata["id"] = str(uuid.uuid4())
        sdoc.metadata["parent_id"] = doc.metadata["id"]
    air_total_docs.append(doc)
    air_total_docs.extend(split_docs)
# ====================================================

# ====================================================
# 設定資料庫路徑
persist_dir = "./vectorstore/health_db"

# 建立或載入現有的 Chroma 向量資料庫
sop_vector_store = Chroma(
    collection_name="cleaning_sop",
    embedding_function=embeddings,
    persist_directory=persist_dir,
)
sop_vector_store.delete_collection()
sop_vector_store = Chroma(
    collection_name="cleaning_sop",
    embedding_function=embeddings,
    persist_directory=persist_dir,
)

sop_vector_store.add_documents(sop_total_docs)

# 建立或載入現有的 Chroma 向量資料庫
air_vector_store = Chroma(
    collection_name="air_purifier",
    embedding_function=embeddings,
    persist_directory=persist_dir,
)
air_vector_store.delete_collection()
air_vector_store = Chroma(
    collection_name="air_purifier",
    embedding_function=embeddings,
    persist_directory=persist_dir,
)

air_vector_store.add_documents(air_total_docs)
# ====================================================


# ====================================================
# n:總共找多少相關文件
# k:取出幾份文件
def parent_document_retrieval(vector_store, question, n=20, k=1):
    docs = vector_store.similarity_search(question, k=n)
    seen_ids = set()
    documents = []
    for doc in docs:
        if doc.metadata["parent_id"] not in seen_ids:
            seen_ids.add(doc.metadata["parent_id"])
            parent_docs = vector_store.similarity_search(
                query="", k=1, filter={"id": doc.metadata["parent_id"]}
            )
            if len(parent_docs) > 0:
                documents.append(parent_docs[0])

    return documents[:k]
# ====================================================

# ====================================================
sop_parent_document_parallel = RunnableParallel(
    question=RunnablePassthrough(),
    context=RunnableLambda(
        lambda input: parent_document_retrieval(
            vector_store=sop_vector_store, question=input["question"], n=20
        )  # 從dict取值
    ),
)
air_parent_document_parallel = RunnableParallel(
    question=RunnablePassthrough(),
    context=RunnableLambda(
        lambda input: parent_document_retrieval(
            vector_store=air_vector_store, question=input["question"], n=20
        )  # 從dict取值
    ),
)
# ====================================================

system_prompt = """
你是一位耐心且有清潔專業的 AI 助教，負責回答顧客的問題。

請根據「提供的參考資料（context）」來回答問題：
- 只能使用 context 中的資訊作答
- 不要自行補充未出現在 context 中的知識
- 回答時請使用清楚、白話、教學導向的說明方式
- 回答時不使用Markdown格式
- 問答跟參考資料無關時拒絕回答 

參考資料：
{context}
"""

user_prompt = """
問題：
{question}
"""

question_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", user_prompt)]
)

# ====================================================
sop_parent_document_rag_chain = (
    sop_parent_document_parallel | question_prompt | llm | StrOutputParser()
)
air_parent_document_rag_chain = (
    air_parent_document_parallel | question_prompt | llm | StrOutputParser()
)
sop_answer = sop_parent_document_rag_chain.invoke(
    {"question": "洗衣機槽可以用過碳酸鈉嗎？"}
)
print("sop answer", sop_answer)

air_answer = air_parent_document_rag_chain.invoke({"question": "濾網一年要花多少錢？"})
print("air answer", air_answer)
# ====================================================
