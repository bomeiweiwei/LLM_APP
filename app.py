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

loader = DirectoryLoader(
    path="data/sources/clean_sop",
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)

documents = loader.load()
# print(f"總共有{len(documents)}筆資料")

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)


total_docs = []

# 分割父層文件
parent_docs = parent_splitter.split_documents(documents)

for doc in parent_docs:
    doc.metadata["id"] = str(uuid.uuid4())
    doc.metadata["parent_id"] = doc.metadata["id"]
    # 分割子層文件
    split_docs = child_splitter.split_documents([doc])
    for sdoc in split_docs:
        sdoc.metadata["id"] = str(uuid.uuid4())
        sdoc.metadata["parent_id"] = doc.metadata["id"]
    total_docs.append(doc)
    total_docs.extend(split_docs)

# 設定資料庫路徑
persist_dir = "./vectorstore/health_db"

# 建立或載入現有的 Chroma 向量資料庫
vector_store = Chroma(
    collection_name="cleaning_sop",  # collection 名稱（相當於一個資料表）
    embedding_function=embeddings,  # 指定嵌入函式
    persist_directory=persist_dir,  # 向量資料儲存路徑
)
vector_store.delete_collection()
vector_store = Chroma(
    collection_name="cleaning_sop",  # collection 名稱（相當於一個資料表）
    embedding_function=embeddings,  # 指定嵌入函式
    persist_directory=persist_dir,  # 向量資料儲存路徑
)

vector_store.add_documents(total_docs)
# print("成功新增新資料至 Chroma。")


# n:總共找多少相關文件
# k:取出幾份文件
def parent_document_retrieval(question, n=20, k=2):
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


# docsresp=parent_document_retrieval("直立式洗衣機多久要洗一次？")
# print(docsresp)

parent_document_parallel = RunnableParallel(
    question=RunnablePassthrough(),
    context=RunnableLambda(
        lambda input: parent_document_retrieval(
            question=input["question"], n=20
        )  # 從dict取值
    ),
)

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

parent_document_rag_chain = (
    parent_document_parallel | question_prompt | llm | StrOutputParser()
)
answer = parent_document_rag_chain.invoke({"question": "直立式洗衣機多久要洗一次？"})
print(answer)
