from langchain_openai import ChatOpenAI
from langchain.embeddings.base import Embeddings
from openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda 
import uuid

llm_model_name = "gemma-3-12b-it"
llm_base_url = "http://localhost:1234/v1"
text_embedding_model_name = "text-embedding-bge-large-zh-v1.5"

llm = ChatOpenAI(
    model=llm_model_name,
    openai_api_key="not-needed",
    openai_api_base=llm_base_url 
)

class LmStudioEmbeddings(Embeddings):
    def __init__(self, model_name, url):
        self.model_name = model_name
        self.url = url
        self.client = OpenAI(base_url=url, api_key="lm-studio")

    def embed_query(self, text: str):
        response = self.client.embeddings.create(
            input=text,
            model=self.model_name
        )
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]):
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        return [x.embedding for x in response.data]

embeddings = LmStudioEmbeddings(model_name=text_embedding_model_name, url=llm_base_url)

loader = DirectoryLoader(
    path="data/sources/clean_sop",   
    glob="**/*.md",    
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"} 
)

documents = loader.load()
# print(f"總共有{len(documents)}筆資料")

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)


total_docs = []

# 分割父層文件
parent_docs = parent_splitter.split_documents(documents)

for doc in parent_docs:
    doc.metadata['id'] = str(uuid.uuid4())
    doc.metadata['parent_id'] = doc.metadata['id']
    # 分割子層文件
    split_docs = child_splitter.split_documents([doc])
    for sdoc in split_docs:
        sdoc.metadata['id'] = str(uuid.uuid4())
        sdoc.metadata['parent_id'] = doc.metadata['id']
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
print("成功新增新資料至 Chroma。")