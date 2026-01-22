from langchain_community.document_loaders import DirectoryLoader 
from langchain_community.document_loaders import TextLoader

loader = DirectoryLoader(
    path="data/sources",   
    glob="**/*.md",    
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"} 
)

documents = loader.load()
print(documents[0])