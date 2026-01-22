from langchain_community.document_loaders import DirectoryLoader, TextLoader


def load_markdown_dir(path: str):
    loader = DirectoryLoader(
        path=path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()
