from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class SplitConfig:
    parent_chunk_size: int = 1500
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 200
    child_chunk_overlap: int = 50


def get_parent_splitter(cfg: SplitConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.parent_chunk_size,
        chunk_overlap=cfg.parent_chunk_overlap,
    )


def get_child_splitter(cfg: SplitConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.child_chunk_size,
        chunk_overlap=cfg.child_chunk_overlap,
    )
