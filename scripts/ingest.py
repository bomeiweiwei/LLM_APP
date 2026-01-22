from src.config import settings
from src.embeddings import get_embeddings
from src.loaders import load_markdown_dir
from src.splitters import SplitConfig, get_parent_splitter, get_child_splitter
from src.vector_db import reset_collection, get_vector_store
from src.chains import build_parent_child_docs

def main(reset: bool = True):
    embeddings = get_embeddings()

    cfg = SplitConfig()
    parent_splitter = get_parent_splitter(cfg)
    child_splitter = get_child_splitter(cfg)

    # load sources
    sop_docs = load_markdown_dir(settings.clean_sop_dir)
    air_docs = load_markdown_dir(settings.air_knowledge_dir)

    sop_total = build_parent_child_docs(sop_docs, "clean_sop", parent_splitter, child_splitter)
    air_total = build_parent_child_docs(air_docs, "air_purifier_knowledge", parent_splitter, child_splitter)

    # vector stores
    if reset:
        sop_vs = reset_collection("cleaning_sop", embeddings)
        air_vs = reset_collection("air_purifier", embeddings)
    else:
        sop_vs = get_vector_store("cleaning_sop", embeddings)
        air_vs = get_vector_store("air_purifier", embeddings)

    sop_vs.add_documents(sop_total)
    air_vs.add_documents(air_total)

    print(f"[OK] Ingested SOP docs: {len(sop_total)} into collection=cleaning_sop")
    print(f"[OK] Ingested AIR docs: {len(air_total)} into collection=air_purifier")

if __name__ == "__main__":
    main(reset=True)
