from src.embeddings import get_embeddings
from src.llm_client import get_llm
from src.vector_db import get_vector_store
from src.chains import build_rag_chain

def demo():
    llm = get_llm()
    embeddings = get_embeddings()

    sop_vs = get_vector_store("cleaning_sop", embeddings)
    air_vs = get_vector_store("air_purifier", embeddings)

    sop_chain = build_rag_chain(llm, sop_vs, n=20, k=1)
    air_chain = build_rag_chain(llm, air_vs, n=20, k=1)

    print("sop answer:", sop_chain.invoke({"question": "洗衣機槽可以用過碳酸鈉嗎？"}))
    print("air answer:", air_chain.invoke({"question": "濾網一年要花多少錢？"}))

def main():
    demo()

if __name__ == "__main__":
    main()
