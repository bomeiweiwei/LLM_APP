# LLM_APP

本專案是一個以 **Python + LangChain + Chroma + Gradio** 建立的 LLM 應用，
主要提供兩類知識問答：

- 清潔 SOP
- 空氣清淨機知識框架

系統使用 RAG（Retrieval-Augmented Generation）方式，
僅根據指定資料來源進行回答。

---

## 專案結構

```text
LLM_APP/
├─ data/
│  ├─ drafts/
│  └─ sources/
│     ├─ air_purifier_knowledge/
│     └─ clean_sop/
│
├─ scripts/
│  └─ ingest.py
│
├─ src/
│  ├─ chains.py
│  ├─ config.py
│  ├─ embeddings.py
│  ├─ llm_client.py
│  ├─ loaders.py
│  ├─ retrieval.pyя
│  ├─ splitters.py
│  └─ vector_db.py
│
├─ vectorstore/
│
├─ app.py
├─ requirements.txt
└─ README.md
