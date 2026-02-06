# LLM_APP - Local RAG QA System  
（居家清潔 / 空氣清淨機知識問答系統）

本專案為一個 **本地端 LLM + RAG (Retrieval-Augmented Generation)** 問答系統，  
使用 **LM Studio 本地模型** 進行推理，不依賴外部雲端 API。

專案主題來自於個人實際需求：  
在研究與購買空氣清淨機過程中，整理大量居家清潔與空氣淨化相關知識，  
並將其轉換為可查詢的知識問答系統。

---

## 🚀 技術棧

- Python
- LangChain
- Chroma (Vector Database)
- Gradio (Web UI)
- LM Studio (Local LLM Inference)

---

## 🧠 系統架構流程

1. 將 Word / Markdown 文件整理為知識來源  
2. 使用 TextSplitter 進行 Chunk 切分  
3. 產生 Embeddings  
4. 儲存至 Chroma 向量資料庫  
5. 啟動 Retrieval Chain (Top-K 檢索)  
6. 將檢索結果組合 Prompt  
7. 呼叫 LM Studio 本地模型生成回應  
8. 透過 Gradio 提供 Web UI  

---

## 📂 專案結構

```
LLM_APP/
│
├─ data/
│   ├─ drafts/
│   └─ sources/
│       ├─ air_purifier_knowledge/
│       └─ clean_sop/
│
├─ scripts/
│   └─ ingest.py
│
├─ src/
│   ├─ chains.py
│   ├─ config.py
│   ├─ embeddings.py
│   ├─ llm_client.py
│   ├─ loaders.py
│   ├─ retrieval.py
│   ├─ splitters.py
│   └─ vector_db.py
│
├─ vectorstore/        (向量資料庫，可重建)
├─ app.py
└─ requirements.txt
```

---

## 🛠 環境需求

- Python 3.10+
- LM Studio
- 已下載本地模型（例如 Gemma / Llama / Mistral 等）

---

## ⚙️ LM Studio 設定方式

1. 開啟 LM Studio  
2. 載入模型  
3. 開啟 **OpenAI-compatible server**  
4. 記下：
   - Base URL（例如 `http://127.0.0.1:1234/v1`）
   - Model Name  

---

## 🔧 安裝與啟動

### 1️⃣ 安裝套件

```bash
pip install -r requirements.txt
```

### 2️⃣ 建立向量資料庫

```bash
python scripts/ingest.py
```

### 3️⃣ 啟動應用程式

```bash
python app.py
```

瀏覽器將自動開啟 Gradio UI。

---

## ✨ 功能特色

- 雙知識域（清潔 SOP / 空氣清淨機知識）
- 快速按鈕輔助提問
- 僅依據指定資料來源回答（降低幻覺）
- 本地模型推理，保障資料隱私
- 模組化設計，方便擴充

---

## 📊 RAG 設計重點

- Chunk-based Retrieval
- Top-K Similarity Search
- Prompt 組合控制回答範圍
- 僅使用指定知識來源

---

## 🔒 資料隱私

- 不上傳資料至第三方 API
- 所有推理在本機 LM Studio 執行
- 向量資料庫為本地 Chroma

---

## 🧪 未來優化方向

- 顯示引用來源 (Source Citation)
- 增加簡易評估資料集
- 加入 Docker 部署版本
- 改進 Chunk Strategy

---

## 📌 專案目的

本專案旨在實作一個可控資料來源的本地端 RAG 系統，  
理解向量資料庫建構流程與檢索生成架構，  
並探索如何降低 LLM 幻覺問題。

---

## 👤 作者

HsinWei Chung  
Backend Engineer (.NET) / AI RAG Practice
