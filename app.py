import gradio as gr

from src.embeddings import get_embeddings
from src.llm_client import get_llm
from src.vector_db import get_vector_store
from src.chains import build_rag_chain

# 初始化
def init_chains():
    llm = get_llm()
    embeddings = get_embeddings()

    sop_vs = get_vector_store("cleaning_sop", embeddings)
    air_vs = get_vector_store("air_purifier", embeddings)

    sop_chain = build_rag_chain(llm, sop_vs, n=20, k=1)
    air_chain = build_rag_chain(llm, air_vs, n=20, k=1)

    return sop_chain, air_chain

SOP_CHAIN, AIR_CHAIN = init_chains()

# UI
def fn_cleaning_sop(msg: str) -> str:
    msg = (msg or "").strip()
    if not msg:
        return "請輸入問題。"
    return SOP_CHAIN.invoke({"question": msg})

def fn_air_purifier(msg: str) -> str:
    msg = (msg or "").strip()
    if not msg:
        return "請輸入問題。"
    return AIR_CHAIN.invoke({"question": msg})

with gr.Blocks(title="LLM 應用") as demo:
    gr.Markdown("## LLM 應用（居家清潔）")

    with gr.Tabs():
        with gr.Tab("清潔SOP"):
            sop_input = gr.Textbox(
                label="輸入你對清潔SOP的問題",
                placeholder="例如：洗衣機槽可以用過碳酸鈉嗎？",
            )
            sop_button = gr.Button("送出")
            sop_output = gr.Textbox(label="Output", lines=8)

            sop_button.click(fn=fn_cleaning_sop, inputs=sop_input, outputs=sop_output)
            sop_input.submit(fn=fn_cleaning_sop, inputs=sop_input, outputs=sop_output)

        with gr.Tab("空氣清淨機知識"):
            air_input = gr.Textbox(
                label="輸入你對空氣清淨機知識的問題",
                placeholder="例如：HEPA 和活性碳各自解決什麼問題？",
            )
            air_button = gr.Button("送出")
            air_output = gr.Textbox(label="Output", lines=8)

            air_button.click(fn=fn_air_purifier, inputs=air_input, outputs=air_output)
            air_input.submit(fn=fn_air_purifier, inputs=air_input, outputs=air_output)

if __name__ == "__main__":
    demo.launch()