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
            # ---- 預設問題按鈕 ----
            with gr.Row():
                sop_btn_1 = gr.Button("洗衣機槽可以用過碳酸鈉嗎？")
                sop_btn_2 = gr.Button("浴室牆角發霉要怎麼處理？")
                sop_btn_3 = gr.Button("冰箱除臭有哪些安全的方法？")

            sop_input = gr.Textbox(
                label="輸入你對清潔SOP的問題",
                placeholder="也可以自行輸入問題",
            )
            sop_button = gr.Button("送出")
            sop_output = gr.Textbox(label="Output", lines=8)

            # ---- 預設問題按鈕 ----
            sop_btn_1.click(
                fn=fn_cleaning_sop,
                inputs=gr.State("洗衣機槽可以用過碳酸鈉嗎？"),
                outputs=sop_output,
            )
            sop_btn_2.click(
                fn=fn_cleaning_sop,
                inputs=gr.State("浴室牆角發霉要怎麼處理？"),
                outputs=sop_output,
            )
            sop_btn_3.click(
                fn=fn_cleaning_sop,
                inputs=gr.State("冰箱除臭有哪些安全的方法？"),
                outputs=sop_output,
            )

            sop_button.click(fn=fn_cleaning_sop, inputs=sop_input, outputs=sop_output)
            sop_input.submit(fn=fn_cleaning_sop, inputs=sop_input, outputs=sop_output)

        with gr.Tab("空氣清淨機知識"):
            # ---- 預設問題按鈕 ----
            with gr.Row():
                air_btn_1 = gr.Button("HEPA 和活性碳各自解決什麼問題？")
                air_btn_2 = gr.Button("空氣清淨機應該放在哪裡效果最好？")
                air_btn_3 = gr.Button("濾網一年大概要花多少錢？")

            air_input = gr.Textbox(
                label="輸入你對空氣清淨機知識的問題",
                placeholder="也可以自行輸入問題",
            )
            air_button = gr.Button("送出")
            air_output = gr.Textbox(label="Output", lines=8)

            # ---- 預設問題按鈕 ----
            air_btn_1.click(
                fn=fn_air_purifier,
                inputs=gr.State("HEPA 和活性碳各自解決什麼問題？"),
                outputs=air_output,
            )
            air_btn_2.click(
                fn=fn_air_purifier,
                inputs=gr.State("空氣清淨機應該放在哪裡效果最好？"),
                outputs=air_output,
            )
            air_btn_3.click(
                fn=fn_air_purifier,
                inputs=gr.State("濾網一年大概要花多少錢？"),
                outputs=air_output,
            )

            air_button.click(fn=fn_air_purifier, inputs=air_input, outputs=air_output)
            air_input.submit(fn=fn_air_purifier, inputs=air_input, outputs=air_output)

if __name__ == "__main__":
    demo.launch()
