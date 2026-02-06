import gradio as gr
from rag_chain import get_rag_answer


def chat_fn(user_question):
    answer, docs, conf = get_rag_answer(user_question, top_k=2)
    return f"{answer}\n\n(Confidence: {conf:.2f})"


demo = gr.Interface(
    fn=chat_fn,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask something from your documents..."
    ),
    outputs="text",
    title="Week 20 Domain Q&A Bot",
    description="Ask questions from your private PDFs. The bot answers with sources."
)


if __name__ == "__main__":
    demo.launch()
