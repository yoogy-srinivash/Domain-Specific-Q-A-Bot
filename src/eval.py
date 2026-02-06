from langchain_ollama import ChatOllama
from rag_chain import get_rag_answer

# Local LLM (NO RAG)
llm = ChatOllama(
    model="phi3:mini",
    temperature=0
)

# Evaluation questions
EVAL_QUESTIONS = [
    "What is supervised learning?",
    "What is labeled data?",
    "What is the goal of supervised learning?",
    "What is the difference between classification and regression?",
    "Give an example of supervised learning."
]

# Without RAG
def answer_without_rag(question: str):
    response = llm.invoke(question)
    return response.content


# Run evaluation
def run_eval():
    for q in EVAL_QUESTIONS:
        print("\nQUESTION:")
        print(q)

        # No RAG
        no_rag_answer = answer_without_rag(q)

        # With RAG
        rag_answer, docs, conf = get_rag_answer(q)

        print("\n--- WITHOUT RAG (LLM only)  ---")
        print(no_rag_answer)

        print("\n--- WITH RAG (Docs + LLM) ---")
        print(rag_answer)
        print(f"\nConfidence: {conf:.2f}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    run_eval()
