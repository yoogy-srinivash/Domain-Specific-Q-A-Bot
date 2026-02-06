from pathlib import Path
import re

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Absolute paths (MUST match ingest.py)
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = BASE_DIR / "chroma_store"
COLLECTION_NAME = "week20_domain_qa"


# Format retrieved docs for prompt
def format_docs(docs):
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "NA")
        chunk_id = d.metadata.get("chunk_id", "NA")

        blocks.append(
            f"[Source: {src} | Page: {page} | Chunk: {chunk_id}]\n"
            f"{d.page_content}"
        )

    return "\n\n".join(blocks)


# Simple, robust confidence check
def simple_confidence(question: str, docs):
    if not docs:
        return 0.0

    q_words = [
        w.lower()
        for w in re.findall(r"\b[a-zA-Z]{4,}\b", question)
    ]

    if not q_words:
        return 0.0

    joined = " ".join(d.page_content for d in docs).lower()
    hits = sum(1 for w in q_words if w in joined)

    return hits / len(q_words)

# Main RAG function
def get_rag_answer(question: str, top_k: int = 2):
    # Embeddings (must match ingest)
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    # Vector DB
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    retriever = db.as_retriever(search_kwargs={"k": top_k})

    # Retrieve documents
    docs = retriever.invoke(question)
    print("Retrieved docs:", len(docs))

    # 🚫 Hard stop: no retrieval → no answer
    if not docs:
        return (
            "⚠️ I couldn’t find any relevant information in your documents.",
            docs,
            0.0
        )

    # Confidence check
    confidence = simple_confidence(question, docs)

    if confidence < 0.3:
        return (
            "I’m not confident because the documents don’t strongly match your question.",
            docs,
            confidence
        )

    # Build context (CAP SIZE for speed)
    MAX_CONTEXT_CHARS = 2500
    context = format_docs(docs)[:MAX_CONTEXT_CHARS]

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. "
        "Answer ONLY using the provided context. "
        "If the answer is not in the context, say: 'I don't know.' "
        "You MUST end your response with exactly ONE line in this format:\n"
        "Source: <filename>, Page <number>"
    ),
    (
        "user",
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )
])

    # ⚡ FAST local LLM (critical change)
    llm = ChatOllama(
        model="phi3:mini",
        temperature=0,
        num_predict=150,
        top_k=20,
        top_p=0.9
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer, docs, confidence