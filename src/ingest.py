from pathlib import Path
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Absolute paths
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "docs"
CHROMA_PATH = BASE_DIR / "chroma_store"
COLLECTION_NAME = "week20_domain_qa"


# Load PDFs safely
def load_pdf_as_documents(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    docs = []

    MAX_PAGE_CHARS = 8000  # page-level safety guard

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if not text.strip():
            continue

        # Skip pathological pages
        if len(text) > MAX_PAGE_CHARS:
            print(
                f"⚠️ Skipping large page {i + 1} in {pdf_path.name} "
                f"({len(text)} chars)"
            )
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": i + 1
                }
            )
        )

    return docs


# Chunk documents (Ollama-safe)
def chunk_documents(docs, chunk_size=200, overlap=40):
    chunked = []
    MAX_CHARS = 1500  # hard cap per chunk

    for d in docs:
        words = d.page_content.split()
        start = 0
        chunk_id = 0

        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunk_text = chunk_text[:MAX_CHARS]

            chunked.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        **d.metadata,
                        "chunk_id": chunk_id
                    }
                )
            )

            chunk_id += 1
            start = end - overlap

    return chunked


# Build vector database
def build_vector_db():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    # Load PDFs
    all_docs = []
    for file in DOCS_DIR.glob("*.pdf"):
        all_docs.extend(load_pdf_as_documents(file))

    if not all_docs:
        print("❌ No documents found. Check data/docs/")
        return

    # Chunk documents
    chunked_docs = chunk_documents(all_docs)

    print(f"Loaded {len(all_docs)} pages")
    print(f"Prepared {len(chunked_docs)} chunks")

    # Create / connect to Chroma
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    # Insert in SMALL batches (critical for Ollama)
    BATCH_SIZE = 8
    for i in range(0, len(chunked_docs), BATCH_SIZE):
        batch = chunked_docs[i:i + BATCH_SIZE]
        db.add_documents(batch)
        print(f"Inserted {i + len(batch)} / {len(chunked_docs)}")

    print(
        f"\n✅ Successfully stored {len(chunked_docs)} chunks "
        f"in ChromaDB at {CHROMA_PATH}"
    )


# Run
if __name__ == "__main__":
    build_vector_db()