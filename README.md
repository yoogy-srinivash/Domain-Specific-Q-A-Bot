## Domain Specific Q&A Bot

## architecture  
User Question
     ↓
Chroma Retriever (top-k chunks)
     ↓
Context + citations
     ↓
LangChain Prompt Template
     ↓
Local LLM (Ollama)
     ↓
Answer + Source + Confidence

## Key Features
Uses local PDF only (no external knowledge)
ChromaDB for vector storage and retrieval
Ollama for local embeddings and LLM inference
Confidence-based refusal when evidence is weak
Clear source citation (PDF name + page)
Simple Gradio-based UI

## How to run
put PDF in data/docs/ 
run python src/ingest.py 
run python src/ui_app.py 

## Evaluation
Run:python src/eval.py

This compares:
LLM answers without RAG
Answers with RAG + citations
