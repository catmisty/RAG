# Aviation RAG Chatbot

An AI-powered chatbot that answers questions strictly from provided aviation documents (textbooks, SOPs, manuals).
Designed for reliability, traceability, and hallucination control.

## Features
- **RAG Pipeline**: PDF ingestion, chunking, and FAISS vector storage using `all-MiniLM-L6-v2`.
- **Strict Grounding**: Answers are grounded in retrieved context. If information is missing, it explicitly refuses to answer.
- **Citations**: Provides source document and page number for every answer.
- **Evaluation**: Includes tools to generate questions and evaluate the system's performance.

## Tech Stack
- **Language**: Python 3.10+
- **API**: FastAPI
- **Vector DB**: FAISS
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Llama3-8b via Groq API

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=gsk_...
   ```

3. **Ingest Documents**
   Place your PDFs in the `data/` folder and run:
   ```bash
   python ingest.py
   ```
   Or trigger via API: `POST /ingest`

4. **Run API**
   ```bash
   uvicorn main:app --reload
   ```

## API Usage

**Ask a Question:**
```http
POST /ask
{
  "question": "What is Vso?",
  "debug": true
}
```

## Evaluation

1. **Generate Questions** (Optional, requires ingested chunks):
   ```bash
   python generate_questions.py
   ```

2. **Run Evaluation**:
   ```bash
   python evaluate.py
   ```
   Generates `report.md` and `evaluation_results.csv`.
