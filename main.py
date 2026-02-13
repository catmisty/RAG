from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import ingest
import rag
import llm
import os

app = FastAPI(title="Aviation RAG Chatbot")


if not os.path.exists("static"):
    os.makedirs("static")


app.mount("/static", StaticFiles(directory="static"), name="static")

class AskRequest(BaseModel):
    question: str
    debug: bool = False

class ChunkInfo(BaseModel):
    text: str
    source: str
    page: int
    score: float

class AskResponse(BaseModel):
    answer: str
    citations: List[str]
    retrieved_chunks: Optional[List[ChunkInfo]] = None

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.post("/ingest")
async def trigger_ingestion(background_tasks: BackgroundTasks):
    """
    Triggers document ingestion in the background.
    """
    background_tasks.add_task(ingest.run_ingestion)
    return {"message": "Ingestion started in background"}

@app.get("/ask")
async def ask_get():
    return {"message": "Use POST to submit questions, or visit http://127.0.0.1:8000/ to use the Chat UI."}

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Asks a question to the RAG system.
    """
    try:
        # 1. Retrieve chunks
        chunks = rag.retrieve(request.question, k=5)
        
        # 2. Build prompt
        prompt = llm.build_prompt(request.question, chunks)
        
        # 3. Ask LLM
        response_text = llm.ask_llm(prompt)
        
        # 4. Parse response
        answer = response_text
        citations = []
        
        if "Citations:" in response_text:
            parts = response_text.split("Citations:")
            answer = parts[0].replace("Answer:", "").strip()
            citations_text = parts[1].strip()
            citations = [c.strip() for c in citations_text.split(";") if c.strip()]
        else:
             answer = response_text.replace("Answer:", "").strip()

        if "This information is not available" in answer:
            answer = "This information is not available in the provided document(s)."
            citations = []
        
        retrieved_chunks = None
        if request.debug:
            retrieved_chunks = [
                ChunkInfo(
                    text=c['text'], 
                    source=c['source'], 
                    page=c['page'], 
                    score=c.get('score', 0.0)
                ) for c in chunks
            ]
            
        return AskResponse(
            answer=answer, 
            citations=citations, 
            retrieved_chunks=retrieved_chunks
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
