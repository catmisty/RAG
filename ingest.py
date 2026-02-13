import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

DATA_FOLDER = "data"
# Consistency: match this with rag.py
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

def load_documents():
    documents = []
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_FOLDER, file)
            print(f"Loading {file}...")
            try:
                reader = PdfReader(path)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        documents.append({
                            "text": text,
                            "source": file,
                            "page": page_num + 1
                        })
            except Exception as e:
                print(f"Error reading {file}: {e}")

    return documents


def chunk_text(text, chunk_size=800, overlap=100):
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    
    
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        
        
        if end < text_len:
            excerpt = text[start:end]
            last_space = excerpt.rfind(' ')
            if last_space != -1 and last_space > chunk_size * 0.7: 
               
                end = start + last_space
        
        chunk = text[start:end].strip()
        if len(chunk) > 50: # Filter tiny chunks
            chunks.append(chunk)
        
        start = end - overlap # Move back by overlap
        
        # Prevent infinite loop if overlap >= chunk_size 
        if start >= end:
            start = end

    return chunks


def create_chunks(documents):
    all_chunks = []
    print(f"Chunking {len(documents)} pages...")
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": doc["source"],
                "page": doc["page"]
            })
    return all_chunks


def build_faiss(chunks):
    print(f"Encoding {len(chunks)} chunks with {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    if not chunks:
        print("No chunks to index.")
        return None, model

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    return index, model


def save_index(index, chunks):
    if index is None:
        return
        
    faiss.write_index(index, "faiss_index.bin")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("Index saved to disk.")

def run_ingestion():
    print("Starting ingestion pipeline...")
    documents = load_documents()
    if not documents:
        print("No documents found in 'data/' folder.")
        return {"status": "error", "message": "No documents found"}
        
    chunks = create_chunks(documents)
    index, model = build_faiss(chunks)
    save_index(index, chunks)
    print("Ingestion complete!")
    return {"status": "success", "chunks_count": len(chunks)}

if __name__ == "__main__":
    run_ingestion()
