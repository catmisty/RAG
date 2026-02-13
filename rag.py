import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os


EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"

class RAGSystem:
    def __init__(self):
        self.index = None
        self.chunks = None
        self.model = None
        # self.load_resources() 

    def load_resources(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
            print("Loading FAISS index and chunks...")
            self.index = faiss.read_index(INDEX_FILE)
            with open(CHUNKS_FILE, "rb") as f:
                self.chunks = pickle.load(f)
            print("Loading embedding model...")
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        else:
            print("Index or chunks not found. Please run ingestion first.")

    def retrieve(self, query, k=5, score_threshold=1.5):
        """
        Retrieves top k chunks.
        score_threshold: FAISS uses L2 distance (lower is better).
        If distance is too high, it might be irrelevant.
        """
        if not self.index or not self.model:
            self.load_resources()
            if not self.index:
                return []

        query_embedding = self.model.encode([query])
        D, I = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(I[0]):
            if idx != -1: # FAISS returns -1 if not found
                distance = D[0][i]
                
                
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(distance)
                chunk['id'] = int(idx)
                results.append(chunk)

        return results

# Singleton instance
rag_system = RAGSystem()

def retrieve(query, k=5):
    return rag_system.retrieve(query, k)

if __name__ == "__main__":
    # Test
    query = "What is stall speed?"
    results = retrieve(query)
    for r in results:
        print(f"\nSOURCE: {r['source']} | Page: {r['page']} | Score: {r['score']:.4f}")
        print(r["text"][:200] + "...")
