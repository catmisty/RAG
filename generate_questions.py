import pickle
import random
import json
import os
from llm import ask_llm

CHUNKS_FILE = "chunks.pkl"
OUTPUT_FILE = "questions.json"

def generate_questions():
    if not os.path.exists(CHUNKS_FILE):
        print("Chunks file not found. Run ingestion first.")
        return

    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    
    print(f"Loaded {len(chunks)} chunks. Sampling for question generation...")
    
    questions = []
    
    # Categories
    categories = {
        "factual": 20,
        "applied": 20,
        "reasoning": 10
    }
    
    # We will sample random chunks and ask LLM to generate questions
    # To get 50 questions, we might need to try multiple times or batch chunks
    
    total_needed = 50
    generated_count = 0
    
    # Simple strategy: iterate and generate
    # Shuffle chunks to get random coverage
    random.shuffle(chunks)
    
    for category, count in categories.items():
        print(f"Generating {count} {category} questions...")
        current_cat_count = 0
        
        while current_cat_count < count:
            if not chunks:
                break
                
            chunk = chunks.pop()
            text = chunk["text"]
            
            prompt = f"""
            You are an expert aviation instructor.
            Based strictly on the following text, generate 1 {category} question.
            
            Definition of {category} question:
            - factual: Simple definition or lookup.
            - applied: Scenario-based, operational or procedural.
            - reasoning: Multi-step, trade-offs, conditional logic.
            
            Text:
            {text[:1500]}
            
            Output ONLY the question.
            """
            
            try:
                question = ask_llm(prompt)
                if question and len(question) > 10:
                    questions.append({
                        "question": question.strip(),
                        "category": category,
                        "source_chunk_id": chunk.get("source", "unknown")
                    })
                    current_cat_count += 1
                    print(f"Generated: {question.strip()[:50]}...")
                time.sleep(1)
            except Exception as e:
                print(f"Error: {e}")
                
    with open(OUTPUT_FILE, "w") as f:
        json.dump(questions, f, indent=2)
    
    print(f"Saved {len(questions)} questions to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_questions()
