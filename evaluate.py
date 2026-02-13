import json
import requests
import pandas as pd
from tqdm import tqdm
import time

# Configuration
API_URL = "http://127.0.0.1:8000"
QUESTIONS_FILE = "questions.json"
REPORT_FILE = "report.md"

def evaluate():
    print("Loading questions...")
    with open(QUESTIONS_FILE, "r") as f:
        questions = json.load(f)

    results = []
    
    print(f"Evaluating {len(questions)} questions...")
    for q in tqdm(questions):
        question_text = q["question"]
        category = q.get("category", "general")
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{API_URL}/ask", 
                json={"question": question_text, "debug": True}
            )
            response.raise_for_status()
            data = response.json()
            
            latency = time.time() - start_time
            
            answer = data["answer"]
            citations = data["citations"]
            retrieved_chunks = data["retrieved_chunks"]
            
            # Metrics
            # 1. Retrieval Hit Rate (Approximation: Did we get chunks?)
            # Ideally we compare with ground truth source, but here we just check if chunks were returned.
            retrieval_success = len(retrieved_chunks) > 0
            
            # 2. Hallucination Check (Basic) - Did it refuse to answer?
            refusal = "not available in the provided document" in answer
            
            results.append({
                "question": question_text,
                "category": category,
                "answer": answer,
                "citations": citations,
                "latency": latency,
                "retrieved_chunks_count": len(retrieved_chunks),
                "refusal": refusal
            })
            
        except Exception as e:
            print(f"Error processing question '{question_text}': {e}")
            results.append({
                "question": question_text,
                "error": str(e)
            })

    # Save detailed results
    pd.DataFrame(results).to_csv("evaluation_results.csv", index=False)
    
    # Generate Report
    generate_report(results)

def generate_report(results):
    df = pd.DataFrame(results)
    
    total = len(df)
    answered = df[~df["refusal"]].shape[0] if "refusal" in df.columns else 0
    refused = df[df["refusal"]].shape[0] if "refusal" in df.columns else 0
    avg_latency = df["latency"].mean() if "latency" in df.columns else 0
    
    report = f"""# RAG Evaluation Report

## Summary
- **Total Questions**: {total}
- **Answered**: {answered}
- **Refused**: {refused}
- **Average Latency**: {avg_latency:.2f}s

## metrics by Category
"""
    if "category" in df.columns:
        cat_stats = df.groupby("category").apply(
            lambda x: pd.Series({
                "count": len(x),
                "answered": (~x["refusal"]).sum(),
                "refusal_rate": x["refusal"].mean()
            })
        )
        report += cat_stats.to_markdown()

    report += "\n\n## Qualitative Analysis\n(To be filled manually or with LLM-as-a-judge)\n"

    with open(REPORT_FILE, "w") as f:
        f.write(report)
    
    print(f"Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    evaluate()
