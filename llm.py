import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an aviation assistant.
You answer questions ONLY based on the provided context.

GUARDRAILS:
1. If the answer is not present in the context below, you must respond EXACTLY with:
"This information is not available in the provided document(s)."
2. Do NOT use outside knowledge or hallucinate.
3. Your response must be structured as follows:
Answer: [Your answer here]
Citations: [List of citations separated by semicolon, e.g. "Book 1, Page 23; Manual, Page 10"]
"""

def ask_llm(prompt, model="openai/gpt-oss-120b"):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_completion_tokens=8192,
            top_p=1,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error communicating with LLM: {str(e)}"

def build_prompt(question, retrieved_chunks):
    if not retrieved_chunks:
        return "Context: None\n\nQuestion: " + question
        
    context_str = ""
    for i, c in enumerate(retrieved_chunks):
        context_str += f"--- Chunk {i+1} ---\nSource: {c['source']}\nPage: {c['page']}\nText: {c['text']}\n\n"

    prompt = f"""
CONTEXT:
{context_str}

QUESTION:
{question}

Based strictly on the context above, answer the question. 
Remember to provide citations in the format "Source, Page" separated by semicolon.
If the answer is not in the context, say "This information is not available in the provided document(s)."
"""
    return prompt
