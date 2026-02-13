import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY55 not found in .env file")

# Create Groq client
client = Groq(api_key=api_key)

print("🤖 Groq Chatbot Started")
print("Type exit() to stop\n")

# Chat history (optional but recommended)
messages = []

while True:
    user_input = input("You: ")

    if user_input.strip() == "exit()":
        print("👋 Exiting chatbot...")
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=1,
        max_completion_tokens=8192,
        top_p=1,
        stream=True,
    )

    print("Bot: ", end="", flush=True)

    full_response = ""

    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            full_response += content

    print("\n")

    messages.append({
        "role": "assistant",
        "content": full_response
    })
