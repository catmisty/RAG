from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("rag.retrieve")
@patch("llm.ask_llm")
def test_ask(mock_ask_llm, mock_retrieve):
    
    mock_retrieve.return_value = [
        {"text": "Vso is the stall speed in landing configuration.", "source": "book1.pdf", "page": 10, "score": 0.1},
        {"text": "Vne is never exceed speed.", "source": "book1.pdf", "page": 12, "score": 0.2}
    ]
    
    
    mock_ask_llm.return_value = "Answer: Vso is the stall speed in landing configuration.\nCitations: book1.pdf, Page 10"
    
    response = client.post("/ask", json={"question": "What is Vso?", "debug": True})
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Vso is the stall speed in landing configuration."
    assert "book1.pdf, Page 10" in data["citations"]
    assert len(data["retrieved_chunks"]) == 2

@patch("rag.retrieve")
@patch("llm.ask_llm")
def test_ask_refusal(mock_ask_llm, mock_retrieve):
    # Mock retrieval with irrelevant chunks
    mock_retrieve.return_value = []
    
    # Mock LLM response
    mock_ask_llm.return_value = "This information is not available in the provided document(s)."
    
    response = client.post("/ask", json={"question": "How to cook pasta?"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This information is not available in the provided document(s)."
    assert data["citations"] == []

import traceback

if __name__ == "__main__":
    # Manually run tests if pytest not available
    try:
        test_health()
        test_ask()
        test_ask_refusal()
        print("All tests passed!")
    except Exception as e:
        print(f"Tests failed: {e}")
        traceback.print_exc()
