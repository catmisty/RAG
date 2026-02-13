import streamlit as st
import os
import llm
# Import ingest only when needed or keep for re-ingest
import ingest

st.set_page_config(page_title="Aviation RAG Chatbot", page_icon="✈️", layout="wide")

@st.cache_resource
def get_rag_system():
    
    import rag
    # Force load resources immediately
    rag.rag_system.load_resources()
    return rag.rag_system

# Get the cached system
rag_system = get_rag_system()

st.title("✈️ Aviation RAG Chatbot")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Aviation Assistant. Ask me anything from the provided manuals."}
    ]

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    debug_mode = st.checkbox("Debug Mode (Show Retrieved Chunks)", value=False)
    
    st.divider()
    
    st.header("Data Management")
    if st.button("Re-ingest Documents"):
        with st.spinner("Ingesting documents... This may take a while."):
            result = ingest.run_ingestion()
            if result.get("status") == "error":
                st.error(result.get("message"))
            else:
                st.success(f"Ingestion complete! Indexed {result.get('chunks_count')} chunks.")
                # Clear resource cache to force reload next time
                st.cache_resource.clear()
                # Reload immediately
                get_rag_system()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            st.markdown("---")
            st.markdown("**Sources:**")
            for citation in message["citations"]:
                st.markdown(f"- {citation}")
        
        if "chunks" in message and message["chunks"]:
            with st.expander("retrieved Context (Debug)"):
                for i, chunk in enumerate(message["chunks"]):
                    st.markdown(f"**Chunk {i+1}** (Score: {chunk['score']:.4f})")
                    st.markdown(f"*Source: {chunk['source']} (Page {chunk['page']})*")
                    st.text(chunk['text'])
                    st.divider()

# Chat Input
if prompt := st.chat_input("What is Vso?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching manuals..."):
            try:
                # 1. Retrieve
                # Use the cached instance directly
                chunks = rag_system.retrieve(prompt, k=5)
                
                # 2. Build Prompt
                llm_prompt = llm.build_prompt(prompt, chunks)
                
                # 3. Ask LLM
                response_text = llm.ask_llm(llm_prompt)
                
                # 4. Parse Response
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

                # Display Answer
                st.markdown(answer)
                
                # Display Citations
                if citations:
                    st.markdown("---")
                    st.markdown("**Sources:**")
                    for citation in citations:
                        st.markdown(f"- {citation}")
                
                # Display Debug Chunks
                if debug_mode and chunks:
                    with st.expander("retrieved Context (Debug)"):
                        for i, chunk in enumerate(chunks):
                            st.markdown(f"**Chunk {i+1}** (Score: {chunk.get('score', 0):.4f})")
                            st.markdown(f"*Source: {chunk['source']} (Page {chunk['page']})*")
                            st.text(chunk['text'])
                            st.divider()

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "citations": citations,
                    "chunks": chunks if debug_mode else None
                })
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
