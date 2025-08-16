import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

# 🔹 Load Groq API key from environment variable
load_dotenv()
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# 🔹 ChromaDB setup
client = chromadb.PersistentClient(path="../chroma_data")
collection_name = "med"

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Retrieve collection
try:
    collection = client.get_collection(name=collection_name)
except Exception:
    st.error(f"Collection '{collection_name}' not found.")
    st.stop()

# 🔹 Streamlit UI
st.set_page_config(page_title="Groq + Chroma Chat", page_icon="💬")
st.title("💬 Chat with Your Chroma Data via Groq LLM")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

# Input box
if prompt := st.chat_input("Ask me something..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Step 1: Embed query & search Chroma
    query_embedding = embedding_model.encode([prompt]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=5)

    retrieved_docs = [doc for doc in results["documents"][0]]
    context_text = "\n".join(retrieved_docs)

    # Step 2: Send to Groq    
    full_prompt = (
    "You are a medical expert. Using the context below, answer the query clearly and concisely. "
    "If the answer is not in the context, say 'I don’t have enough information.'\n\n"
    f"Context:\n{context_text}\n\nQuestion: {prompt}")

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    # Step 3: Display assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
