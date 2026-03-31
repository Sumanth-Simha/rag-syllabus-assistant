import streamlit as st
import shutil

from rag_pipeline import build_rag_chain

rag_chain = build_rag_chain()


st.title("🧠 RAG Chatbot")

query = st.text_input("Ask something:")

if query:
    response = rag_chain.invoke(query)
    st.write(response)


# 🔥 REBUILD BUTTON (WORKING)
if st.button("🔄 Rebuild Knowledge Base"):
    shutil.rmtree("chroma_db", ignore_errors=True)
    st.success("Rebuilding knowledge base...")
    st.rerun()