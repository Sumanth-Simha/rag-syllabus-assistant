def build_rag_chain():

    # =========================
    # 🔐 LOAD ENV
    # =========================
    from dotenv import load_dotenv
    import os

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY not found!")

    os.environ["GOOGLE_API_KEY"] = api_key


    # =========================
    # 🔹 IMPORTS
    # =========================
    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.documents import Document

    import os
    import re


    # =========================
    # 🔹 LOAD DOCUMENT
    # =========================
    loader = TextLoader("knowledge_base.txt")
    documents = loader.load()

    raw_text = documents[0].page_content


    # =========================
    # 🔥 STRUCTURED CHUNKING (MODULE BASED)
    # =========================
    def split_into_modules(text):
        pattern = r"(Module\s*\d+.*?)(?=Module\s*\d+|$)"
        matches = re.findall(pattern, text, re.DOTALL)

        docs = []
        for i, m in enumerate(matches):
            docs.append(
                Document(
                    page_content=m.strip(),
                    metadata={"module": f"Module {i+1}"}
                )
            )
        return docs

    docs = split_into_modules(raw_text)


    # =========================
    # 🔹 EMBEDDINGS
    # =========================
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


    # =========================
    # 🔹 VECTOR DATABASE (SMART LOAD)
    # =========================
    persist_dir = "./chroma_db"

    if os.path.exists(persist_dir):
        vector_db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
    else:
        vector_db = Chroma.from_documents(
            docs,
            embedding_model,
            persist_directory=persist_dir
        )
        vector_db.persist()


    # =========================
    # 🔹 RETRIEVER (PRECISE)
    # =========================
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})


    # =========================
    # 🔹 GEMINI LLM
    # =========================
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.3
    )


    # =========================
    # 🔹 HELPERS
    # =========================
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    def build_prompt(data):
        return f"""
You are an AI academic assistant.

Context:
{data["context"]}

Question:
{data["question"]}

Instructions:
1. Extract ONLY the relevant module from context
2. Keep syllabus EXACT (no modification)
3. Do NOT include other modules
4. Then explain clearly

Format:

📘 Syllabus (As Provided):
<exact module text>

📖 Explanation:
<detailed explanation>
"""


    # =========================
    # 🔹 RAG CHAIN
    # =========================
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | RunnableLambda(build_prompt)
        | llm
        | StrOutputParser()
    )

    return rag_chain