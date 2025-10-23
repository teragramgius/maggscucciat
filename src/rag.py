import os
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA


# Persistent DB directory
DB_DIR = os.getenv("CHROMA_DB_DIR", ".chroma")

# Load API configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY. Please set it in your .env file.")

def _get_db() -> Chroma:
    os.makedirs(DB_DIR, exist_ok=True)
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL  # ✅ correct argument
        )
    )

def ingest_pdf_paths(paths: List[str]) -> int:
    """
    Carica più PDF, li splitta in chunk e li aggiunge alla Chroma DB.
    Ritorna il numero di chunk indicizzati.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    all_chunks = []
    for p in paths:
        loader = PyPDFLoader(p)
        docs = loader.load()
        # Aggiungiamo metadati per tracciare la fonte
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata.update({"source": os.path.basename(p)})
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    db = _get_db()
    db.add_documents(all_chunks)
    db.persist()
    return len(all_chunks)


def answer_query(q: str) -> Tuple[str, List[str]]:
    """
    Esegue RetrievalQA sui documenti indicizzati e ritorna (risposta, fonti).
    """
    db = _get_db()
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # Use OpenRouter-compatible LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # You can also try "mistralai/mixtral-8x7b" or "anthropic/claude-3.5-sonnet"
        temperature=0,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
    )

    result = chain.invoke({"query": q})

    sources = []
    for d in result.get("source_documents", []) or []:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        if src not in sources:
            sources.append(src)

    return result.get("result", ""), sources


""" #test
from langchain_openai import OpenAIEmbeddings
e = OpenAIEmbeddings(model="text-embedding-3-small", api_key="YOUR_KEY", base_url="https://api.openai.com/v1")
print(e.embed_query("Hello world"))
"""