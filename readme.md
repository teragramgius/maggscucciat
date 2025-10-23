# EU-Policy RAG Assistant

RAG assistant for Horizon Europe and related EU guidelines.  
Built with **LangChain**, **Chroma** and **Streamlit**.

---

## Features
- PDF ingestion with chunk size 512 / overlap 100
- Metadata filtering (programme, year, section)
- Re-ranking pass and source citations
- Simple Streamlit UI

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill your keys
streamlit run src/app.py