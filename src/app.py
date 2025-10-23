import streamlit as st
from rag import answer_query, ingest_pdf_paths

st.set_page_config(page_title="EU-Policy RAG Assistant", page_icon="📘")
st.title("📘 EU-Policy RAG Assistant")

with st.expander("➕ (Facoltativo) Ingesta PDF"):
    uploaded = st.file_uploader("Carica uno o più PDF", type=["pdf"], accept_multiple_files=True)
    if st.button("Ingesta nei vettori") and uploaded:
        # Salva temporaneamente e indicizza
        paths = []
        for f in uploaded:
            p = f"data/sample/{f.name}"
            with open(p, "wb") as out:
                out.write(f.read())
            paths.append(p)
        n = ingest_pdf_paths(paths)
        st.success(f"Ingest riuscita. Chunks indicizzati: {n}")

st.subheader("🔎 Fai una domanda")
q = st.text_input("Esempi: 'Quali sono i criteri di eleggibilità HEU?'")
if st.button("Cerca") and q:
    with st.spinner("Sto cercando nelle fonti…"):
        ans, sources = answer_query(q)
    st.markdown("### ✅ Risposta")
    st.write(ans)
    if sources:
        st.markdown("### 📚 Fonti")
        for s in sources:
            st.write(f"- `{s}`")

st.caption("Tip: popola la cartella `data/sample/` con PDF di policy per risultati migliori.")