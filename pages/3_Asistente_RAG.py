import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="Asistente RAG", layout="wide")
st.title("Ejercicio 3: Asistente de Q&A sobre tu Documento (RAG)")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# Groq client
try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error("No se encontró GROQ_API_KEY en .streamlit/secrets.toml")
    st.stop()

def chunk_text(text: str):
    # Estrategia simple: párrafos separados por dobles saltos de línea
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return chunks

def retrieve_context(question: str, chunks, chunk_embeddings, top_k=3):
    q_emb = embedder.encode([question])
    sims = cosine_similarity(q_emb, chunk_embeddings)[0]
    idxs = np.argsort(sims)[-top_k:][::-1]
    return "\n\n".join([chunks[i] for i in idxs])

# Estado
if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.embeddings = None
    st.session_state.processed = False

with st.sidebar:
    st.header("1) Carga tu .txt")
    file = st.file_uploader("Sube un archivo .txt", type="txt")
    if file is not None and st.button("Procesar documento"):
        with st.spinner("Procesando y generando embeddings..."):
            content = file.getvalue().decode("utf-8", errors="ignore")
            st.session_state.chunks = chunk_text(content)
            st.session_state.embeddings = embedder.encode(st.session_state.chunks)
            st.session_state.processed = True
        st.success("¡Documento procesado!")

st.header("2) Pregunta sobre tu documento")
if not st.session_state.processed:
    st.warning("Carga y procesa un documento en la barra lateral.")
else:
    q = st.text_input("Escribe tu pregunta:")
    if q:
        context = retrieve_context(q, st.session_state.chunks, st.session_state.embeddings)
        prompt = f"""
INSTRUCCIÓN: Responde la pregunta basándote **ÚNICAMENTE** en el contexto.
Si la respuesta no está en el contexto, di: "No tengo información en el documento para responder.".

CONTEXTO:
{context}

PREGUNTA:
{q}
"""
        try:
            completion = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
            )
            answer = completion.choices[0].message.content
            st.info("Respuesta")
            st.write(answer)
            with st.expander("Ver contexto usado"):
                st.write(context)
        except Exception as e:
            st.error(f"Error al generar respuesta: {e}")
