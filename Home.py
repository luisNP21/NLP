import streamlit as st

st.set_page_config(page_title="NLP con Streamlit", layout="centered")
st.title("NLP con Streamlit: Demo de 3 Apps")

st.markdown("""
Bienvenido 👋. Usa el menú **Pages** (arriba a la izquierda) para abrir:
1. **Clasificador Zero-Shot**: etiqueta textos sin re-entrenar.
2. **Chat con Groq (memoria)**: historial en sesión.
3. **Asistente RAG**: preguntas sobre tu propio .txt.

> Tip: configura tu clave en `.streamlit/secrets.toml` para las apps que usan Groq.
""")
