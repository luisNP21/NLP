import streamlit as st
from groq import Groq

st.set_page_config(page_title="Chat con Groq", layout="centered")
st.title("Ejercicio 2: Chat con Memoria (Groq)")

# Carga de clave
try:
    api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("No se encontró GROQ_API_KEY en .streamlit/secrets.toml")
    st.stop()

client = Groq(api_key=api_key)

# Estado de conversación
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input del usuario
if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            )
            reply = completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error con la API de Groq: {e}")
            reply = "Lo siento, ocurrió un error."

        placeholder.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
