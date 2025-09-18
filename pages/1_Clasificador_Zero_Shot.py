import streamlit as st
from transformers import pipeline
import pandas as pd

st.set_page_config(page_title="Zero-Shot", layout="wide")
st.title("Ejercicio 1: Clasificador Zero-Shot")

@st.cache_resource
def load_classifier():
    # Modelo recomendado en el documento: facebook/bart-large-mnli
    # Si tu equipo tiene poca RAM, prueba: "valhalla/distilbart-mnli-12-3"
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_classifier()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Texto a clasificar")
    text = st.text_area(
        "Ingresa el texto:",
        "Apple is set to release the new iPhone 16 this fall with advanced AI features.",
        height=200
    )
    st.subheader("Etiquetas candidatas")
    labels_str = st.text_input(
        "Separadas por comas:",
        "tecnología, política, deportes, negocios"
    )

with col2:
    st.subheader("Resultados")
    if st.button("Clasificar texto"):
        if text and labels_str.strip():
            labels = [e.strip() for e in labels_str.split(",") if e.strip()]
            with st.spinner("Clasificando..."):
                result = classifier(text, labels)
            df = pd.DataFrame({"Etiqueta": result["labels"], "Score": result["scores"]})
            st.bar_chart(df, x="Etiqueta", y="Score")
            st.success(f"Etiqueta más probable: **{result['labels'][0]}**")
            with st.expander("Ver respuesta completa del modelo"):
                st.json(result)
        else:
            st.warning("Por favor, ingresa texto y etiquetas.")
