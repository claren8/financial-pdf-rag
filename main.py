import os
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
import PyPDF2
from embeddings import get_embedding
import pickle

# ==============================
# Configuración inicial
# ==============================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PDF_PATH = "sample.pdf"  # cambia el nombre si tu PDF es distinto
CHUNK_SIZE = 500

# Variables globales (se cargan una sola vez)
chunks = []
chunk_embeddings = []

CACHE_FILE = "embeddings_cache.pkl"


# ==============================
# Utilidades
# ==============================

def read_pdf(path):
    text = ""
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def split_text(text, size=CHUNK_SIZE):
    return [text[i:i+size] for i in range(0, len(text), size)]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ==============================
# Carga inicial (se ejecuta una vez)
# ==============================

def load_document():
    global chunks, chunk_embeddings

    # Si existe cache → cargar
    if os.path.exists(CACHE_FILE):
        print("Cargando embeddings desde cache...")
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
            chunks = data["chunks"]
            chunk_embeddings = data["embeddings"]
        print("Cache cargado.")
        return

    # Si no existe → procesar PDF
    print("Cargando PDF...")
    text = read_pdf(PDF_PATH)

    print("Dividiendo en chunks...")
    chunks = split_text(text)

    print("Generando embeddings (esto ocurre solo una vez)...")
    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

    # Guardar cache
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "embeddings": chunk_embeddings
        }, f)

    print("Embeddings guardados en cache.")

# ==============================
# Función principal para consultas
# ==============================

def ask_question(question: str):
    # Embedding de la pregunta
    question_embedding = get_embedding(question)

    # Buscar los chunks más similares
    similarities = [
        cosine_similarity(question_embedding, emb)
        for emb in chunk_embeddings
    ]

    # Top 3 chunks
    top_indices = np.argsort(similarities)[-3:][::-1]
    context = "\n\n".join([chunks[i] for i in top_indices])

    # Prompt
    prompt = f"""
Responde la pregunta usando SOLO la información del contexto.

Contexto:
{context}

Pregunta:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content


# ==============================
# Modo consola (opcional)
# ==============================

if __name__ == "__main__":
    load_document()
    print("Modo consola. Escribe 'salir' para terminar.")
    while True:
        q = input("\nPregunta: ")
        if q.lower() == "salir":
            break
        answer = ask_question(q)
        print("\nRespuesta:", answer)