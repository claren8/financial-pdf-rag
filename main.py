import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

from pdf_reader import read_pdf
from embeddings import get_embedding

load_dotenv()
client = OpenAI()

PDF_PATH = "sample.pdf"
CHUNK_SIZE = 500


def split_text(text, size):
    return [text[i:i+size] for i in range(0, len(text), size)]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


print("Leyendo PDF...")
text = read_pdf(PDF_PATH)

chunks = split_text(text, CHUNK_SIZE)

print("Creando embeddings del documento...")
chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

print("Listo. Puedes hacer preguntas.\n")

while True:
    question = input("Pregunta: ")

    if question.lower() == "salir":
        break

    question_embedding = get_embedding(question)

    similarities = [
        cosine_similarity(question_embedding, emb)
        for emb in chunk_embeddings
    ]

    best_index = np.argmax(similarities)
    context = chunks[best_index]

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=f"""
        Usa el siguiente contexto para responder la pregunta.

        Contexto:
        {context}

        Pregunta:
        {question}
        """
    )

    print("\nRespuesta:")
    print(response.output_text)
    print("\n---\n")