from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from main import ask_question, load_document, process_pdf, DOCUMENTS_FOLDER
import shutil

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():
    load_document()

@app.post("/ask")
def ask(request: QuestionRequest):
    answer = ask_question(request.question)
    return {"answer": answer}


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    file_path = f"{DOCUMENTS_FOLDER}/{file.filename}"

    # Guardar archivo
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Procesarlo
    process_pdf(file_path)

    return {"message": f"{file.filename} cargado y procesado"}