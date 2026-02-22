from fastapi import FastAPI
from pydantic import BaseModel
from main import ask_question, load_document

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