from fastapi import FastAPI
from rag import answer_query

app = FastAPI()

@app.get("/")
def home():
    return {"message": "AI Teaching Assistant API"}

@app.post("/ask")
def ask(question: str):
    response = answer_query(question)
    return {"answer": response}