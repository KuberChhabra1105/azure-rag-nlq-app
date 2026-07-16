from fastapi import FastAPI
from combined_pipeline import ask

app = FastAPI()


@app.get("/")
def home():
    return {"message": "RAG + NLQ API is running"}


@app.get("/ask")
def ask_question(question: str):
    answer = ask(question)
    return {"question": question, "answer": answer}