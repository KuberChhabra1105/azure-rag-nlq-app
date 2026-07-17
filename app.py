from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from combined_pipeline import ask

app = FastAPI()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG + NLQ Assistant</title>
    <style>
        body { font-family: Arial; max-width: 700px; margin: 50px auto; padding: 20px; }
        input { width: 80%; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; }
        #answer { margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 8px; white-space: pre-wrap; }
        #loading { display: none; margin-top: 10px; color: gray; }
    </style>
</head>
<body>
    <h2>Ask about Physics, Science, or Employee Data</h2>
    <input type="text" id="question" placeholder="Type your question here">
    <button onclick="askQuestion()">Ask</button>
    <div id="loading">Thinking...</div>
    <div id="answer"></div>

    <script>
        async function askQuestion() {
            const question = document.getElementById("question").value;
            if (!question) return;

            document.getElementById("loading").style.display = "block";
            document.getElementById("answer").innerText = "";

            const response = await fetch("/ask?question=" + encodeURIComponent(question));
            const data = await response.json();

            document.getElementById("loading").style.display = "none";
            document.getElementById("answer").innerText = data.answer;
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.get("/ask")
def ask_question(question: str):
    answer = ask(question)
    return {"question": question, "answer": answer}