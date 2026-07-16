import os
import json
import psycopg2
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("RAG_DB_USER")
DB_PASSWORD = os.getenv("RAG_DB_PASSWORD")
RAG_DB_NAME = os.getenv("RAG_DB_NAME")
NLQ_DB_NAME = "nlq_db"

client = AzureOpenAI(api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_OPENAI_API_VERSION, azure_endpoint=AZURE_OPENAI_ENDPOINT)

rag_conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=RAG_DB_NAME, user=DB_USER, password=DB_PASSWORD, sslmode="require", connect_timeout=30)
rag_cur = rag_conn.cursor()

nlq_conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=NLQ_DB_NAME, user=DB_USER, password=DB_PASSWORD, sslmode="require", connect_timeout=30)
nlq_cur = nlq_conn.cursor()


def reconnect_if_needed(conn, cur, dbname):
    # check if connection alive, else reconnect
    try:
        cur.execute("SELECT 1")
        return conn, cur
    except:
        print("Reconnecting to", dbname)
        conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=dbname, user=DB_USER, password=DB_PASSWORD, sslmode="require", connect_timeout=30)
        cur = conn.cursor()
        return conn, cur


def get_embedding(text):
    # get embedding vector from azure openai
    response = client.embeddings.create(model=AZURE_EMBEDDING_DEPLOYMENT, input=text)
    return response.data[0].embedding


def search_textbook(query):
    # search textbook chunks using vector similarity
    global rag_conn, rag_cur
    rag_conn, rag_cur = reconnect_if_needed(rag_conn, rag_cur, RAG_DB_NAME)

    embedding = get_embedding(query)
    vector = "[" + ",".join(map(str, embedding)) + "]"

    rag_cur.execute("""
        SELECT book_name, raw_text_content
        FROM ncert_book_chunks
        ORDER BY vector_embedding <=> %s::vector
        LIMIT 5
    """, (vector,))

    rows = rag_cur.fetchall()

    result = ""
    for row in rows:
        result += f"\nBook : {row[0]}\n{row[1]}\n"

    return result


def get_schema_text():
    # fetch live schema of nlq database
    global nlq_conn, nlq_cur
    nlq_conn, nlq_cur = reconnect_if_needed(nlq_conn, nlq_cur, NLQ_DB_NAME)

    nlq_cur.execute("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
    """)

    rows = nlq_cur.fetchall()

    schema_text = ""
    current_table = ""

    for row in rows:
        if row[0] != current_table:
            schema_text += f"\nTable: {row[0]}\nColumns: "
            current_table = row[0]
        else:
            schema_text += ", "
        schema_text += row[1]

    return schema_text


def query_database(query):
    # ask gpt to write sql, then run it safely
    global nlq_conn, nlq_cur
    schema_text = get_schema_text()

    sql_response = client.chat.completions.create(
        model=AZURE_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": f"Write a PostgreSQL SQL query for this question. Only use these tables and columns. Return ONLY the SQL, no explanation.\n{schema_text}"},
            {"role": "user", "content": query}
        ],
        max_completion_tokens=2000
    )

    sql = sql_response.choices[0].message.content.strip()

    nlq_conn, nlq_cur = reconnect_if_needed(nlq_conn, nlq_cur, NLQ_DB_NAME)

    try:
        nlq_cur.execute(sql)
        rows = nlq_cur.fetchall()
        columns = [desc[0] for desc in nlq_cur.description]
        result = ""
        for row in rows:
            result += str(dict(zip(columns, row))) + "\n"
        return result
    except Exception as e:
        nlq_conn.rollback()
        return f"Database error: {e}"


# tools gpt can choose to call on its own, no forced keywords
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_textbook",
            "description": "Performs a semantic search over a collection of documents to find relevant passages that can help answer conceptual or explanatory questions.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "the search text"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Runs a query against a structured business database to retrieve exact facts, numbers, or records.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "the question needing a database lookup"}},
                "required": ["query"]
            }
        }
    }
]


def ask(question):
    # let gpt decide on its own which tool or tools to call
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Decide for yourself, based on the question, whether you need to call any tools, one tool, or multiple tools, to gather information before answering. Only call a tool if it is actually needed."},
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model=AZURE_CHAT_DEPLOYMENT,
        messages=messages,
        tools=tools,
        max_completion_tokens=2000
    )

    reply = response.choices[0].message

    # if gpt did not call any tool, just return its direct answer
    if not reply.tool_calls:
        return reply.content

    messages.append(reply)

    # run whichever tools gpt decided to call
    for call in reply.tool_calls:

        args = json.loads(call.function.arguments)

        if call.function.name == "search_textbook":
            result = search_textbook(args["query"])
        elif call.function.name == "query_database":
            result = query_database(args["query"])
        else:
            result = "Unknown tool"

        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # ask gpt again, now with tool results, to write final answer
    final_response = client.chat.completions.create(
        model=AZURE_CHAT_DEPLOYMENT,
        messages=messages,
        max_completion_tokens=2000
    )

    return final_response.choices[0].message.content


if __name__ == "__main__":
    while True:
        question = input("\nAsk Question : ")
        if question.lower() == "exit":
            break
        answer = ask(question)
        print("\nAnswer:\n")
        print(answer)