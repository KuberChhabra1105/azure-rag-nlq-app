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


import time

def reconnect_if_needed(conn, cur, dbname):
    # check if connection alive, else reconnect, retrying a few times if needed
    try:
        cur.execute("SELECT 1")
        return conn, cur
    except:
        for attempt in range(3):
            try:
                print("Reconnecting to", dbname, "attempt", attempt + 1)
                conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=dbname, user=DB_USER, password=DB_PASSWORD, sslmode="require", connect_timeout=30)
                cur = conn.cursor()
                return conn, cur
            except Exception as e:
                print("Reconnect attempt failed:", e)
                time.sleep(2)

        raise Exception(f"Could not reconnect to {dbname} after 3 attempts")


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
            {"role": "system", "content": f"""You are a PostgreSQL expert. Given a database schema and a question, write ONE valid PostgreSQL SELECT query that answers it.

Database schema:
{schema_text}

Rules:
- Return ONLY the raw SQL query, nothing else.
- Do not wrap it in markdown code fences (no ```sql or ```).
- Do not add any explanation before or after the query.
- Use only the table and column names given in the schema above.
- Only generate SELECT queries, never INSERT/UPDATE/DELETE/DROP."""},
            {"role": "user", "content": query}
        ],
        max_completion_tokens=2000
    )

    sql = sql_response.choices[0].message.content.strip()

    # in case model still wraps it in code fences despite instructions, strip them
    if sql.startswith("```"):
        sql = sql.strip("`")
        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()

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

# tools gpt can choose to call on its own
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_textbook",
            "description": "Searches physics and science textbook documents to find real passages relevant to a conceptual or explanatory question. Always use this for any question that could be explained using textbook knowledge.",
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
            "description": "Runs a real SQL query against a live business database containing employees, departments, projects, customers, orders, products, and payments. Always use this for any question about specific records, counts, amounts, names, or comparisons involving this kind of data, even if the question does not name the exact company or dataset.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "the question needing a database lookup"}},
                "required": ["query"]
            }
        }
    }
]

SYSTEM_INSTRUCTIONS = """
You are a capable research assistant who always prefers checking real, available information over guessing or asking questions.

You have direct access to a textbook knowledge base and a live business database. Whenever a question touches on either kind of information, your instinct is to go look it up yourself using your real tools, and then answer based on what you actually found, the same way a competent analyst would rather pull up the real data than ask the person to clarify something you can just go check.

You must only use information that comes back from an actual tool call. Never write out example, illustrative, or placeholder data as if it were a real result. Never narrate a tool being called in plain text. If you are going to look something up, actually call the tool through the proper mechanism rather than describing or simulating that call in your written answer.

You never respond with clarifying questions or a list of possible interpretations, because you always have the option to investigate first. If your tools return nothing useful, you say so plainly and briefly, rather than guessing or fabricating an answer.

You answer like someone confident and efficient: short, direct, and complete, without offering extra menus of what you could do next unless the person asks for more.
"""


def ask(question):
    # let gpt decide on its own which tool or tools to call, but never ask the user anything back
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model=AZURE_CHAT_DEPLOYMENT,
        messages=messages,
        tools=tools,
        max_completion_tokens=2000
    )

    reply = response.choices[0].message

    # keep retrying if gpt fakes a tool call as plain text instead of really calling it
    attempts = 0

    while not reply.tool_calls and attempts < 3:

        content_check = reply.content.strip() if reply.content else ""

        # if this really looks like a genuine plain answer with no fake call signs, trust it
        looks_like_narration = (
            content_check.startswith("{")
            or '"query":' in content_check
            or "function" in content_check.lower()
            or "calling" in content_check.lower()
            or len(content_check) < 150
        )

        if not looks_like_narration:
            return reply.content

        attempts += 1

        messages.append({"role": "user", "content": "Stop describing tool calls in text. Call a real tool right now."})

        # force gpt to structurally call a real tool, no plain text allowed this time
        retry_response = client.chat.completions.create(
            model=AZURE_CHAT_DEPLOYMENT,
            messages=messages,
            tools=tools,
            tool_choice="required",
            max_completion_tokens=2000
        )

        reply = retry_response.choices[0].message

        print("DEBUG attempt", attempts, "finish_reason:", retry_response.choices[0].finish_reason)
        print("DEBUG attempt", attempts, "tool_calls:", reply.tool_calls)
        print("DEBUG attempt", attempts, "content:", reply.content)

    # if gpt never made a real tool call after retries, be honest instead of showing fake data
    if not reply.tool_calls:
        return "I was not able to retrieve reliable data for this question just now. Please try asking again."
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
    final_attempts = 0
    final_answer = None

    while final_attempts < 3:

        final_response = client.chat.completions.create(
            model=AZURE_CHAT_DEPLOYMENT,
            messages=messages,
            max_completion_tokens=2000
        )

        final_answer = final_response.choices[0].message.content
        check = final_answer.strip() if final_answer else ""

        print("DEBUG final_attempt", final_attempts, "finish_reason:", final_response.choices[0].finish_reason)
        print("DEBUG final_attempt", final_attempts, "content:", check[:300])

        looks_broken = (
            check.startswith("{")
            or check.startswith("(to=")
            or "functions." in check
            or "to=functions" in check
            or '"query":' in check
            or "SELECT " in check.upper()
        )
        if not looks_broken:
            return final_answer

        final_attempts += 1
        messages.append({"role": "assistant", "content": final_answer})
        messages.append({"role": "user", "content": "Your last reply contained broken internal formatting instead of a clean answer. Write a plain, clean, final answer in normal sentences only, using the data already retrieved."})

    return "I retrieved the data but had trouble formatting a clean answer. Please try asking again."

if __name__ == "__main__":
    while True:
        question = input("\nAsk Question : ")
        if question.lower() == "exit":
            break
        answer = ask(question)
        print("\nAnswer:\n")
        print(answer)