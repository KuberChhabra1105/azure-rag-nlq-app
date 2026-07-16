from RAG.combined_pipeline import *

print("Downloading PDFs...")
download_blobs()

print("Reading PDFs...")
docs = extract_text()
print(len(docs))
print(docs[0]["book"])
print(docs[0]["text"][:500])

print("Creating embeddings...")
store_chunks(docs)

print("RAG Ready")

while True:
    question = input("\nAsk Question : ")

    if question.lower() == "exit":
        break

    answer = ask_llm(question)

    print("\nAnswer:\n")
    print(answer)