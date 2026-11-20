import chromadb
import requests
import json
import os

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:1b"
DISTANCE_THRESHOLD = 0.8   # Guardrail 1 — tune as needed


# ── Guardrail 3: Strengthened system prompt ─────────────
# TODO: Write a SYSTEM_PROMPT that:
#   - Instructs the model to answer ONLY from the provided context
#   - Explicitly says: never make up information
#   - Requires "I don't know" when the answer isn't in the context
#   - Requires source citation by document name
SYSTEM_PROMPT = """You are a strict document-based question-answering assistant.

Answer the user's question using ONLY information explicitly stated
in the provided CONTEXT.

GUARDRAIL RULES:
1. Never make up information, facts, examples, or code.
2. Never use outside knowledge.
3. If the context does not contain enough information, or you are
   unsure of the answer, respond exactly:
   I don't know based on the provided documents.
4. Always cite the source document for factual answers.
   Example: (fastapi.txt)
5. Do not cite documents that do not support your answer.
6. Never combine a factual answer with an uncertainty refusal.
7. Keep your response concise and under 100 words.

The context includes source filenames in [Source: filename] labels.
"""

def load_documents(directory: str) -> list[dict]:

    chunks = []

    for filename in sorted(os.listdir(directory)):

        if not filename.endswith((".txt", ".md")):
            continue

        filepath = os.path.join(directory, filename)

        if not os.path.isfile(filepath):
            continue

        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()

        paragraphs = content.split("\n\n")

        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()

            if not paragraph:
                continue
        
            chunks.append({
                "text": paragraph,
                "id": f"{filename}_{i}",
                "metadata": {
                    "source": filename, 
                    "chunk_index": str(i)
                }
            })

    return chunks

#Chunk and store all documents in ChromaDB. Returns chunk count.
def ingest(collection, docs_directory: str) -> int:

    chunks = load_documents(docs_directory)
    
    if not chunks:
        print("Warning:No document chunks found.")
        return 0

    collection.upsert(
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
        ids=[chunk["id"] for chunk in chunks]
    )

    print(f"Successfully ingested {len(chunks)} chunks.")

    return len(chunks)


# ── Step 2: Retrieval ──────────────────
#Query ChromaDB, return the most relevant chunks..
def retrieve(collection, query: str, n_results: int = 3) -> list[dict]:

    if collection.count() == 0:
        print("Warning: The ChromaDB collection is empty.")
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count())
    )

    chunks = []

    for text, metadata, distance in zip(
        results["documents"][0], 
        results["metadatas"][0], 
        results["distances"][0]
    ):
        chunks.append({
            "text": text,
            "metadata": metadata,
            "distance": distance
        })
    return chunks

#Generate a complete answer using Ollama.
def generate(messages: list[dict]) -> str:

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1}
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",json=payload,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return data["message"]["content"].strip()

    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError) as error:
        return f"ERROR: Ollama request failed: {error}"

# ── Guardrail 1 + 2: Filter and score ─────────────────────────────────────

#Keep only chunks with a distance below the threshold.
def filter_chunks(chunks: list[dict], threshold: float = DISTANCE_THRESHOLD) -> list[dict]:

    return [chunk for chunk in chunks if chunk["distance"] < threshold]

#Determine confidence using the lowest chunk distance.
def compute_confidence(chunks: list[dict]) -> str:

    if not chunks:
        return "low"

    best_distance  = min(chunk["distance"] for chunk in chunks)

    if best_distance < 0.5:
        return "high"
    elif best_distance < 1.0:
        return "medium"
    else:
        return "low"


# ── Guardrail 4: Structured RAG query ─────────────────────────────────────

#Run the guardrailed RAG pipeline and return a structured response.
def rag_query(collection, question: str) -> dict:

    #Step 1: Retrieve candidate chunks from ChromaDB.
    raw_chunks = retrieve(collection, question)

    #Step 2: Apply the distance threshold.
    chunks = filter_chunks(raw_chunks)

    #Step 3: Return early if no chunks pass the filter.
    if not chunks:
        return {
            "answer": "I couldn't find relevant information in the documents.",
            "sources": [],
            "confidence": "low",
            "chunks_retrieved": 0
        }

    #Step 4: Compute confidence using the best distance.
    confidence = compute_confidence(chunks)

    #Step 5: Build context with source filenames.
    context_parts = []

    for chunk in chunks:
        source = chunk["metadata"]["source"]
        text = chunk["text"]
        context_parts.append(f"[Source: {source}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    #Step 6: Construct the messages for Ollama.
    messages = [
        {
            "role": "system", 
            "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"
        },
        {
            "role": "user", 
            "content": question 
        }  
    ]

# ── Main: test with 4 query types ─────────────────────────────────────────
    #Step 7: Generate the answer.
    answer = generate(messages)

    #Step 8: Collect unique source filenames.

    sources = sorted({
        chunk["metadata"]["source"] 
        for chunk in chunks
    })

    #Step 9:Ensure factual answers include source attribution.
    refusal = "I don't know based on the provided documents."

    if (
        answer
        and not answer.startswith("ERROR:")
        and answer != refusal
        and not any(source in answer for source in sources)
    ):
        answer += f"\n\nRetrieved sources: {', '.join(sources)}"

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "chunks_retrieved": len(chunks)}

if __name__ == "__main__":
    client = chromadb.PersistentClient(path="./rag_db")
    collection = client.get_or_create_collection(
        "course_docs", configuration={"hnsw": {"space": "cosine"}}
    )

    if collection.count() == 0:
        count = ingest(collection, "docs")
        print(f"Ingested {count} chunks.")
    else:
        print(f"Using existing collection: {collection.count()} chunks.")

    #4 test questions (in-scope, partial, out-of-scope, ambiguous)
    test_questions = [
        (
            "in-scope",
            "How does FastAPI validate incoming request data?"
        ),
        (
            "partial",
            "Who originally created ChromaDB, and in what year?"
        ),
        (
            "out-of-scope",
            "What year did the Titanic sink?"
        ),
        (
            "ambiguous",
            "How does caching work?"
        )
    ]

    for label, question in test_questions:
        print(f"\n{'='*60}")
        print(f"[{label.upper()}] {question}")
        result = rag_query(collection, question)
        print(f"Confidence   : {result['confidence']}")
        print(f"Chunks used  : {result['chunks_retrieved']}")
        print(f"Sources      : {result['sources']}")
        print(f"Answer       :\n{result['answer']}")
