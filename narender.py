import os
import openai
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from docx import Document
import logging
import json
import requests
import base64

# Nastavení OpenAI API klíče
openai.api_key = os.getenv("OPENAI_API_KEY")

# Nastavení GitHub přístupového tokenu
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "Dahor212"
REPO_NAME = "chatbot"
FILE_PATH = "embeddings/embeddings.json"

# Inicializace FastAPI
app = FastAPI()

# Povolení CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Připojení k ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "dokumenty_kolekce"
collection = client.get_or_create_collection(collection_name)

# Nastavení logování
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("api_logger")

class QueryRequest(BaseModel):
    query: str

def generate_embedding(text):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Chyba při generování embeddingu: {e}")
        return None

def load_embeddings_from_github():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        content = response.json()
        file_content = base64.b64decode(content["content"]).decode()
        return json.loads(file_content)
    except requests.exceptions.RequestException as e:
        logger.error(f"Chyba při načítání embeddingů z GitHubu: {e}")
        return None

def save_embeddings_to_github(embeddings):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    new_content = json.dumps(embeddings)
    encoded_content = base64.b64encode(new_content.encode()).decode()
    data = {
        "message": "Aktualizace embeddingů",
        "content": encoded_content,
    }
    try:
        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
        logger.info("Embeddingy byly úspěšně uloženy na GitHub.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Chyba při ukládání embeddingů na GitHub: {e}")

def load_documents_into_chromadb():
    repo_path = "./word"
    documents = []
    for doc_filename in os.listdir(repo_path):
        if doc_filename.endswith(".docx"):
            doc_path = os.path.join(repo_path, doc_filename)
            doc = Document(doc_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            documents.append(text)
    embeddings = load_embeddings_from_github() or []
    if not embeddings:
        embeddings = [generate_embedding(doc) for doc in documents if generate_embedding(doc)]
        save_embeddings_to_github(embeddings)
    document_ids = [f"doc_{i}" for i in range(len(documents))]
    collection.add(ids=document_ids, documents=documents, embeddings=embeddings)

def query_chromadb(query, n_results=5):
    query_embedding = generate_embedding(query)
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Chyba při generování embeddingu pro dotaz.")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents"]
    )
    return results.get("documents", [[]])[0]

@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    documents = query_chromadb(request.query)
    if not documents:
        return {"answer": "Bohužel, odpověď ve své databázi nemám."}
    return {"answer": documents}

if __name__ == "__main__":
    load_documents_into_chromadb()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
