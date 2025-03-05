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

# Nastavení OpenAI API klíče z prostředí
openai.api_key = os.getenv("OPENAI_API_KEY")

# Nastavení GitHub přístupového tokenu
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "Dahor212"
REPO_NAME = "chatbot"
FILE_PATH = "embeddings/embeddings.json"

# Inicializace FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Připojení k ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("dokumenty_kolekce")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("api_logger")

class QueryRequest(BaseModel):
    query: str

def load_embeddings_from_github():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            content = response.json()
            return json.loads(base64.b64decode(content["content"]).decode())
    except Exception as e:
        logger.error(f"Chyba při načítání embeddingů z GitHubu: {e}")
    return {}

def save_embeddings_to_github(embeddings):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    response = requests.get(url, headers=headers)
    sha = ""
    if response.status_code == 200:
        sha = response.json().get("sha", "")
    
    data = {
        "message": "Aktualizace embeddingů",
        "content": base64.b64encode(json.dumps(embeddings).encode()).decode(),
        "sha": sha
    }
    
    try:
        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
        logger.info("Embeddingy byly úspěšně uloženy na GitHub.")
    except Exception as e:
        logger.error(f"Chyba při ukládání embeddingů na GitHub: {e}")

def load_documents_into_chromadb():
    repo_path = "./word"
    documents = []
    logger.info("Načítám dokumenty ze složky '%s'...", repo_path)
    
    for doc_filename in os.listdir(repo_path):
        if doc_filename.endswith(".docx"):
            doc_path = os.path.join(repo_path, doc_filename)
            logger.debug("Načítám dokument '%s'...", doc_path)
            doc = Document(doc_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            documents.append(text)
    
    if not documents:
        logger.warning("Žádné dokumenty nebyly nalezeny ve složce '%s'.", repo_path)
    
    embeddings = []
    for doc in documents:
        try:
            response = openai.Embedding.create(
                input=doc,
                model="text-embedding-ada-002"
            )
            embeddings.append(response["data"][0]["embedding"])
        except Exception as e:
            logger.error("Chyba při generování embeddingu: %s", str(e))
            embeddings.append([])
    
    save_embeddings_to_github(embeddings)
    
    document_ids = [f"doc_{i}" for i in range(len(documents))]
    collection.add(ids=document_ids, documents=documents, embeddings=embeddings)
    logger.info("Dokumenty byly uloženy do ChromaDB.")

def query_chromadb(query, n_results=5):
    try:
        response = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = response["data"][0]["embedding"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba při generování embeddingu: {str(e)}")
    
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
    return {"answer": "\n".join(documents)}

if __name__ == "__main__":
    load_documents_into_chromadb()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
