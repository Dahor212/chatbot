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
    """
    Funkce pro vygenerování embeddingu pomocí OpenAI API (nová verze)
    """
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Chyba při generování embeddingu: {e}")
        return None

def load_documents_into_chromadb():
    repo_path = "./word"
    documents = []
    logger.info("Načítám dokumenty ze složky '%s'...", repo_path)
    
    for doc_filename in os.listdir(repo_path):
        if doc_filename.endswith(".docx"):
            doc_path = os.path.join(repo_path, doc_filename)
            doc = Document(doc_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            documents.append(text)
    
    embeddings = []
    for doc in documents:
        embedding = generate_embedding(doc)
        if embedding:
            embeddings.append(embedding)
        else:
            embeddings.append([])
    
    document_ids = [f"doc_{i}" for i in range(len(documents))]
    
    try:
        collection.add(
            ids=document_ids,
            documents=documents,
            embeddings=embeddings
        )
        logger.info(f"{len(documents)} dokumenty byly uloženy do ChromaDB.")
    except Exception as e:
        logger.error("Chyba při ukládání dokumentů do ChromaDB: %s", str(e))

def query_chromadb(query, n_results=5):
    logger.info("Začínám hledat dokumenty pro dotaz: '%s'...", query)
    query_embedding = generate_embedding(query)
    
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Chyba při generování embeddingu pro dotaz.")
    
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents"]
        )
        return results.get("documents", [[]])[0]
    except Exception as e:
        logger.error("Chyba při dotazu na ChromaDB: %s", str(e))
        raise HTTPException(status_code=500, detail="Chyba při vyhledávání v databázi.")

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
