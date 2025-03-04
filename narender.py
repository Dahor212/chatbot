import os
import openai
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Nastavení OpenAI API klíče
openai.api_key = os.getenv("OPENAI_API_KEY")

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
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="documents")

# Nastavení logování
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("narender")

class QueryRequest(BaseModel):
    query: str

def generate_embedding(text: str):
    """ Vytvoří embedding pro daný text pomocí OpenAI API. """
    try:
        response = openai.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Chyba při generování embeddingu: {e}")
        return None

@app.post("/query")
def query_chromadb(request: QueryRequest):
    logger.info(f"Přijatý dotaz: {request.query}")
    
    query_embedding = generate_embedding(request.query)
    if query_embedding is None:
        raise HTTPException(status_code=500, detail="Chyba při generování embeddingu.")
    
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    
    documents = results.get("documents", [[]])[0]
    if documents:
        odpoved = documents[0]
        logger.info(f"Vrácená odpověď: {odpoved}")
    else:
        odpoved = "Bohužel nemám odpověď na tuto otázku."
        logger.warning("Žádné relevantní dokumenty nebyly nalezeny.")
    
    return {"answer": odpoved}
