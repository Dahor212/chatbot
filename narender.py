import os
import openai
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Nastavení OpenAI API klíče z prostředí
openai.api_key = os.getenv("OPENAI_API_KEY")  # Získejte klíč z prostředí

# Inicializace FastAPI
app = FastAPI()

# Povolení CORS pro všechny domény (pro testování)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Povolit přístup odkudkoli, změňte to na konkrétní domény pro produkci
    allow_credentials=True,
    allow_methods=["*"],  # Povolit všechny metody (GET, POST, OPTIONS, atd.)
    allow_headers=["*"],  # Povolit všechny hlavičky
)

# Připojení k ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

# Nastavení logování
logging.basicConfig(level=logging.DEBUG)  # Nastavení logování na DEBUG
logger = logging.getLogger("narender")  # Vytvoření loggeru pro tento soubor

# Model pro příchozí dotazy
class QueryRequest(BaseModel):
    query: str

def generate_embedding(text):
    """ Funkce pro generování embeddingu """
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        logger.error("Chyba při generování embeddingu: %s", str(e))
        return None

@app.post("/query")
def query_chromadb(request: QueryRequest):
    logger.info(f"Přijatý dotaz: {request.query}")
    
    # Vytvoření embeddingu pro dotaz
    query_embedding = generate_embedding(request.query)
    if query_embedding is None:
        raise HTTPException(status_code=500, detail="Chyba při generování embeddingu pro dotaz.")
    
    # Dotaz na ChromaDB
    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=5, include=["documents"])
        documents = results.get("documents", [[]])[0]  # Vrátí seznam dokumentů
        logger.info(f"Výsledky dotazu z ChromaDB: {results}")
    except Exception as e:
        logger.error("Chyba při dotazu na ChromaDB: %s", str(e))
        raise HTTPException(status_code=500, detail="Chyba při vyhledávání v databázi.")
    
    # Zpracování odpovědi
    if documents:
        odpoved = documents[0]  # Vezmeme první nalezený dokument
        logger.info(f"Vrácená odpověď: {odpoved}")
    else:
        odpoved = "Bohužel nemám odpověď na tuto otázku."
        logger.warning("Žádné relevantní dokumenty nebyly nalezeny.")
    
    return {"answer": odpoved}
