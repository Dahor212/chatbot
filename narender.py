import os
import openai
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from docx import Document
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
client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "dokumenty_kolekce"
collection = client.get_or_create_collection(collection_name)

# Nastavení logování
logging.basicConfig(level=logging.DEBUG)  # Nastavení logování na DEBUG
logger = logging.getLogger("narender")  # Vytvoření loggeru pro tento soubor

# Model pro příchozí dotazy
class QueryRequest(BaseModel):
    query: str

def generate_embedding(text):
    """ Funkce pro generování embeddingu """
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error("Chyba při generování embeddingu: %s", str(e))
        return None

def load_documents_into_chromadb():
    """ Funkce pro načtení dokumentů do ChromaDB """
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

    embeddings = [generate_embedding(doc) for doc in documents]
    document_ids = [f"doc_{i}" for i in range(len(documents))]

    try:
        collection.add(
            ids=document_ids,
            documents=documents,
            embeddings=embeddings
        )
        logger.info(f"{len(documents)} dokumentů bylo uloženo do ChromaDB.")
    except Exception as e:
        logger.error("Chyba při ukládání dokumentů do ChromaDB: %s", str(e))

def query_chromadb(query, n_results=5):
    """ Hledání relevantních dokumentů v ChromaDB """
    logger.info("Začínám hledat dokumenty pro dotaz: '%s'...", query)
    query_embedding = generate_embedding(query)
    
    if query_embedding is None:
        raise HTTPException(status_code=500, detail="Chyba při generování embeddingu pro dotaz.")

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents"]
        )
        documents = results.get("documents", [])
        logger.info(f"Nalezeno {len(documents)} dokumentů.")
    except Exception as e:
        logger.error("Chyba při dotazu na ChromaDB: %s", str(e))
        raise HTTPException(status_code=500, detail="Chyba při vyhledávání v databázi.")
    
    return documents

@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    """ Endpoint pro zpracování dotazů """
    logger.info("Přijat dotaz: '%s'", request.query)
    
    try:
        documents = query_chromadb(request.query)
        if not documents:
            logger.warning("Žádné dokumenty nebyly nalezeny pro dotaz: '%s'.", request.query)
            return {"answer": "Bohužel, odpověď ve své databázi nemám."}
        return {"answer": documents}
    
    except HTTPException as e:
        logger.error("Chyba během zpracování dotazu: %s", str(e.detail))
        return {"answer": "Došlo k chybě při zpracování dotazu."}
    except Exception as e:
        logger.error("Neočekávaná chyba: %s", str(e))
        return {"answer": "Došlo k neočekávané chybě."}
