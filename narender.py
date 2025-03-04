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

# Nastavení OpenAI API klíče z prostředí
openai.api_key = os.getenv("OPENAI_API_KEY")

# Nastavení GitHub přístupového tokenu
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "your_github_username"
REPO_NAME = "your_repository_name"
FILE_PATH = "embeddings/embeddings.json"  # Cesta k souboru na GitHubu

# Inicializace FastAPI
app = FastAPI()

# Povolení CORS pro všechny domény (pro testování)
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

def load_embeddings_from_github():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    # Pokusíme se stáhnout embeddings soubor z GitHubu
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        content = response.json()
        
        # GitHub API vrací soubor jako base64, takže ho dekódujeme
        file_content = json.loads(content["content"].encode().decode("base64"))
        return file_content
    except requests.exceptions.RequestException as e:
        logger.error(f"Chyba při načítání embeddingů z GitHubu: {e}")
        return None

def save_embeddings_to_github(embeddings):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    # Načteme aktuální soubor, abychom získali SHA pro aktualizaci
    current_embeddings = load_embeddings_from_github() or {}
    
    # Aktualizovaný obsah
    new_content = json.dumps(embeddings)
    
    # GitHub API vyžaduje base64 pro uložení souboru
    encoded_content = new_content.encode("base64")
    
    data = {
        "message": "Aktualizace embeddingů",
        "content": encoded_content,
        "sha": current_embeddings.get("sha", "")
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
    
    embeddings = load_embeddings_from_github()  # Načteme embeddingy z GitHubu
    if embeddings is None:  # Pokud embeddingy neexistují, vygenerujeme nové
        embeddings = []
        for doc in documents:
            try:
                response = openai.embeddings.create(
                    input=doc,
                    model="text-embedding-ada-002"
                )
                embeddings.append(response.data[0].embedding)
                logger.debug("Embedding pro dokument úspěšně vygenerován.")
            except Exception as e:
                logger.error("Chyba při generování embeddingu pro dokument: %s", str(e))
                embeddings.append([])

        # Uložíme nově vygenerované embeddingy na GitHub
        save_embeddings_to_github(embeddings)
    
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

# Funkce pro dotazy na ChromaDB
def query_chromadb(query, n_results=5):
    logger.info("Začínám hledat dokumenty pro dotaz: '%s'...", query)
    
    try:
        response = openai.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = response.data[0].embedding
        logger.debug(f"Query embedding: {query_embedding[:10]}...")
    except Exception as e:
        logger.error("Chyba při generování embeddingu pro dotaz: %s", str(e))
        raise HTTPException(status_code=500, detail="Chyba při generování embeddingu pro dotaz.")
    
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents"]
        )
        documents = results.get("documents", [[]])[0]
        logger.info(f"Nalezeno {len(documents)} dokumentů.")
    except Exception as e:
        logger.error("Chyba při dotazu na ChromaDB: %s", str(e))
        raise HTTPException(status_code=500, detail="Chyba při vyhledávání v databázi.")
    
    return documents

@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    logger.info("Přijat dotaz: '%s'", request.query)
    
    try:
        documents = query_chromadb(request.query)
        if not documents:
            logger.warning("Žádné dokumenty nebyly nalezeny pro dotaz: '%s'.", request.query)
            return {"answer": "Bohužel, odpověď ve své databázi nemám."}
        
        answer = generate_answer_with_chatgpt(request.query, documents)
        return {"answer": answer}
    except HTTPException as e:
        logger.error("Chyba během zpracování dotazu: %s", str(e.detail))
        return {"answer": "Došlo k chybě při zpracování dotazu."}
    except Exception as e:
        logger.error("Neočekávaná chyba: %s", str(e))
        return {"answer": "Došlo k neočekávané chybě."}

if __name__ == "__main__":
    logger.info("Spouštím aplikaci a nahrávám dokumenty...")
    load_documents_into_chromadb()
    import uvicorn
    logger.info("Spouštím server na portu 10000...")
    uvicorn.run(app, host="0.0.0.0", port=10000)
