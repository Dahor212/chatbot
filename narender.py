import os
import openai
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from docx import Document
import logging

# Nastavení OpenAI API klíče z prostředí
openai.api_key = os.getenv("OPENAI_API_KEY")

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
            logger.debug("Embedding pro dokument úspěšně vygenerován.")
        except Exception as e:
            logger.error("Chyba při generování embeddingu pro dokument: %s", str(e))
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
    
    try:
        response = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = response["data"][0]["embedding"]
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

def generate_answer_with_chatgpt(query, context_documents):
    if not context_documents:
        logger.warning("Žádné relevantní dokumenty nenalezeny pro dotaz: '%s'.", query)
        return "Bohužel, odpověď ve své databázi nemám."
    
    context = "\n\n".join(context_documents)
    prompt = f"""
    Jsi asistent pro helpdesk v oblasti penzijního spoření. Odpovídej pouze na základě následujících dokumentů.
    Pokud odpověď neznáš, odpověz: 'Bohužel, odpověď ve své databázi nemám.'
    
    Kontext dokumentů:
    {context}
    
    Otázka: {query}
    Odpověď:
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Jsi AI asistent."},
                      {"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        answer = response["choices"][0]["message"]["content"].strip()
        logger.info("Generování odpovědi bylo úspěšné: '%s'", answer[:100])
        return answer
    except Exception as e:
        logger.error("Chyba při generování odpovědi s ChatGPT: %s", str(e))
        return "Chyba při generování odpovědi."

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
