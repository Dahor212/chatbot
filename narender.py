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
logger = logging.getLogger(__name__)  # Vytvoření loggeru pro tento soubor

# Model pro příchozí dotazy
class QueryRequest(BaseModel):
    query: str


def load_documents_into_chromadb():
    """ Funkce pro načtení dokumentů do ChromaDB """
    # Předpokládejme, že dokumenty jsou v repozitáři ve složce "word"
    repo_path = "./word"  # Složka, která bude obsahovat dokumenty
    documents = []
    logger.info("Načítám dokumenty ze složky '%s'...", repo_path)

    # Pro každý dokument načteme text
    for doc_filename in os.listdir(repo_path):
        if doc_filename.endswith(".docx"):
            doc_path = os.path.join(repo_path, doc_filename)
            logger.debug("Načítám dokument '%s'...", doc_path)
            doc = Document(doc_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            documents.append(text)

    if not documents:
        logger.warning("Žádné dokumenty nebyly nalezeny ve složce '%s'.", repo_path)

    # Vygeneruj embeddingy pro všechny dokumenty
    embeddings = []
    for doc in documents:
        try:
            response = openai.embeddings.create(  # Upravený způsob volání API
                input=doc,
                model="text-embedding-ada-002"
            )
            embeddings.append(response['data'][0]['embedding'])  # Oprava přístupu k datům
            logger.debug("Embedding pro dokument úspěšně vygenerován.")
        except Exception as e:
            logger.error("Chyba při generování embeddingu pro dokument: %s", str(e))
            embeddings.append([])  # Pokud dojde k chybě, přidáme prázdný embedding

    # Vytvoření ID pro každý dokument
    document_ids = [f"doc_{i}" for i in range(len(documents))]

    # Ulož dokumenty, jejich ID a embeddingy do ChromaDB
    try:
        collection.add(
            ids=document_ids,  # Zde se přidávají ID dokumentů
            documents=documents,
            embeddings=embeddings
        )
        logger.info(f"{len(documents)} dokumenty byly uloženy do ChromaDB.")
    except Exception as e:
        logger.error("Chyba při ukládání dokumentů do ChromaDB: %s", str(e))


def query_chromadb(query, n_results=5):
    """ Hledání relevantních dokumentů v ChromaDB """
    logger.info("Začínám hledat dokumenty pro dotaz: '%s'...", query)

    # Generování embeddingu pro dotaz
    try:
        response = openai.embeddings.create(  # Upravený způsob volání API
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = response['data'][0]['embedding']  # Oprava přístupu k datům
        logger.debug(f"Query embedding: {query_embedding[:10]}...")  # Debug: zobraz první část embeddingu
    except Exception as e:
        logger.error("Chyba při generování embeddingu pro dotaz: %s", str(e))
        raise HTTPException(status_code=500, detail="Chyba při generování embeddingu pro dotaz.")

    # Vyhledání dokumentů v ChromaDB na základě embeddingu
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


def generate_answer_with_chatgpt(query, context_documents):
    """ Generování odpovědi pomocí ChatGPT """
    if not context_documents or not isinstance(context_documents, list):
        logger.warning("Žádné relevantní dokumenty nenalezeny pro dotaz: '%s'.", query)
        return "Bohužel, odpověď ve své databázi nemám."

    # Zajištění správného formátu context_documents
    context_documents = [str(doc) if isinstance(doc, (list, dict)) else doc for doc in context_documents]

    context = "\n\n".join(context_documents) if context_documents else "Žádná data."

    prompt = f"""
    Jsi asistent pro helpdesk v oblasti penzijního spoření. Odpovědaj pouze na základě následujících dokumentů.
    Pokud odpověď neznáš, odpověz: 'Bohužel, odpověď ve své databázi nemám.'

    Kontext dokumentů:
    {context}

    Otázka: {query}
    Odpověď:
    """

    try:
        # Použití nového API pro generování odpovědi
        response = openai.Completion.create(
            model="gpt-3.5-turbo",  # Použití modelu ChatGPT 3.5 Turbo
            prompt=prompt,
            max_tokens=300,
            temperature=0.7
        )
        logger.info("Generování odpovědi bylo úspěšné.")
        return response['choices'][0]['text'].strip()
    except Exception as e:
        logger.error("Chyba při generování odpovědi s ChatGPT: %s", str(e))
        return "Chyba při generování odpovědi."


@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    """ Endpoint pro zpracování dotazů """
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
    # Zavolej tuto funkci, když spustíš kód pro nahrání dokumentů
    logger.info("Spouštím aplikaci a nahrávám dokumenty...")
    load_documents_into_chromadb()

    import uvicorn
    logger.info("Spouštím server na portu 10000...")
    uvicorn.run(app, host="0.0.0.0", port=10000)
