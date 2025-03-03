import os
import openai
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from docx import Document
import tiktoken
import numpy as np

# Nastavení OpenAI API klíče z prostředí
openai.api_key = os.getenv("OPENAI_API_KEY")  # Získejte klíč z prostředí
if not openai.api_key:
    print("Chyba: API klíč není nastaven.")
else:
    print("API klíč je nastaven.")  # Informace o nastavení klíče

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
print(f"ChromaDB kolekce '{collection_name}' byla vytvořena nebo nalezena.")

# Model pro příchozí dotazy
class QueryRequest(BaseModel):
    query: str


def load_documents_into_chromadb():
    """ Funkce pro načtení dokumentů do ChromaDB """
    print("Začínám načítat dokumenty do ChromaDB...")
    repo_path = "./word"  # Složka, která bude obsahovat dokumenty
    documents = []

    # Pro každý dokument načteme text
    for doc_filename in os.listdir(repo_path):
        if doc_filename.endswith(".docx"):
            doc_path = os.path.join(repo_path, doc_filename)
            doc = Document(doc_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            documents.append(text)

    print(f"Nalezeno {len(documents)} dokumentů.")

    # Vygeneruj embeddingy pro všechny dokumenty
    embeddings = []
    for doc in documents:
        try:
            response = openai.embeddings.create(  # Volání API pro embeddingy
                input=doc,
                model="text-embedding-ada-002"
            )
            embeddings.append(response["data"][0]["embedding"])
            print(f"Embedding pro dokument {documents.index(doc)+1} vygenerován.")
        except Exception as e:
            print(f"Chyba při generování embeddingu pro dokument: {e}")

    # Vytvoření ID pro každý dokument
    document_ids = [f"doc_{i}" for i in range(len(documents))]

    # Ulož dokumenty, jejich ID a embeddingy do ChromaDB
    collection.add(
        ids=document_ids,  # Zde se přidávají ID dokumentů
        documents=documents,
        embeddings=embeddings
    )
    print(f"{len(documents)} dokumenty byly uloženy do ChromaDB.")


def query_chromadb(query, n_results=5):
    """ Hledání relevantních dokumentů v ChromaDB """
    print(f"Začínám hledat dokumenty pro dotaz: '{query}'...")

    # Generování embeddingu pro dotaz
    try:
        response = openai.embeddings.create(  # Volání API pro embedding dotazu
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = response["data"][0]["embedding"]
        print(f"Query embedding: {query_embedding[:10]}...")  # Debug: zobraz první část embeddingu
    except Exception as e:
        print(f"Chyba při generování embeddingu pro dotaz: {e}")
        return []

    # Vyhledání dokumentů v ChromaDB na základě embeddingu
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents"]
    )

    documents = results.get("documents", [])
    print(f"Found documents: {documents}")  # Debug: Zobrazení nalezených dokumentů
    return documents


def generate_answer_with_chatgpt(query, context_documents):
    """ Generování odpovědi pomocí ChatGPT """
    print(f"Generuji odpověď na základě {len(context_documents)} dokumentů.")
    if not context_documents or not isinstance(context_documents, list):
        return "Bohužel, odpověď ve své databázi nemám."

    # Zajištění správného formátu context_documents
    context_documents = [str(doc) if isinstance(doc, (list, dict)) else doc for doc in context_documents]

    context = "\n\n".join(context_documents) if context_documents else "Žádná data."

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
            messages=[{"role": "system", "content": "Jsi asistent v oblasti penzijního spoření."},
                      {"role": "user", "content": prompt}],
            max_tokens=300,  # Zvýšení max_tokens pro delší odpovědi
            temperature=0.7,
            stop=["\n\n"]  # Přidání stop sekvence pro ukončení odpovědi
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Chyba při generování odpovědi: {e}")
        return "Bohužel, odpověď ve své databázi nemám."


@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    """ Endpoint pro zpracování dotazů """
    query = request.query
    print(f"Dotaz přijat: {query}")
    documents = query_chromadb(query)

    if not documents:
        return {"answer": "Bohužel, odpověď ve své databázi nemám."}

    answer = generate_answer_with_chatgpt(query, documents)
    return {"answer": answer}


if __name__ == "__main__":
    # Zavolej tuto funkci, když spustíš kód pro nahrání dokumentů
    load_documents_into_chromadb()

    import uvicorn

    print("Spouštím server na portu 10000...")
    uvicorn.run(app, host="0.0.0.0", port=10000)
