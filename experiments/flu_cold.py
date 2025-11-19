# nhs_index_with_metadata.py
import os
import re
import requests
from bs4 import BeautifulSoup

from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from ollama import chat

# --- CONFIG ---
INDEX_DIR = "./nhs_index_metadata"
NHS_URLS = [
    "https://www.nhs.uk/conditions/common-cold/",
    "https://www.nhs.uk/conditions/flu/"
]
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"

# --- HELPER FUNCTIONS ---

def fetch_html(url: str) -> str:
    """Fetch the HTML of a webpage."""
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def extract_title(html: str) -> str:
    """Extract the real page title from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    # fallback: first <h1>
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else "unknown"

def extract_metadata(doc: Document, html: str = None):
    """
    Extract structured metadata from a document for better retrieval.
    """
    text = doc.text.lower()
    
    # Get title
    title = "unknown"
    if html:
        title = extract_title(html)
    elif "title" in doc.metadata:
        title = doc.metadata.get("title", "unknown")
    
    metadata = {
        "title": title,
        "source": doc.metadata.get("url", "unknown"),
        "symptoms": list(set(re.findall(
            r"\b(fever|cough|sore throat|fatigue|sneezing|headache|runny nose|blocked nose|hoarse voice)\b",
            text
        ))),
        "severity": "mild" if "usually mild" in text else "moderate" if "can be severe" in text else "unknown",
        "duration": "short" if "few days" in text else "long" if "weeks" in text else "unspecified"
    }
    return metadata

# --- INDEX BUILDING ---

def build_or_load_index():
    """
    Load an existing NHS index, or fetch pages, parse, attach metadata, and create a new one.
    """
    if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        print("📂 Loading existing NHS index...")
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage_context, embed_model=OllamaEmbedding(model_name=EMBED_MODEL))
    else:
        print("🌐 Fetching NHS pages...")
        documents = []
        for url in NHS_URLS:
            html = fetch_html(url)
            reader_doc = Document(text=BeautifulSoup(html, "html.parser").get_text("\n"), metadata={"url": url})
            # attach metadata
            reader_doc.metadata.update(extract_metadata(reader_doc, html))
            documents.append(reader_doc)
        
        print("🧩 Parsing documents into nodes...")
        parser = SimpleNodeParser()
        nodes = parser.get_nodes_from_documents(documents)

        print("⚙️ Creating and saving vector index with metadata...")
        embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        index.storage_context.persist(persist_dir=INDEX_DIR)

    return index

# --- QUERY FUNCTION ---

def query_nhs(question: str) -> str:
    index = build_or_load_index()
    
    # Use Ollama's chat function
    response = chat(model=LLM_MODEL, messages=[{"role": "user", "content": question}])
    
    print("\n🧠 NHS Summary:")
    print(response['message']['content'])
    return str(response['message']['content'])

# --- MAIN ---

if __name__ == "__main__":
    print("✅ NHS Reader Ready.")
    query = input("\nAsk a question (e.g. 'What does NHS say about fever with palpitations?'):\n> ")
    query_nhs(query)