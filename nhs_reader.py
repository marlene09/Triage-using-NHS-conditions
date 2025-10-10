from llama_index.readers.web import SimpleWebPageReader
#It fetches web pages (HTML) and converts them into clean text documents that can be indexed.
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
# VectorStoreIndex — creates an index that stores documents as embeddings (vectors).
# StorageContext — manages how/where the index is stored (on disk, in memory, etc).
# load_index_from_storage — reloads a saved index from disk, so you don’t have to rebuild it each time.
from llama_index.core.node_parser import SimpleNodeParser
# SimpleNodeParser — breaks documents into smaller chunks (nodes) for better indexing and retrieval.

from llama_index.embeddings.ollama import OllamaEmbedding
#It lets you use Ollama models to generate embeddings for documents and queries.
import ollama
# from llama_index.llms.ollama import OllamaModel
#Lets you call local LLMs (like Mistral, Llama 3, etc).
import os
from ollama import chat


# --- CONFIG ---
INDEX_DIR = "./nhs_index"
NHS_URLS = [
    "https://www.nhs.uk/conditions/fever-in-adults/",
    "https://www.nhs.uk/conditions/heart-palpitations/",
    "https://www.nhs.uk/conditions/sweating/",
]

# --- STEP 1: Build or load the index ---
"loads an existing index from disk, or"
"builds a new one from the NHS web pages."


def build_or_load_index():
    if os.path.exists(INDEX_DIR):
        print("📂 Loading existing NHS index...")
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage_context, embed_model=OllamaEmbedding(model_name="nomic-embed-text"))
    else:
        print("🌐 Fetching NHS pages...")
        documents = SimpleWebPageReader(html_to_text=True).load_data(urls=NHS_URLS)
        parser = SimpleNodeParser()
        nodes = parser.get_nodes_from_documents(documents)

        print("⚙️ Creating and saving index...")
        embed_model = OllamaEmbedding(model_name="nomic-embed-text")  # small & fast
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        index.storage_context.persist(persist_dir=INDEX_DIR)

    return index


# # --- STEP 2: Query function ---
# def query_nhs(question: str, model_name: str = "mistral") -> str:
#     index = build_or_load_index()
#     llm = ollama.ChatCompletion(model="mistral")
#     query_engine = index.as_query_engine(llm=llm)
#     response = query_engine.query(question)
#     print("\n🧠 NHS Summary:")
#     print(response)
#     return str(response)


from ollama import chat

def query_nhs(question: str) -> str:
    index = build_or_load_index()
    
    # Use Ollama's chat function
    response = chat(model="mistral", messages=[
        {"role": "user", "content": question}
    ])
    
    print("\n🧠 NHS Summary:")
    print(response['message']['content'])
    return str(response['message']['content'])


# --- STEP 3: Example run ---
if __name__ == "__main__":
    print("✅ NHS Reader Ready.")
    query = input("\nAsk a question (e.g. 'What does NHS say about fever with palpitations?'):\n> ")
    query_nhs(query)
