# =======================================================
# NHS Pediatric Rash Chatbot POC (Dynamic + Probabilities)
# =======================================================

import os
from PIL import Image
import torch
import clip  # OpenAI CLIP package
from llama_index.readers.web import SimpleWebPageReader  # To fetch NHS pages as text
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SimpleNodeParser  # Breaks text into chunks for indexing
from llama_index.embeddings.ollama import OllamaEmbedding  # Embedding model for indexing
from ollama import chat  # Local LLM (Mistral) for summarization
from llama_index.core.query_engine import RetrieverQueryEngine


# -------------------------
# CONFIGURATION
# -------------------------
INDEX_DIR = "./nhs_index"  # Where the NHS vector index is saved
NHS_RASH_URLS = [
    "https://www.nhs.uk/conditions/skin-rash-children/",
    "https://www.nhs.uk/conditions/skin-rash-babies/",
]

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# LOAD CLIP MODEL
# -------------------------
# CLIP maps images and text into the same embedding space
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# -------------------------
# BUILD OR LOAD VECTOR INDEX
# -------------------------
def build_or_load_index():
    """
    Load existing NHS index from disk if available.
    Otherwise, fetch NHS pages dynamically, parse them into nodes,
    embed them, and persist to disk.
    """
    if os.path.exists(INDEX_DIR):
        # Load existing index for faster queries
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage_context, embed_model=OllamaEmbedding(model_name="nomic-embed-text"))
    else:
        # Fetch NHS pages as text
        documents = SimpleWebPageReader(html_to_text=True).load_data(urls=NHS_RASH_URLS)
        parser = SimpleNodeParser()
        nodes = parser.get_nodes_from_documents(documents)
        # Create embeddings and store
        embed_model = OllamaEmbedding(model_name="nomic-embed-text")
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        index.storage_context.persist(persist_dir=INDEX_DIR)
    return index

# -------------------------
# DYNAMIC RASH CONDITION EXTRACTION
# -------------------------
def extract_conditions_from_index(index):
    """
    Extract pediatric rash names dynamically from NHS index using local retrieval.
    """
    retriever = index.as_retriever()  # get the retriever
    results = retriever.retrieve("List all types of pediatric rashes mentioned in the text")
    
    # Collect all text from retrieved nodes
    text_snippets = "\n".join([node.get_text() for node in results])
    
    # Simple line-based extraction of conditions
    conditions = set()
    for line in text_snippets.split("\n"):
        line = line.strip()
        if "rash" in line.lower():
            conditions.add(line)
    
    # Fallback if extraction fails
    if not conditions:
        conditions = {"Hand Foot Mouth Disease", "Chickenpox", "Measles", "Scarlet Fever", "Eczema", "Allergic Rash", "Viral Exanthem"}
    
    return list(conditions)


# -------------------------
# IMAGE-TO-CONDITION PROBABILITIES USING CLIP
# -------------------------
def classify_rash_clip(image_path, conditions):
    """
    Encode image and text conditions into CLIP embeddings,
    compute cosine similarity, and output probabilities.
    """
    # Preprocess image
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    
    # Tokenize all conditions dynamically
    text_tokens = clip.tokenize(conditions).to(device)
    
    with torch.no_grad():
        # Encode image and text
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text_tokens)
        
        # Normalize embeddings
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Cosine similarity
        similarities = (image_features @ text_features.T).squeeze(0)
        
        # Convert similarities to probabilities via softmax
        probs = torch.softmax(similarities, dim=0)
    
    # Build dictionary {condition: probability}
    class_probs = {conditions[i]: float(probs[i]) for i in range(len(conditions))}
    
    # Structured image description for prompt
    top_condition = max(class_probs, key=class_probs.get)
    description = f"Image features suggest rash: {top_condition} (highest probability: {class_probs[top_condition]*100:.1f}%)"
    
    return class_probs, description

# -------------------------
# TEXT-BASED SCORING
# -------------------------
def extract_text_scores(patient_text, conditions):
    """
    Simple scoring based on text mentions of conditions.
    Can be replaced with NLP-based similarity scoring.
    """
    text = patient_text.lower()
    scores = {cond: 0.0 for cond in conditions}
    for cond in conditions:
        cond_lower = cond.lower()
        if cond_lower in text:
            scores[cond] += 0.2  # lower weight to reduce text domination
    return scores

# -------------------------
# FUSION & TOP-N
# -------------------------
def fuse_scores(image_probs, text_scores, alpha=0.8, beta=0.2, top_n=5):
    """
    Weighted fusion of image and text scores.
    """
    final_scores = {}
    for cond in image_probs.keys():
        final_scores[cond] = alpha*image_probs.get(cond, 0) + beta*text_scores.get(cond, 0)
    
    # Sort descending by score and take top N
    sorted_top = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Ensure each entry is a tuple (condition, score)
    return [(cond, score) for cond, score in sorted_top]


# -------------------------
# CREATE STRUCTURED PROMPT
# -------------------------
def create_structured_prompt(patient_text, image_description, top_conditions):
    """
    Generate a prompt with top-N provisional diagnoses including probabilities.
    """
    # top_conditions must be list of (condition, score) tuples
    top_str = "\n".join([
        f"{i+1}. {cond} ({score*100:.1f}%)"
        for i, (cond, score) in enumerate(top_conditions)
    ])

    prompt = f"""
Chief complaint: {patient_text}
Rash image features: {image_description}

Top 3 provisional diagnoses (with probabilities):
{top_str}

Instructions for clinician review:

- Describe physical characteristics for each provisional diagnosis.
- Include typical course, symptomatic relief, and red flags.
- Ensure image evidence is considered in the reasoning.
"""
    return prompt



# -------------------------
# QUERY NHS
# -------------------------
def query_nhs(patient_text, image_path=None):
    """
    Main function to:
    - Load/build index
    - Extract conditions dynamically
    - Compute image + text probabilities
    - Fuse top-N conditions
    - Query LLM for structured summary
    """
    index = build_or_load_index()
    conditions = extract_conditions_from_index(index)
    
    image_probs = {cond:0.0 for cond in conditions}
    image_description = ""
    
    if image_path:
        image_probs, image_description = classify_rash_clip(image_path, conditions)
    
    text_scores = extract_text_scores(patient_text, conditions)
    
    # Weighted fusion (image priority)
    top_conditions = fuse_scores(image_probs, text_scores, alpha=0.8, beta=0.2)
    
    # Structured prompt for LLM
    structured_prompt = create_structured_prompt(patient_text, image_description, top_conditions)
    
    # Query LLM
    response = chat(model="mistral", messages=[{"role": "user", "content": structured_prompt}])
    
    print("\n🧠 Provisional NHS Summary (for clinician review):")
    print(response['message']['content'])
    return response['message']['content']

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    print("✅ NHS Pediatric Rash Chatbot Ready.")
    patient_text = input("\nEnter patient symptoms and observations:\n> ")
    img_choice = input("Do you want to include an image? (y/n): ").lower().strip()
    
    if img_choice == 'y':
        img_path = input("Enter the image file path:\n> ")
        query_nhs(patient_text, image_path=img_path)
    else:
        query_nhs(patient_text)
