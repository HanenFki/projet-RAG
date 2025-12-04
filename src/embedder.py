from indexer import chunk_text
from utils import load_document
from sentence_transformers import SentenceTransformer

# Charger un petit modèle rapide
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks):
    """
    Reçoit une liste de chunks (texte)
    Retourne une liste de vecteurs (embeddings)
    """
    embeddings = model.encode(chunks)
    return embeddings

#Test 
# Charger un texte
text = load_document("../docs/RAG.docx")

# Chunking
chunks = chunk_text(text)

print("Nombre de chunks :", len(chunks))

# Embeddings
vectors = embed_chunks(chunks)

print("Nombre de vecteurs générés :", len(vectors))
print("Dimension d’un vecteur :", len(vectors[0]))

# Afficher un vecteur (optionnel)
print("\nPremier embedding :\n", vectors[0])