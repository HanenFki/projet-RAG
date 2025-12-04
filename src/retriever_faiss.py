import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss_index(embeddings):
    """
    embeddings : liste de vecteurs (Python)
    On construit un index FAISS L2 normalisé.
    """
    
    # Convertir en float32
    vectors = np.array(embeddings).astype("float32")
    
    # Normaliser pour que L2 = Cosine
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]

    # Index simple basé sur la distance L2
    index = faiss.IndexFlatL2(dimension)

    # Ajouter tous les vecteurs
    index.add(vectors)
    
    return index


def faiss_retrieve(query, chunks, index, k=3):
    """
    Recherche FAISS pour trouver les chunks les plus proches.
    """
    # 1) Embedding de la question
    query_emb = model.encode([query]).astype("float32")

    # Normaliser pour utiliser cosinus via L2
    faiss.normalize_L2(query_emb)

    # 2) Recherche : FAISS retourne distances + indices
    distances, indices = index.search(query_emb, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append((dist, chunks[idx]))

    return results
