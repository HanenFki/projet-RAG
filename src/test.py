# from docx import Document

# def read_docx(path):
#     doc = Document(path)
#     text = ""
#     for para in doc.paragraphs:
#         text += para.text + "\n"
#     return text

# # Test
# text = read_docx("docs/RAG.docx")
# print(text[:500])  # affiche les 500 premiers caractères

from utils import read_pdf
from indexer import chunk_text
from embedder import embed_chunks
from retriever_faiss import build_faiss_index, faiss_retrieve

# Charger document
text = read_pdf("../docs/rag.pdf")

# Découper en chunks
chunks = chunk_text(text)

# Générer embeddings
embeddings = embed_chunks(chunks)

# Construire index FAISS
index = build_faiss_index(embeddings)

# Question test
question = "Qu’est-ce que le RAG ?"

# Recherche FAISS
results = faiss_retrieve(question, chunks, index, k=3)

print("\n=== Résultats FAISS ===")
for dist, chunk in results:
    print(f"\nDistance : {dist:.4f}")
    print(chunk[:250], "...")
