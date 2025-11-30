from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils import load_document

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    #Découpe le texte en morceaux (chunks) avec chevauchement.
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  
    return chunks
text = load_document("../docs/RAG.docx")
chunks = chunk_text(text)
print(f"Nombre de chunks : {len(chunks)}")
print("Premier chunk :\n", chunks[0])