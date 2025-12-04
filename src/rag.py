# import os
# from dotenv import load_dotenv  # Si tu utilises .env
# load_dotenv()  # Charge les vars si .env existe

# from pathlib import Path
# import numpy as np
# import faiss
# import torch
# from groq import Groq  # Client Groq

# # Modules pour PDF / embeddings (inchangés)
# from utils import load_document
# from indexer import chunk_text
# from embedder import embed_chunks

# # === 1) Charger le document PDF ===
# path = input("Chemin du document : ")
# text = load_document(path)

# # === 2) Chunking du texte ===
# chunks = chunk_text(text)

# # === 3) Embeddings pour les chunks ===
# chunk_embeddings = embed_chunks(chunks)  # Assume liste ou array de vecteurs
# if len(chunk_embeddings) == 0:  # Safe pour lists/arrays
#     raise ValueError("Aucun embedding généré pour les chunks. Vérifiez embed_chunks.")
# embeddings = np.array(chunk_embeddings).astype("float32")
# print(f"Shape des embeddings : {embeddings.shape}")  # Debug

# # === 4) Création de l'index FAISS ===
# dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings)
# print(f"Index FAISS créé avec {index.ntotal} vecteurs.")  # Debug

# # === 5) Config Groq ===
# groq_api_key = os.getenv('GROQ_API_KEY') or input("Entrez votre clé Groq API : ")
# client = Groq(api_key=groq_api_key)

# print("\n=== RAG prêt. Posez vos questions ! Tapez 'exit' pour quitter ===\n")

# # === 6) Boucle de questions ===
# while True:
#     query = input("Question : ")
#     if query.lower() in ["exit", "quit"]:
#         print("Fin du programme.")
#         break

#     # 6a) Embedding de la question et recherche (CORRIGÉ pour ligne 49)
#     query_embedding = embed_chunks([query])  # Liste avec 1 embedding
#     if len(query_embedding) == 0:  # Fix: utilise len() au lieu de "not" pour éviter l'ambiguïté avec arrays
#         print("Erreur : Embedding de query vide.")
#         continue
#     vec = np.array(query_embedding).astype("float32")  # (1, dim)
#     print(f"Shape de vec : {vec.shape}")  # Debug

#     distances, indices = index.search(vec, 3)  # top 3
#     print(f"Distances : {distances[0]} | Indices : {indices[0]}")  # Debug

#     valid_indices = [i for i, dist in zip(indices[0], distances[0]) if i >= 0 and dist < 1e6]
#     if not valid_indices:
#         print("Aucun chunk pertinent trouvé.")
#         continue

#     retrieved = [chunks[i] for i in valid_indices]
#     context = "\n\n".join(retrieved)
#     print(f"Contexte retrouvé : {context[:200]}...")  # Debug

#     # 6b) Génération avec Groq
#     prompt = f"Tu es un assistant RAG. Tu réponds UNIQUEMENT à partir du contexte suivant :\n{context}\n\nQuestion : {query}\nRéponse claire et courte :"
    
#     response = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=200,
#         temperature=0.0
#     )
#     answer = response.choices[0].message.content
#     print("\n=== RÉPONSE RAG ===\n", answer, "\n")
import os
import streamlit as st  # Nouveau : Pour l'interface graphique
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import faiss
from groq import Groq

# Modules pour PDF / embeddings (assume ils existent ; adapte si besoin)
from utils import load_document  # Modifie pour gérer upload (voir ci-dessous)
from indexer import chunk_text
from embedder import embed_chunks

# Fonction modifiée pour load_document (gère upload Streamlit)
def load_document_from_upload(uploaded_file):
    if uploaded_file is None:
        return ""
    if uploaded_file.name.endswith('.pdf'):
        # Assume ton load_document gère les paths ; ici, on écrit temp file
        temp_path = f"./temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        text = load_document(temp_path)
        os.remove(temp_path)  # Nettoyage
        return text
    elif uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8')
    else:
        st.error("Format non supporté. Utilisez PDF ou TXT.")
        return ""

# Config Groq
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    st.error("Clé GROQ_API_KEY manquante. Ajoutez-la dans .env.")
    st.stop()
client = Groq(api_key=groq_api_key)

# Interface Streamlit
st.title("Interrogez vos Documents")
st.markdown("Upload un PDF/TXT, posez une question, et obtenez une réponse contextualisée !")

# Upload du document
uploaded_file = st.file_uploader("Choisissez un document (PDF ou TXT)", type=['pdf', 'txt'])

# Persistance via session state
if 'index' not in st.session_state or uploaded_file != st.session_state.get('last_file'):
    if uploaded_file:
        with st.spinner("Traitement du document..."):
            text = load_document_from_upload(uploaded_file)
            if text:
                chunks = chunk_text(text)
                chunk_embeddings = embed_chunks(chunks)
                if len(chunk_embeddings) == 0:
                    st.error("Aucun embedding généré. Vérifiez embed_chunks.")
                    st.stop()
                embeddings = np.array(chunk_embeddings).astype("float32")
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)
                
                # Stocke dans session state
                st.session_state['index'] = index
                st.session_state['chunks'] = chunks
                st.session_state['last_file'] = uploaded_file
                st.success(f"Document chargé ! {len(chunks)} chunks, {index.ntotal} vecteurs.")
            else:
                st.error("Erreur lors du chargement du document.")

# Input question
query = st.text_input("Votre question :", "")
if st.button("Envoyer") and query and 'index' in st.session_state:
    with st.spinner("Recherche et génération..."):
        # Retrieval
        query_embedding = embed_chunks([query])
        if len(query_embedding) == 0:
            st.error("Embedding de query vide.")
        else:
            vec = np.array(query_embedding).astype("float32")
            distances, indices = st.session_state['index'].search(vec, 3)
            valid_indices = [i for i, dist in zip(indices[0], distances[0]) if i >= 0 and dist < 1e6]
            if not valid_indices:
                st.warning("Aucun chunk pertinent trouvé.")
            else:
                retrieved = [st.session_state['chunks'][i] for i in valid_indices]
                context = "\n\n".join(retrieved)
                
                # Debug affiché
                st.markdown("**Debug Retrieval :**")
                st.write(f"Distances : {distances[0]} | Indices : {indices[0]}")
                st.text_area("Contexte retrouvé :", context, height=150)
                
                # Génération (prompt personnalisé exemple ; adapte ici)
                prompt = f"Tu es un professeur expert en IA et RAG. Explique de manière pédagogique, en utilisant des exemples simples, en te basant UNIQUEMENT sur ce contexte :\n{context}\n\nQuestion de l'étudiant : {query}\nRéponse structurée : 1. Définition clé, 2. Explication, 3. Exemple."
                
                try:
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2000,
                        temperature=0.0
                    )
                    answer = response.choices[0].message.content
                    st.markdown("**Réponse RAG :**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Erreur Groq : {e}")