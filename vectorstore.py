from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_or_update_vectorstore(chunks, storage_path="faiss_index_cti"):
    """
    Transforme les chunks en vecteurs et les sauvegarde localement.
    """
    print("🧠 Initialisation du modèle d'embeddings (HuggingFace)...")
    
    # Utilisation du modèle recommandé par votre guide RAG1.pdf
    # 'all-mpnet-base-v2' est excellent pour la similarité sémantique technique
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'} # Changez en 'cuda' si vous avez une carte GPU
    )

    print(f"📊 Création de la base vectorielle pour {len(chunks)} morceaux...")
    
    # Création de la base FAISS à partir des documents
    vector_db = FAISS.from_documents(chunks, embeddings)

    # Sauvegarde locale pour éviter de tout recalculer la prochaine fois
    vector_db.save_local(storage_path)
    
    print(f"✅ Base FAISS sauvegardée dans le dossier : {storage_path}")
    return vector_db

def load_local_vectorstore(storage_path="faiss_index_cti"):
    """
    Charge une base FAISS existante sans avoir à relire les CSV.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    if os.path.exists(storage_path):
        return FAISS.load_local(storage_path, embeddings, allow_dangerous_deserialization=True)
    return None