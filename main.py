import os
import sys
from loader import load_full_cti_dataset
from processor import get_text_chunks
from vectorstore import create_or_update_vectorstore, load_local_vectorstore
from rag_engine import get_rag_chain

# Configuration pour l'affichage Windows (UTF-8)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    # --- CONFIGURATION DES CHEMINS ---
    POSTS_PATH = "data/posts" 
    REPLIES_PATH = "data/replies"
    INDEX_NAME = "faiss_index_cti"

    print("=== SYSTÈME CTI DARKGRAM (PHI-3.5 + OLLAMA) ===")

    # --- ÉTAPE 1, 2 & 3 : PRÉPARATION DES DONNÉES ---
    # On vérifie si l'index FAISS existe déjà pour gagner du temps
    if os.path.exists(INDEX_NAME):
        print(f"📁 Index '{INDEX_NAME}' détecté. Chargement en cours...")
        vector_db = load_local_vectorstore(INDEX_NAME)
    else:
        print("🚀 Index non trouvé. Initialisation du pipeline complet...")
        
        # 1. Chargement et Nettoyage (Page 1)
        raw_documents = load_full_cti_dataset(POSTS_PATH, REPLIES_PATH)
        if not raw_documents:
            print("❌ Erreur : Aucun document n'a été chargé. Vérifiez vos dossiers data/.")
            return

        # 2. Découpage en Chunks (Page 2)
        print(f"✂️ Découpage de {len(raw_documents)} documents...")
        final_chunks = get_text_chunks(raw_documents)

        # 3. Création de la base vectorielle (Page 4)
        print("🏗️ Création de la base FAISS (cela peut prendre du temps selon votre CPU)...")
        vector_db = create_or_update_vectorstore(final_chunks, INDEX_NAME)

    # --- ÉTAPE 4 : INITIALISATION DU MOTEUR RAG (Page 5) ---
    print("🤖 Connexion à Phi-3.5 via Ollama...")
    try:
        rag_system = get_rag_chain(vector_db)
        print("✅ Système prêt pour l'analyse !\n")
    except Exception as e:
        print(f"❌ Erreur de connexion à Ollama : {e}")
        print("Assurez-vous qu'Ollama est lancé et que le modèle phi3.5 est téléchargé.")
        return

    # --- ÉTAPE 5 : BOUCLE DE CHAT (INTERFACE TERMINAL) ---
    print("--- POSEZ VOS QUESTIONS (tapez 'exit' pour quitter) ---")
    while True:
        query = input("\n🔍 Question CTI : ")
        
        if query.lower() in ['exit', 'quitter', 'quit']:
            print("👋 Fermeture du système.")
            break

        if not query.strip():
            continue

        print("⏳ Recherche et analyse en cours...")
        try:
            # Appel de la chaîne RAG
            response = rag_system.invoke({"query": query})
            
            # Affichage de la réponse de Phi-3.5
            print("\n📝 ANALYSE DE L'IA :")
            print(response["result"])
            
            # Affichage des sources (Optionnel mais recommandé pour le PFE)
            print("\n📚 SOURCES RÉCUPÉRÉES :")
            sources = set()
            for doc in response["source_documents"]:
                # On récupère le nom du fichier ou de la source dans les metadata
                source_info = doc.metadata.get('source', 'Inconnu')
                sources.add(source_info)
            
            for s in sources:
                print(f"- {s}")
                
        except Exception as e:
            print(f"❌ Une erreur est survenue lors de la génération : {e}")

if __name__ == "__main__":
    main()