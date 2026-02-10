import pandas as pd
import glob
import os
import sys
import re
from langchain.docstore.document import Document

# Fix pour l'affichage des caractères spéciaux sur Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def clean_cti_text(text):
    """
    Nettoie le texte en préservant les URLs, points et caractères techniques.
    Indispensable pour la Cyber Threat Intelligence (IoC).
    """
    if pd.isna(text):
        return ""
    # On remplace les sauts de ligne et espaces multiples par un seul espace
    text = re.sub(r'\s+', ' ', str(text))
    return text.strip()

def load_full_cti_dataset(posts_root, replies_root):
    """
    Charge les posts et leurs réponses de manière optimisée.
    Structure attendue : posts/{category}/{channel}.csv
    """
    all_docs = []
    
    # 1. On liste tous les fichiers CSV dans le dossier posts
    post_files = glob.glob(os.path.join(posts_root, "**/*.csv"), recursive=True)
    
    print(f"📂 Analyse de {len(post_files)} fichiers de posts...")

    for file in post_files:
        try:
            # Lecture rapide en forçant le type string pour éviter les DtypeWarning
            df_posts = pd.read_csv(file, low_memory=False, dtype=str)
            df_posts.columns = [c.strip().lower() for c in df_posts.columns]
            
            if 'message' not in df_posts.columns:
                continue

            # Extraction de la catégorie et du nom du canal pour la reconstruction du chemin
            normalized_path = os.path.normpath(file)
            path_parts = normalized_path.split(os.sep)
            
            channel_name = os.path.basename(file).replace('.csv', '')
            # On récupère le dossier parent (ex: malware, phishing, etc.)
            category = path_parts[-2] if len(path_parts) > 1 else ""

            for _, row in df_posts.iterrows():
                if pd.isna(row['message']): 
                    continue
                
                # Identifiant du post (pour lier les replies)
                p_id = row.get('post id', row.get('id', 'unknown'))
                
                # Nettoyage du contenu
                clean_content = clean_cti_text(row['message'])
                
                # --- ÉTAPE 1 : Création du Document pour le Post Original ---
                post_text = f"CHANNEL: {channel_name}\nCONTENT: {clean_content}"
                metadata = {
                    "source": row.get('url', 'N/A'),
                    "date": str(row.get('date', 'N/A')),
                    "type": "original_post",
                    "post_id": p_id,
                    "channel": channel_name,
                    "category": category
                }
                all_docs.append(Document(page_content=post_text, metadata=metadata))
                
                # --- ÉTAPE 2 : OPTIMISATION DES REPLIES (Accès Direct) ---
                if p_id != 'unknown':
                    # On construit le chemin vers le fichier de réponses spécifique à ce post
                    # Format : replies/{category}/{channel}/{post_id}.csv
                    reply_file_path = os.path.join(replies_root, category, channel_name, f"{p_id}.csv")
                    
                    if os.path.exists(reply_file_path):
                        try:
                            df_replies = pd.read_csv(reply_file_path, low_memory=False, dtype=str)
                            df_replies.columns = [c.strip().lower() for c in df_replies.columns]
                            
                            if 'message' in df_replies.columns:
                                for _, r_row in df_replies.iterrows():
                                    if pd.isna(r_row['message']): 
                                        continue
                                    
                                    r_clean = clean_cti_text(r_row['message'])
                                    reply_text = f"REPLY TO POST {p_id} ({channel_name}): {r_clean}"
                                    
                                    all_docs.append(Document(
                                        page_content=reply_text, 
                                        metadata={
                                            "type": "reply", 
                                            "parent_post": p_id, 
                                            "channel": channel_name,
                                            "category": category,
                                            "source": r_row.get('url', 'N/A')
                                        }
                                    ))
                        except Exception:
                            # Si un fichier reply est corrompu, on continue sans bloquer
                            continue

        except Exception as e:
            print(f"❌ Erreur lors du traitement de {file}: {e}")
                    
    return all_docs