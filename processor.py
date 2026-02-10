from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Recommandé pour la CTI
        chunk_overlap=100,     # Pour ne pas perdre le contexte
        separators=["\n\n", "\n", " ", ""], # Ne pas couper sur "." à cause des URLs
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks