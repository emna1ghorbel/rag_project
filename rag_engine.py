from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def get_rag_chain(vector_db):
    """
    Relie FAISS à Phi-3.5 tournant localement sur Ollama.
    """
    
    # 1. Configuration du LLM Local (Ollama)
    llm = OllamaLLM(
        model="phi3.5",  # Le nom exact dans Ollama (vérifiez avec 'ollama list')
        temperature=0,
    )

    # 2. Template de Prompt CTI (Français/Anglais selon votre besoin)
    template = """Vous êtes un analyste en Cyber Threat Intelligence (CTI).
    Utilisez les informations suivantes issues de Telegram pour répondre à la question.
    Soyez technique et précis. Si l'information n'est pas dans le contexte, dites-le.

    CONTEXTE : {context}

    QUESTION : {question}

    ANALYSE CTI :"""

    PROMPT = PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )

    # 3. Création de la chaîne RAG
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return chain