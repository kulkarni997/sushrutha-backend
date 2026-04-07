from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

_vectorstore = None

def get_vectorstore(index_path: str = "rag/faiss_index"):
    global _vectorstore
    if _vectorstore is None:
        if not os.path.exists(index_path):
            return None
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        _vectorstore = FAISS.load_local(
            index_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
    return _vectorstore

def retrieve_context(dosha: str, symptoms: str = "", top_k: int = 5) -> str:
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return ""
    
    query = f"{dosha} dosha imbalance {symptoms}"
    docs = vectorstore.similarity_search(query, k=top_k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context