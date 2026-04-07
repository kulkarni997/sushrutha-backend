from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def ingest_pdfs(pdf_folder: str = "rag/pdfs", index_path: str = "rag/faiss_index"):
    documents = []
    
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        print(f"Created {pdf_folder} — add Ayurvedic PDFs here")
        return
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("No PDFs found in rag/pdfs/ — add Sushruta/Charaka PDFs")
        return
    
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
        documents.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"FAISS index saved to {index_path}")

if __name__ == "__main__":
    ingest_pdfs()