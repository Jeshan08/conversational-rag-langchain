from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def get_retriever(chunks):
  embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  
  )
  
  if os.path.exists("./chroma_db"):
    print("Loding existing vector store")
    vectorstore = Chroma(
      persist_directory="./chroma_db",
      embedding_function=embeddings,
    )
  else: 
    print("Creating new vector store...")
    vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
  )
  
  retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
  )
  
  return retriever