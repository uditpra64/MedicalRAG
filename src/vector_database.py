from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd

def load_medical_data(file_path, file_type="csv"):
    """
    Load medical data from a file.
    
    Args:
        file_path (str): Path to the data file
        file_type (str): Type of the file ('csv', 'txt', etc.)
        
    Returns:
        list: List of Document objects
    """
    if file_type.lower() == "csv":
        loader = CSVLoader(file_path=file_path)
    elif file_type.lower() == "txt":
        loader = TextLoader(file_path=file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    documents = loader.load()
    return documents

def process_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into chunks for processing.
    
    Args:
        documents (list): List of Document objects
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of processed document chunks
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"
    )
    
    doc_chunks = text_splitter.split_documents(documents)
    return doc_chunks

def setup_vector_db(documents, embeddings, persist_directory="./medical_chroma_db"):
    """
    Set up a vector database with the provided documents and embeddings.
    
    Args:
        documents (list): List of Document objects
        embeddings: Embedding model
        persist_directory (str): Directory to save the vector database
        
    Returns:
        Chroma: Vector database instance
    """
    # Create and persist the vector database
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Persist the database
    vectordb.persist()
    print(f"Vector database created and saved to {persist_directory}")
    
    return vectordb

def load_existing_vectordb(embeddings, persist_directory="./medical_chroma_db"):
    """
    Load an existing vector database.
    
    Args:
        embeddings: Embedding model
        persist_directory (str): Directory where the vector database is saved
        
    Returns:
        Chroma: Vector database instance
    """
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Loaded existing vector database from {persist_directory}")
    
    return vectordb

