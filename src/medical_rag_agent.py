import os
import logging
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Setup logging
logger = logging.getLogger(__name__)

def format_docs(docs):
    """Format documents into a single string for the LLM context."""
    return "\n\n".join(doc.page_content for doc in docs)

class MedicalRAG_Agent:
    """
    Medical-specific RAG Agent using specialized biomedical embeddings.
    Adapted from the InfoDeliver RAG_Agent implementation.
    """

    def __init__(self, 
                 llm,  
                 embedding_model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                 vector_store_path="./medical_faiss_index",
                 device=None):
        """
        Initialize the Medical RAG Agent.
        
        Args:
            llm: Language model from LangChain
            embedding_model_name: Name of the medical embedding model
            vector_store_path: Path to store/load the FAISS index
            device: Device to run embeddings on ('cpu' or 'cuda')
        """
        logger.info(f"Initializing Medical RAG Agent with model: {embedding_model_name}")
        
        # Determine device (CPU/GPU)
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Initialize medical embedding model
            embedding_kwargs = {
                'model_name': embedding_model_name,
                'model_kwargs': {'device': device},
                'encode_kwargs': {'normalize_embeddings': True}
            }
            self.embedding = HuggingFaceEmbeddings(**embedding_kwargs)
            logger.debug(f"Medical embeddings initialized using {embedding_model_name} on {device}")
            
            # Test embeddings
            test_vector = self.embedding.embed_query("diabetes mellitus")
            logger.debug(f"Embedding test successful. Vector dimensions: {len(test_vector)}")
        
        except Exception as e:
            logger.exception(f"Error initializing embeddings: {e}")
            raise
        
        # Store parameters
        self.llm = llm
        self.vector_store_path = vector_store_path
        
        # Initialize or load vector store
        self.vectorstore = None
        
        # Create RAG chain
        self.rag_chain = None
    
    def load_medical_data(self, data_paths: List[str], file_types: Optional[List[str]] = None):
        """
        Load medical data from files.
        
        Args:
            data_paths: List of paths to medical data files
            file_types: List of file types corresponding to each path ('csv', 'txt', etc.)
        """
        logger.info(f"Loading medical data from {len(data_paths)} files")
        
        if file_types is None:
            file_types = ['csv'] * len(data_paths)
        
        all_docs = []
        
        for idx, file_path in enumerate(data_paths):
            file_type = file_types[idx] if idx < len(file_types) else 'csv'
            
            try:
                # Load documents based on file type
                if file_type.lower() == 'csv':
                    loader = CSVLoader(file_path=file_path)
                elif file_type.lower() in ['txt', 'md']:
                    loader = TextLoader(file_path=file_path, encoding='utf-8-sig')
                else:
                    logger.warning(f"Unsupported file type: {file_type}. Treating as text.")
                    loader = TextLoader(file_path=file_path, encoding='utf-8-sig')
                
                docs = loader.load()
                logger.debug(f"Loaded {len(docs)} documents from {file_path}")
                
                all_docs.extend(docs)
            
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
        
        logger.info(f"Total documents loaded: {len(all_docs)}")
        
        # Process and split documents
        return self._process_documents(all_docs)
    
    def _process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process and split documents for the vector store.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of processed document chunks
        """
        logger.debug("Processing and splitting documents")
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        
        # Split documents
        doc_chunks = text_splitter.split_documents(documents)
        logger.debug(f"Created {len(doc_chunks)} document chunks")
        
        return doc_chunks
    
    def create_vector_store(self, documents: List[Document]):
        """
        Create a new vector store with the provided documents.
        
        Args:
            documents: List of Document objects
        """
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        # Create FAISS vector store
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding
        )
        
        # Save the index
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        self.vectorstore.save_local(self.vector_store_path)
        
        logger.info(f"Vector store created and saved to {self.vector_store_path}")
        
        # Create retriever
        self._create_rag_chain()
    
    def load_vector_store(self):
        """Load an existing vector store."""
        if not os.path.exists(self.vector_store_path):
            logger.warning(f"Vector store {self.vector_store_path} does not exist")
            return False
        
        logger.info(f"Loading vector store from {self.vector_store_path}")
        
        try:
            # Set allow_dangerous_deserialization flag to True
            # Note: Only use this with trusted data sources
            self.vectorstore = FAISS.load_local(
                folder_path=self.vector_store_path,
                embeddings=self.embedding,
                allow_dangerous_deserialization=True  # Add this parameter
            )
            
            logger.info("Vector store loaded successfully")
            
            # Create retriever
            self._create_rag_chain()
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
        
    def _create_rag_chain(self):
        """Create the RAG chain for medical term standardization."""
        if self.vectorstore is None:
            logger.error("Vector store not initialized")
            return
        
        logger.debug("Creating RAG chain")
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Create prompt for medical term standardization
        template = """
        You are a medical expert specialized in standardizing medical terminology.
        Your task is to convert non-standard medical diagnostic expressions to formal standard disease names.
        
        Here is relevant medical information:
        {context}
        
        Non-standard diagnostic expression: {question}
        
        Please provide:
        1. The standard disease name that best matches this expression
        2. Your confidence level (0.0-1.0) in this mapping
        3. Alternative possible diagnoses if your confidence is below 0.9
        4. Whether human expert review is needed
        5. Brief reasoning for your decision
        
        Format your response as:
        STANDARD DIAGNOSIS: [disease name]
        CONFIDENCE: [score]
        ALTERNATIVES: [list of alternatives, if any]
        NEEDS REVIEW: [Yes/No]
        REASONING: [brief explanation]
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the RAG chain
        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.debug("RAG chain created successfully")
    
    def standardize_term(self, query: str) -> str:
        """
        Standardize a non-standard medical term using the RAG system.
        
        Args:
            query: Non-standard medical expression
            
        Returns:
            Standardized result with confidence scoring
        """
        logger.info(f"Processing query: {query}")
        
        if self.rag_chain is None:
            logger.error("RAG chain not initialized")
            return "Error: RAG system not initialized. Please load data first."
        
        try:
            # Execute the RAG chain
            result = self.rag_chain.invoke(query)
            logger.debug(f"RAG response: {result}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error: {str(e)}"

    def parse_standardized_result(self, result: str) -> dict:
        """
        Parse the structured result into a dictionary.
        
        Args:
            result: RAG chain result string
            
        Returns:
            Dictionary with parsed results
        """
        parsed = {
            "standard_diagnosis": "",
            "confidence_score": 0.0,
            "alternative_diagnoses": [],
            "needs_human_review": False,
            "reasoning": ""
        }
        
        try:
            lines = result.strip().split("\n")
            
            for line in lines:
                if line.startswith("STANDARD DIAGNOSIS:"):
                    parsed["standard_diagnosis"] = line.replace("STANDARD DIAGNOSIS:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    confidence_str = line.replace("CONFIDENCE:", "").strip()
                    parsed["confidence_score"] = float(confidence_str)
                elif line.startswith("ALTERNATIVES:"):
                    alternatives = line.replace("ALTERNATIVES:", "").strip()
                    if alternatives:
                        parsed["alternative_diagnoses"] = [alt.strip() for alt in alternatives.split(",")]
                elif line.startswith("NEEDS REVIEW:"):
                    needs_review = line.replace("NEEDS REVIEW:", "").strip().lower()
                    parsed["needs_human_review"] = needs_review in ["yes", "true", "1"]
                elif line.startswith("REASONING:"):
                    parsed["reasoning"] = line.replace("REASONING:", "").strip()
            
            return parsed
        
        except Exception as e:
            logger.error(f"Error parsing result: {e}")
            return {
                **parsed,
                "error": str(e),
                "needs_human_review": True
            }

    def create_hybrid_retriever(self):
        """
        Create a hybrid retriever that combines vector, keyword, and fuzzy matching.
        
        This implements the hybrid search mentioned in the PDF:
        - Semantic search (vector similarity)
        - Keyword search (BM25)
        - Edit distance-based fuzzy matching
        - Ensemble combining these methods with configurable weights
        """
        if self.vectorstore is None:
            logger.error("Vector store not initialized")
            return None
        
        logger.info("Creating hybrid retriever")
        
        # 1. Vector search retriever
        vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # 2. BM25 keyword retriever
        from langchain_community.retrievers import BM25Retriever
        
        # Get all documents from vector store to create BM25 index
        # This is a simplification - in production you'd want to use the original documents
        documents = self.vectorstore.get()
        
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 5
        
        # 3. Fuzzy matching retriever using Levenshtein distance
        from langchain_core.retrievers import BaseRetriever
        import Levenshtein
        
        class FuzzyRetriever(BaseRetriever):
            def __init__(self, documents, k=5):
                self.documents = documents
                self.k = k
                
            def get_relevant_documents(self, query):
                # Calculate Levenshtein distance for each document
                scored_docs = []
                for doc in self.documents:
                    # Use minimum distance among content chunks
                    chunks = doc.page_content.split()
                    min_distance = min(Levenshtein.distance(query, chunk) for chunk in chunks)
                    
                    # Normalize by length and convert to similarity score
                    max_len = max(len(query), max(len(chunk) for chunk in chunks))
                    score = 1 - (min_distance / max_len if max_len > 0 else 0)
                    
                    scored_docs.append((doc, score))
                
                # Sort by score (highest first) and take top k
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in scored_docs[:self.k]]
        
        fuzzy_retriever = FuzzyRetriever(documents, k=5)
        
        # 4. Create ensemble retriever
        from langchain_community.retrievers import EnsembleRetriever
        
        # Configure weights for each retriever
        weights = {
            "vector": 0.6,  # Vector search gets highest weight
            "bm25": 0.3,    # Keyword search
            "fuzzy": 0.1    # Fuzzy matching gets lowest weight
        }
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever, fuzzy_retriever],
            weights=[weights["vector"], weights["bm25"], weights["fuzzy"]]
        )
        
        logger.info("Hybrid retriever created successfully")
        return ensemble_retriever


# Example usage:
"""
# Initialize with OpenAI LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
medical_rag = MedicalRAG_Agent(
    llm=llm,
    embedding_model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
)

# Load medical data
data_paths = ["medical_database.csv"]
documents = medical_rag.load_medical_data(data_paths)
medical_rag.create_vector_store(documents)

# Or load existing vector store
# medical_rag.load_vector_store()

# Use for standardization
result = medical_rag.standardize_term("chronic high blood sugar with increased thirst")
parsed_result = medical_rag.parse_standardized_result(result)
print(f"Standard diagnosis: {parsed_result['standard_diagnosis']}")
print(f"Confidence: {parsed_result['confidence_score']}")
"""