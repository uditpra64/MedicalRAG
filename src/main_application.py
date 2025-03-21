import os
import logging
import torch
from typing import List, Dict, Any, Optional, Union

# Import components
from src.medical_rag_agent import MedicalRAG_Agent
from src.azure_openai_wrapper import AzureOpenAIWrapper
from src.medical_ontology import MedicalOntologyManager
from src.feedback_manager import FeedbackManager

class MedicalDiseaseNameSearchSystem:
    """
    A system for standardizing non-standard medical expressions to formal disease names
    using RAG with medical-specific embeddings.
    """
    
    def __init__(
        self, 
        embedding_model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        vector_db_path: str = "./data/vector_stores/sapbert_faiss",
        llm_model_name: str = "gptest", # Default Azure model name
        confidence_threshold: float = 0.7,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        azure_endpoint: str = "https://formaigpt.openai.azure.com",
        device: Optional[str] = None,
        icd10_path: Optional[str] = None,
        snomed_ct_path: Optional[str] = None
    ):
        """
        Initialize the Medical Disease Name Search System.
        
        Args:
            embedding_model_name: Name of the medical embedding model
            vector_db_path: Path to store/load the vector database
            llm_model_name: Name of the Azure OpenAI model deployment
            confidence_threshold: Threshold for confidence score
            api_key: API key for Azure OpenAI (if not set in environment)
            api_version: Azure OpenAI API version
            azure_endpoint: Azure OpenAI endpoint URL
            device: Device to run embeddings on ('cpu' or 'cuda')
            icd10_path: Path to ICD-10 data file
            snomed_ct_path: Path to SNOMED CT data folder
        """
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing Medical Disease Name Search System with {embedding_model_name}")
        
        # Set configuration
        self.embedding_model_name = embedding_model_name
        self.vector_db_path = vector_db_path
        self.llm_model_name = llm_model_name
        self.confidence_threshold = confidence_threshold
        self.api_key = api_key
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        
        # Determine device (CPU/GPU)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self._initialize_components()

        self.ontology_manager = MedicalOntologyManager(
            icd10_path=icd10_path,
            snomed_ct_path=snomed_ct_path
        )
        self.feedback_manager = FeedbackManager()

    def record_expert_feedback(self, query, system_diagnosis, expert_diagnosis, is_correct, expert_notes=None):
        '''Record expert feedback for a diagnosis.'''
        return self.feedback_manager.record_feedback(
            query=query,
            system_diagnosis=system_diagnosis,
            expert_diagnosis=expert_diagnosis,
            confidence_score=self.last_confidence_score,
            is_correct=is_correct,
            expert_notes=expert_notes
        )

    def _initialize_components(self):
        """Initialize system components."""
        try:
            # Initialize LLM using Azure OpenAI wrapper
            self.llm = AzureOpenAIWrapper(
                model=self.llm_model_name,
                temperature=0,
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint
            )
            self.logger.info(f"Azure OpenAI LLM initialized: {self.llm_model_name}")
            
            # Initialize Medical RAG
            self.medical_rag = MedicalRAG_Agent(
                llm=self.llm,
                embedding_model_name=self.embedding_model_name,
                vector_store_path=self.vector_db_path,
                device=self.device
            )
            self.logger.info("Medical RAG initialized")
            
            # Check for existing vector store
            vector_store_exists = self.medical_rag.load_vector_store()
            
            if vector_store_exists:
                self.logger.info(f"Loaded existing vector store from {self.vector_db_path}")
            else:
                self.logger.warning("No vector store found. Use load_data() to create one.")
                
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
            
    def load_data(self, data_paths: Union[str, List[str]], file_types: Optional[List[str]] = None):
        """
        Load medical data and create vector database.
        
        Args:
            data_paths: Path or list of paths to medical data files
            file_types: List of file types corresponding to each path ('csv', 'txt', etc.)
            
        Returns:
            bool: Success status
        """
        try:
            # Convert single path to list
            if isinstance(data_paths, str):
                data_paths = [data_paths]
                
            self.logger.info(f"Loading data from {len(data_paths)} files")
            
            # Load documents
            documents = self.medical_rag.load_medical_data(data_paths, file_types)
            
            # Create vector store
            self.medical_rag.create_vector_store(documents)
            
            self.logger.info("Data loaded and vector store created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False
            
    def convert_medical_expression(
        self, 
        expression: str,
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Convert a non-standard medical expression to a standardized disease name.
        
        Args:
            expression: Non-standard medical expression
            confidence_threshold: Override the default confidence threshold
            
        Returns:
            dict: Standardized results with confidence scoring
        """
        # Use instance threshold if none provided
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            
        try:
            self.logger.info(f"Processing query: '{expression}'")
            
            # Use the Medical RAG for standardization
            result = self.medical_rag.standardize_term(expression)
            
            # Parse the result
            parsed_result = self.medical_rag.parse_standardized_result(result)
            
            # Log the result
            self.logger.info(f"Result: {parsed_result['standard_diagnosis']} " 
                            f"(confidence: {parsed_result['confidence_score']:.2f})")
            
            # Store last confidence score for feedback
            self.last_confidence_score = parsed_result['confidence_score']

            # Queue for expert review based on confidence and severity
            needs_review = parsed_result.get('needs_human_review', False)
            critical_condition = self._is_critical_condition(parsed_result['standard_diagnosis'])
            low_confidence = parsed_result['confidence_score'] < 0.7
            moderate_confidence = 0.7 <= parsed_result['confidence_score'] < 0.85
            
            # Determine queue priority
            if low_confidence or (critical_condition and parsed_result['confidence_score'] < 0.9):
                # High priority: Low confidence OR critical condition with < 0.9 confidence
                self.feedback_manager.queue_for_expert_review(
                    query=expression,
                    system_diagnosis=parsed_result['standard_diagnosis'],
                    confidence_score=parsed_result['confidence_score'],
                    alternative_diagnoses=parsed_result.get('alternative_diagnoses', []),
                    priority="high_priority"
                )
            elif needs_review or (moderate_confidence and critical_condition):
                # Medium priority: Marked for review OR moderate confidence for critical conditions
                self.feedback_manager.queue_for_expert_review(
                    query=expression,
                    system_diagnosis=parsed_result['standard_diagnosis'],
                    confidence_score=parsed_result['confidence_score'],
                    alternative_diagnoses=parsed_result.get('alternative_diagnoses', []),
                    priority="medium_priority"
                )

            return parsed_result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "standard_diagnosis": "Error occurred",
                "confidence_score": 0.0,
                "alternative_diagnoses": [],
                "needs_human_review": True,
                "reasoning": f"Error: {str(e)}",
                "error": str(e)
            }
            
    def _is_critical_condition(self, disease_name: str) -> bool:
        """
        Determine if a disease is a critical condition requiring careful review.
        
        Args:
            disease_name: Standard disease name
            
        Returns:
            bool: True if critical condition, False otherwise
        """
        critical_conditions = [
            "myocardial infarction", "heart attack", 
            "stroke", "cerebrovascular accident",
            "pulmonary embolism", "sepsis",
            "meningitis", "encephalitis",
            "pneumonia", "covid"
        ]
        
        return any(critical in disease_name.lower() for critical in critical_conditions)
            
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a direct search without LLM generation.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            list: List of matching documents
        """
        try:
            # Get relevant documents
            documents = self.medical_rag.vectorstore.similarity_search(query, k=top_k)
            
            # Convert to dictionary format
            results = []
            for doc in documents:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": getattr(doc, "score", None)
                })
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error in search: {e}")
            return []
            
    def get_medical_ontologies(self, disease_name: str) -> Dict[str, Any]:
        """
        Get related medical ontology information for a disease name.
        
        Args:
            disease_name: Standard disease name
            
        Returns:
            dict: Related ontology information (ICD-10, SNOMED CT, etc.)
        """
        # This would be implemented when medical_ontology module is available
        return self.ontology_manager.get_ontology_data(disease_name)
    
        
    def evaluate(self, test_data_path: str) -> Dict[str, Any]:
        """
        Evaluate the system with a test dataset.
        
        Args:
            test_data_path: Path to test data CSV
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            from utils.evaluation import MedicalSystemEvaluator
            
            evaluator = MedicalSystemEvaluator(self)
            metrics = evaluator.evaluate_with_testset(test_data_path)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in evaluation: {e}")
            return {"error": str(e)}