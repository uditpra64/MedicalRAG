import os
import sys
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the MedicalDiseaseNameSearchSystem
from src.main_application import MedicalDiseaseNameSearchSystem

def test_with_queries(system, queries):
    """Run a list of test queries through the system"""
    print("\n" + "=" * 60)
    print("Testing Medical RAG System with Azure OpenAI")
    print("=" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: '{query}'")
        print("-" * 60)
        
        try:
            # Process the query
            result = system.convert_medical_expression(query)
            
            # Display results
            print(f"Standard Diagnosis: {result['standard_diagnosis']}")
            print(f"Confidence: {result['confidence_score']:.2f}")
            
            if result.get('alternative_diagnoses'):
                print("\nAlternative Diagnoses:")
                for alt in result['alternative_diagnoses']:
                    print(f"  - {alt}")
            
            if result.get('needs_human_review', False):
                print("\n⚠️ This case needs human review")
            
            if result.get('reasoning'):
                print(f"\nReasoning: {result['reasoning']}")
                
        except Exception as e:
            print(f"Error processing query: {e}")
    
    print("\n" + "=" * 60)
    print("Testing completed")
    print("=" * 60)

def create_sample_database(filename="sample_medical_database.csv"):
    """Create a sample medical database for testing if none exists"""
    import pandas as pd
    
    if os.path.exists(filename):
        print(f"Using existing database: {filename}")
        return filename
        
    print(f"Creating sample database: {filename}")
    
    data = [
        {
            "standard_disease_name": "Type 2 Diabetes Mellitus",
            "description": "A metabolic disorder characterized by high blood sugar, insulin resistance, and relative lack of insulin.",
            "synonyms": "adult-onset diabetes, non-insulin-dependent diabetes mellitus, NIDDM",
            "icd_code": "E11"
        },
        {
            "standard_disease_name": "Essential Hypertension",
            "description": "Persistently elevated blood pressure in the arteries without identifiable cause.",
            "synonyms": "high blood pressure, HTN",
            "icd_code": "I10"
        },
        {
            "standard_disease_name": "Myocardial Infarction",
            "description": "Death of heart muscle due to prolonged lack of oxygen supply.",
            "synonyms": "heart attack, MI, cardiac infarction",
            "icd_code": "I21"
        },
        {
            "standard_disease_name": "Rheumatoid Arthritis",
            "description": "Autoimmune disease causing chronic inflammation of the joints.",
            "synonyms": "RA, rheumatoid disease",
            "icd_code": "M05"
        },
        {
            "standard_disease_name": "Migraine",
            "description": "Recurrent moderate to severe headache with associated symptoms like nausea and sensitivity to light.",
            "synonyms": "migraine headache, sick headache, hemicrania",
            "icd_code": "G43"
        }
    ]
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    df.to_csv(filename, index=False)
    
    return filename

def main():
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("Warning: AZURE_OPENAI_API_KEY environment variable not set")
        print("Using default key in AzureOpenAIWrapper")
    
    # Create directories if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/vector_stores", exist_ok=True)
    
    # Create a sample database for testing
    database_file = "data/sample_medical_database.csv"
    create_sample_database(database_file)
    
    print("Initializing Medical Disease Name Search System...")
    
    # Initialize the system with Azure OpenAI
    system = MedicalDiseaseNameSearchSystem(
        embedding_model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        vector_db_path="data/vector_stores/sapbert_faiss",
        llm_model_name="gpt-35-turbo",  # Azure deployment name
        api_key=api_key,
        api_version="2024-02-15-preview",
        azure_endpoint="https://formaigpt.openai.azure.com"
    )
    
    # Load the sample database
    try:
        print(f"Loading data from {database_file}...")
        system.load_data(database_file)
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Continuing with existing vector store (if available)...")
    
    # Test queries
    test_queries = [
        "patient has high blood sugar and increased thirst",
        "persistent elevated blood pressure",
        "chest pain radiating to left arm with shortness of breath",
        "painful, swollen joints with morning stiffness"
    ]
    
    # Run the tests
    test_with_queries(system, test_queries)

if __name__ == "__main__":
    main()