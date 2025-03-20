import os
import pandas as pd
from langchain_openai import ChatOpenAI
from src.medical_rag_agent import MedicalRAG_Agent

# Function to create a sample medical database
def create_sample_database(filename="sample_medical_database.csv"):
    """Create a sample medical database for testing."""
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
        },
        {
            "standard_disease_name": "Asthma",
            "description": "Chronic lung disease characterized by inflammation and narrowing of the airways.",
            "synonyms": "bronchial asthma, reactive airway disease",
            "icd_code": "J45"
        },
        {
            "standard_disease_name": "Pneumonia",
            "description": "Infection that inflames the air sacs in one or both lungs.",
            "synonyms": "pneumonitis, bronchopneumonia",
            "icd_code": "J18"
        },
        {
            "standard_disease_name": "Irritable Bowel Syndrome",
            "description": "Chronic disorder affecting the large intestine with symptoms of abdominal pain, bloating, and altered bowel habits.",
            "synonyms": "IBS, spastic colon",
            "icd_code": "K58"
        },
        {
            "standard_disease_name": "Gastroesophageal Reflux Disease",
            "description": "Chronic digestive disease where stomach acid flows back into the esophagus.",
            "synonyms": "GERD, acid reflux, heartburn",
            "icd_code": "K21"
        },
        {
            "standard_disease_name": "COVID-19",
            "description": "Infectious disease caused by the SARS-CoV-2 virus primarily affecting the respiratory system.",
            "synonyms": "coronavirus disease 2019, SARS-CoV-2 infection",
            "icd_code": "U07.1"
        }
    ]
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Sample medical database created: {filename}")
    return filename

def main():
    print("Medical RAG System Test")
    print("=" * 30)
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set")
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Create sample database if it doesn't exist
    database_file = "sample_medical_database.csv"
    if not os.path.exists(database_file):
        database_file = create_sample_database(database_file)
    
    # Initialize LLM
    print("\nInitializing LLM...")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Initialize Medical RAG with various medical embedding models
    embedding_models = [
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",  # SapBERT
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",  # BiomedBERT
        "emilyalsentzer/Bio_ClinicalBERT"  # ClinicalBERT
        # Note: SciBERT might need more resources, uncomment if you have enough RAM
        # "allenai/scibert_scivocab_uncased"  # SciBERT
    ]
    
    # Test queries
    test_queries = [
        "patient has high blood sugar and increased thirst",
        "persistent elevated blood pressure",
        "chest pain radiating to left arm with shortness of breath",
        "painful, swollen joints with morning stiffness",
        "severe headache with visual disturbances and nausea"
    ]
    
    # Dictionary to store results for comparison
    model_results = {}
    
    # Test each embedding model
    for model_name in embedding_models:
        print(f"\nTesting with embedding model: {model_name}")
        
        # Create a unique vector store path for each model
        vector_store_path = f"./medical_faiss_{model_name.split('/')[-1]}"
        
        # Initialize Medical RAG
        medical_rag = MedicalRAG_Agent(
            llm=llm,
            embedding_model_name=model_name,
            vector_store_path=vector_store_path
        )
        
        # Load data and create vector store if it doesn't exist
        if not os.path.exists(vector_store_path):
            print(f"Creating new vector store at {vector_store_path}")
            documents = medical_rag.load_medical_data([database_file])
            medical_rag.create_vector_store(documents)
        else:
            print(f"Loading existing vector store from {vector_store_path}")
            medical_rag.load_vector_store()
        
        # Store results for this model
        model_results[model_name] = {}
        
        # Test queries
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            # Get standardized result
            result = medical_rag.standardize_term(query)
            parsed_result = medical_rag.parse_standardized_result(result)
            
            # Store result
            model_results[model_name][query] = parsed_result
            
            # Display result
            print(f"Standard diagnosis: {parsed_result['standard_diagnosis']}")
            print(f"Confidence: {parsed_result['confidence_score']}")
            
            if parsed_result['alternative_diagnoses']:
                print("Alternative diagnoses:")
                for alt in parsed_result['alternative_diagnoses']:
                    print(f"  - {alt}")
            
            if parsed_result['needs_human_review']:
                print("⚠️ This case needs human review")
            
            if parsed_result['reasoning']:
                print(f"Reasoning: {parsed_result['reasoning']}")
    
    # Compare results across models
    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        
        for model_name in model_results.keys():
            result = model_results[model_name][query]
            short_model_name = model_name.split('/')[-1]
            
            print(f"{short_model_name:30} | {result['standard_diagnosis']:30} | " 
                  f"Confidence: {result['confidence_score']:.2f}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()