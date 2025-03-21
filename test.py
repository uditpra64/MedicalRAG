import os
import sys
import logging
import pandas as pd
import time
from dotenv import load_dotenv
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the system components
from src.main_application import MedicalDiseaseNameSearchSystem

def setup_test_environment():
    """Create necessary directories and sample data for testing."""
    print("\n===== Setting up test environment =====")
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/vector_stores", exist_ok=True)
    os.makedirs("data/feedback", exist_ok=True)
    os.makedirs("data/icd10", exist_ok=True)
    
    # Create sample medical database if it doesn't exist
    create_sample_database()
    
    # Create sample ICD-10 data
    create_sample_icd10_data()
    
    print("Test environment setup complete!")

def create_sample_database(filename="data/sample_medical_database.csv"):
    """Create a sample medical database for testing."""
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
        },
        {
            "standard_disease_name": "Asthma",
            "description": "Chronic lung disease characterized by inflammation and narrowing of the airways causing wheezing and shortness of breath.",
            "synonyms": "bronchial asthma, reactive airway disease",
            "icd_code": "J45"
        },
        {
            "standard_disease_name": "Community-acquired Pneumonia",
            "description": "Infection of the lungs acquired outside of healthcare settings.",
            "synonyms": "CAP, bacterial pneumonia, atypical pneumonia",
            "icd_code": "J18"
        },
        {
            "standard_disease_name": "Generalized Anxiety Disorder",
            "description": "Excessive, persistent worry and anxiety that interferes with daily activities.",
            "synonyms": "GAD, anxiety neurosis, chronic anxiety",
            "icd_code": "F41.1"
        }
    ]
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    df.to_csv(filename, index=False)
    
    print(f"Created sample database with {len(df)} entries")
    return filename

def create_sample_icd10_data(filename="data/icd10_codes.csv"):
    """Create a sample ICD-10 data file for testing."""
    if os.path.exists(filename):
        print(f"Using existing ICD-10 data: {filename}")
        return filename
        
    print(f"Creating sample ICD-10 data: {filename}")
    
    # Create simple ICD-10 data with a few common codes
    data = [
        {"code": "E11", "description": "Type 2 diabetes mellitus"},
        {"code": "I10", "description": "Essential (primary) hypertension"},
        {"code": "I21", "description": "Acute myocardial infarction"},
        {"code": "J44", "description": "Chronic obstructive pulmonary disease"},
        {"code": "M05", "description": "Rheumatoid arthritis with rheumatoid factor"},
        {"code": "G43", "description": "Migraine"},
        {"code": "F33", "description": "Major depressive disorder, recurrent"},
        {"code": "K21", "description": "Gastro-esophageal reflux disease"},
        {"code": "N18", "description": "Chronic kidney disease"},
        {"code": "C50", "description": "Malignant neoplasm of breast"},
        {"code": "J45", "description": "Asthma"},
        {"code": "J18", "description": "Pneumonia, unspecified organism"},
        {"code": "F41.1", "description": "Generalized anxiety disorder"}
    ]
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create and save dataframe
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    
    print(f"Created sample ICD-10 data with {len(df)} entries")
    return filename

def test_initialization():
    """Test system initialization with various configurations."""
    print("\n===== Testing system initialization =====")
    
    # API key from environment or use default
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("Warning: AZURE_OPENAI_API_KEY environment variable not set")
        print("Using default key in AzureOpenAIWrapper")
    
    # Test basic initialization
    print("Initializing system with default settings...")
    try:
        system = MedicalDiseaseNameSearchSystem(
            embedding_model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
            vector_db_path="data/vector_stores/sapbert_faiss",
            llm_model_name="gptest",  # Azure deployment name
            api_key=api_key,
            icd10_path="data/icd10_codes.csv"
        )
        print("✅ System initialization successful")
        return system
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        raise

def test_vector_store(system):
    """Test vector store creation and loading."""
    print("\n===== Testing vector store =====")
    
    database_file = "data/sample_medical_database.csv"
    
    # Test data loading and vector store creation
    print(f"Loading data from {database_file}...")
    try:
        system.load_data(database_file)
        print("✅ Data loaded and vector store created successfully")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Continuing with existing vector store (if available)...")
    
    # Test vector store access
    print("Testing vector store search...")
    try:
        results = system.search("diabetes", top_k=2)
        print(f"✅ Vector store search successful, found {len(results)} results")
        for i, result in enumerate(results):
            print(f"  Result {i+1}: {result['content'][:100]}...")
    except Exception as e:
        print(f"❌ Vector store search failed: {e}")

def test_hybrid_search(system):
    """Test the hybrid search functionality."""
    print("\n===== Testing hybrid search =====")
    
    # First, try to create a hybrid retriever
    try:
        hybrid_retriever = system.medical_rag.create_hybrid_retriever()
        print("✅ Hybrid retriever created successfully")
        
        # Test the hybrid retriever with a query
        print("Testing hybrid retriever with query 'high blood sugar'...")
        docs = hybrid_retriever.get_relevant_documents("high blood sugar")
        print(f"Hybrid search returned {len(docs)} documents")
        for i, doc in enumerate(docs[:2]):  # Show first 2 results
            print(f"  Result {i+1}: {doc.page_content[:100]}...")
        
    except Exception as e:
        print(f"❌ Hybrid retriever creation failed: {e}")

def test_medical_term_standardization(system, queries):
    """Test medical term standardization with various queries."""
    print("\n===== Testing medical term standardization =====")
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: '{query}'")
        print("-" * 60)
        
        try:
            # Process the query
            start_time = time.time()
            result = system.convert_medical_expression(query)
            elapsed_time = time.time() - start_time
            
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
            
            print(f"\nQuery processed in {elapsed_time:.2f} seconds")
            print("✅ Query processing successful")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")

def test_feedback_system(system):
    """Test the feedback system with a clear workflow."""
    print("\n===== Testing feedback system =====")
    
    # 1. Clear existing feedback data for a clean test
    try:
        # Reset the feedback CSV file
        import pandas as pd
        columns = [
            "timestamp", "query", "system_diagnosis", "expert_diagnosis", 
            "confidence_score", "is_correct", "expert_notes"
        ]
        feedback_df = pd.DataFrame(columns=columns)
        feedback_file = system.feedback_manager.feedback_file
        feedback_df.to_csv(feedback_file, index=False)
        
        # Reset the active learning queue
        queue = {
            "high_priority": [],
            "medium_priority": [],
            "review_complete": []
        }
        queue_file = system.feedback_manager.active_learning_file
        import json
        with open(queue_file, "w") as f:
            json.dump(queue, f, indent=2)
            
        print("✅ Feedback system reset for clean testing")
    except Exception as e:
        print(f"⚠️ Could not reset feedback system: {e}")
    
    # 2. Process several test queries to generate results
    test_queries = [
        "high blood sugar with excessive thirst",
        "constant chest pain with left arm numbness",
        "recurring headaches with light sensitivity",
        "joint pain and stiffness in the morning"
    ]
    
    print("\nProcessing test queries to generate system diagnoses...")
    results = []
    for query in test_queries:
        try:
            result = system.convert_medical_expression(query)
            results.append((query, result))
            print(f"  ✅ Processed: '{query}' → '{result['standard_diagnosis']}' (conf: {result['confidence_score']:.2f})")
        except Exception as e:
            print(f"  ❌ Error processing '{query}': {e}")
    
    # 3. Check if queries with low confidence were automatically queued for review
    print("\nChecking if low-confidence queries were automatically queued...")
    try:
        high_priority = system.feedback_manager.get_expert_review_queue("high_priority")
        medium_priority = system.feedback_manager.get_expert_review_queue("medium_priority")
        
        # Print queue contents before our manual additions
        print(f"High priority queue: {len(high_priority)} items")
        for i, item in enumerate(high_priority):
            print(f"  {i+1}. '{item['query']}' → '{item['system_diagnosis']}' (conf: {item['confidence_score']:.2f})")
            
        print(f"Medium priority queue: {len(medium_priority)} items")
        for i, item in enumerate(medium_priority):
            print(f"  {i+1}. '{item['query']}' → '{item['system_diagnosis']}' (conf: {item['confidence_score']:.2f})")
    except Exception as e:
        print(f"❌ Error checking review queues: {e}")
    
    # 4. Manually provide expert feedback for one query
    print("\nSimulating expert feedback for a query...")
    if results:
        query, result = results[0]  # Get the first query result
        try:
            system.record_expert_feedback(
                query=query,
                system_diagnosis=result["standard_diagnosis"],
                expert_diagnosis="Type 2 Diabetes Mellitus",  # "Correct" expert diagnosis
                is_correct=True,
                expert_notes="Clear symptoms of Type 2 Diabetes, correct identification."
            )
            print(f"✅ Recorded positive feedback for '{query}'")
            
            # Add a second feedback with incorrect diagnosis
            system.record_expert_feedback(
                query=results[1][0],
                system_diagnosis=results[1][1]["standard_diagnosis"],
                expert_diagnosis="Stable Angina",  # Different diagnosis
                is_correct=False,
                expert_notes="Symptoms more consistent with stable angina than MI."
            )
            print(f"✅ Recorded corrective feedback for '{results[1][0]}'")
        except Exception as e:
            print(f"❌ Error recording feedback: {e}")
    
    # 5. Check if items were removed from queue after feedback
    print("\nChecking if reviewed items were removed from queues...")
    try:
        high_priority_after = system.feedback_manager.get_expert_review_queue("high_priority")
        medium_priority_after = system.feedback_manager.get_expert_review_queue("medium_priority")
        review_complete = system.feedback_manager.active_learning_file
        
        # Try to read the review_complete list
        with open(review_complete, "r") as f:
            queue_data = json.load(f)
            completed = queue_data.get("review_complete", [])
            
        print(f"High priority queue: {len(high_priority_after)} items (was {len(high_priority)})")
        print(f"Medium priority queue: {len(medium_priority_after)} items (was {len(medium_priority)})")
        print(f"Completed reviews: {len(completed)} items")
        
        # Check if our reviewed items moved to completed
        if len(completed) > 0:
            print("✅ Confirmed items moved to completed after review")
        else:
            print("⚠️ No items found in completed list after review")
    except Exception as e:
        print(f"❌ Error checking updated queues: {e}")
    
    # 6. Test retrieval of feedback statistics
    print("\nGetting feedback statistics...")
    try:
        stats = system.feedback_manager.get_feedback_statistics()
        print("✅ Feedback statistics retrieved successfully")
        print(f"Total feedback entries: {stats['total_feedback_entries']}")
        print(f"Correct diagnoses: {stats['correct_diagnoses']}")
        print(f"Overall accuracy: {stats['overall_accuracy']:.2f}")
        
        # Check accuracy by confidence band
        if 'accuracy_by_confidence' in stats:
            print("Accuracy by confidence band:")
            for band, acc in stats['accuracy_by_confidence'].items():
                print(f"  {band}: {acc:.2f}")
    except Exception as e:
        print(f"❌ Error retrieving feedback statistics: {e}")
    
    # 7. Test improvement suggestions
    print("\nGetting improvement suggestions...")
    try:
        suggestions = system.feedback_manager.get_improvement_suggestions()
        print(f"✅ Got {len(suggestions)} improvement suggestions")
        for i, suggestion in enumerate(suggestions):
            print(f"  {i+1}. {suggestion['suggestion']} ({suggestion['count']} instances)")
    except Exception as e:
        print(f"❌ Error getting improvement suggestions: {e}")
    
    print("\nFeedback system test completed")

def test_ontology_integration(system):
    """Test the ontology integration."""
    print("\n===== Testing ontology integration =====")
    
    # Test ICD-10 lookup
    print("Testing ICD-10 lookup for 'Type 2 Diabetes Mellitus'...")
    try:
        ontology_data = system.get_medical_ontologies("Type 2 Diabetes Mellitus")
        print("✅ Ontology data retrieved successfully")
        
        # Display ICD-10 results
        if ontology_data["icd10"]:
            print("ICD-10 Codes:")
            for code in ontology_data["icd10"]:
                print(f"  {code['code']}: {code['description']}")
        else:
            print("No ICD-10 codes found")
            
    except Exception as e:
        print(f"❌ Error retrieving ontology data: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test the Medical Disease Name Search System")
    parser.add_argument("--section", type=str, help="Run a specific test section (init, vector, hybrid, term, feedback, ontology)")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--setup", action="store_true", help="Only set up the test environment")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Set up test environment
    if args.setup or args.all or args.section:
        setup_test_environment()
    
    if args.setup:
        # Only set up the environment, don't run tests
        return
    
    # System initialization (needed for all tests)
    system = None
    if args.all or args.section in ['init', 'vector', 'hybrid', 'term', 'feedback', 'ontology']:
        system = test_initialization()
    
    # Run specific test section or all tests
    if args.section == 'vector' or args.all:
        test_vector_store(system)
    
    if args.section == 'hybrid' or args.all:
        test_hybrid_search(system)
    
    if args.section == 'term' or args.all:
        # Test queries
        test_queries = [
            "patient has high blood sugar and increased thirst",
            "persistent elevated blood pressure",
            "chest pain radiating to left arm with shortness of breath",
            "painful, swollen joints with morning stiffness",
            "wheezing and shortness of breath triggered by allergens",
            "fever, cough, and chest pain with difficult breathing",
            "persistent worry and anxiety interfering with daily activities"
        ]
        test_medical_term_standardization(system, test_queries)
    
    if args.section == 'feedback' or args.all:
        test_feedback_system(system)
    
    if args.section == 'ontology' or args.all:
        test_ontology_integration(system)
    
    print("\n===== Testing completed =====")

if __name__ == "__main__":
    main()