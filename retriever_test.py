import os
import sys
from dotenv import load_dotenv

# Set up paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the system
from src.main_application import MedicalDiseaseNameSearchSystem

def test_retriever():
    """Test the retriever functionality."""
    print("\n===== Testing Retriever Functionality =====")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize the system
    try:
        system = MedicalDiseaseNameSearchSystem(
            embedding_model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
            vector_db_path="./data/vector_stores/sapbert_faiss",
            llm_model_name="gpt-35-turbo"  # Azure deployment name
        )
        print("✅ System initialized successfully")
        
        # Test vector store or create if needed
        if not os.path.exists("./data/vector_stores/sapbert_faiss"):
            print("Creating vector store...")
            system.load_data("data/sample_medical_database.csv")
        
        # Test a simple query to verify the retriever works
        print("\nTesting retriever with query: 'diabetes'")
        documents = system.search("diabetes", top_k=2)
        
        print(f"Retrieved {len(documents)} documents")
        for i, doc in enumerate(documents):
            print(f"\nDocument {i+1}:")
            print(f"Content: {doc['content'][:100]}...")
            
        print("\nRetriever test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error during retriever test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_retriever()