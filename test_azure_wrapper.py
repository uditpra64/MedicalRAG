import os
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the Azure OpenAI wrapper
from src.azure_openai_wrapper import AzureOpenAIWrapper

def main():
    # Load environment variables
    load_dotenv()
    
    # Check environment
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if deployment:
        print(f"Found AZURE_OPENAI_DEPLOYMENT in environment: {deployment}")
    else:
        print("No AZURE_OPENAI_DEPLOYMENT found in environment. Using default.")
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if api_key:
        print("Found AZURE_OPENAI_API_KEY in environment.")
    else:
        print("No AZURE_OPENAI_API_KEY found in environment. Using default.")
    
    # List available Azure OpenAI models
    print("\nTesting Azure OpenAI wrapper...")
    print("-" * 60)
    
    # Create instance of the wrapper
    wrapper = AzureOpenAIWrapper()
    
    # Print configuration
    print(f"Using deployment: {wrapper.model}")
    
    # Test a simple prompt
    test_prompt = "Briefly describe what rheumatoid arthritis is."
    
    print(f"\nSending test prompt: '{test_prompt}'")
    print("-" * 60)
    
    try:
        response = wrapper.invoke(test_prompt)
        print("\nResponse:")
        print(response)
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check that your Azure OpenAI API key is correct")
        print("2. Verify that your deployment name exists in your Azure OpenAI service")
        print("3. Make sure your Azure endpoint URL is correct")
        print("4. Confirm that your API version is supported")

if __name__ == "__main__":
    main()