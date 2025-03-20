import os
import argparse
import logging
import yaml
from dotenv import load_dotenv

# Set up logging
from utils.logging_config import setup_logging
logger = setup_logging()

def load_config(config_path="config/app_config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        # Return default configuration
        return {
            "embedding_model": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
            "vector_store_path": "./data/vector_stores/sapbert_faiss",
            "llm": {
                "provider": "azure",
                "model": "gpt-35-turbo",
                "api_version": "2024-02-15-preview",
                "azure_endpoint": "https://formaigpt.openai.azure.com"
            },
            "confidence_threshold": 0.7
        }

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Medical Disease Name Search System")
    
    # Add command-line arguments
    parser.add_argument("--config", type=str, default="config/app_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--ui", action="store_true",
                        help="Launch the Streamlit UI")
    parser.add_argument("--batch", type=str,
                        help="Process batch file and save results")
    parser.add_argument("--query", type=str,
                        help="Process a single query")
    parser.add_argument("--model", type=str,
                        help="Override the embedding model")
    parser.add_argument("--provider", type=str, choices=["azure", "openai"],
                        help="LLM provider (azure or openai)")
    parser.add_argument("--azure-model", type=str,
                        help="Azure OpenAI model deployment name")
    parser.add_argument("--azure-endpoint", type=str,
                        help="Azure OpenAI endpoint URL")
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    # Load environment variables
    load_dotenv()
    
    # Check for API keys
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command-line arguments
    if args.model:
        config["embedding_model"] = args.model
    
    if args.provider:
        config["llm"]["provider"] = args.provider
        
    if args.azure_model:
        config["llm"]["model"] = args.azure_model
        
    if args.azure_endpoint:
        config["llm"]["azure_endpoint"] = args.azure_endpoint
    
    # Import here to avoid circular imports
    from src.main_application import MedicalDiseaseNameSearchSystem
    
    # Determine which API key to use
    if config["llm"]["provider"] == "azure":
        if not azure_api_key:
            logger.warning("AZURE_OPENAI_API_KEY environment variable not set")
            print("Warning: AZURE_OPENAI_API_KEY environment variable not set")
            print("Using default API key from AzureOpenAIWrapper")
        api_key = azure_api_key
        api_provider = "Azure OpenAI"
    else:
        if not openai_api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            print("Error: OPENAI_API_KEY environment variable not set")
            print("Please set your OpenAI API key in the .env file or environment")
            return
        api_key = openai_api_key
        api_provider = "OpenAI"
    
    logger.info(f"Using {api_provider} with model {config['llm']['model']}")
    print(f"Using {api_provider} with model {config['llm']['model']}")
    
    # Initialize the system
    logger.info("Initializing Medical Disease Name Search System")
    system = MedicalDiseaseNameSearchSystem(
        embedding_model_name=config["embedding_model"],
        vector_db_path=config["vector_store_path"],
        llm_model_name=config["llm"]["model"],
        api_key=api_key,
        api_version=config["llm"].get("api_version", "2024-02-15-preview"),
        azure_endpoint=config["llm"].get("azure_endpoint", "https://formaigpt.openai.azure.com")
    )
    
    # Process a single query if provided
    if args.query:
        logger.info(f"Processing query: {args.query}")
        result = system.convert_medical_expression(args.query)
        
        # Print results
        print("\nQuery Results:")
        print(f"Standard Diagnosis: {result['standard_diagnosis']}")
        print(f"Confidence: {result['confidence_score']:.2f}")
        
        if result.get('alternative_diagnoses'):
            print("Alternative Diagnoses:")
            for alt in result['alternative_diagnoses']:
                print(f"  - {alt}")
        
        if result.get('needs_human_review'):
            print("⚠️ This case needs human review")
        
        if result.get('reasoning'):
            print(f"Reasoning: {result['reasoning']}")
    
    # Process batch file if provided
    elif args.batch:
        from scripts.batch_standardize import process_batch_file
        
        logger.info(f"Processing batch file: {args.batch}")
        output_file = f"output_{os.path.basename(args.batch)}"
        process_batch_file(system, args.batch, output_file)
        print(f"Batch processing complete. Results saved to {output_file}")
    
    # Launch UI if requested
    elif args.ui:
        try:
            logger.info("Launching Streamlit UI")
            print("Launching Streamlit UI...")
            
            # Launch Streamlit UI in a subprocess
            import subprocess
            subprocess.run(["streamlit", "run", "ui/streamlit_app.py", 
                           "--", f"--config={args.config}"])
        except Exception as e:
            logger.error(f"Error launching UI: {e}")
            print(f"Error launching UI: {e}")
    
    # If no specific action, print help
    else:
        print("Medical Disease Name Search System")
        print("Usage:")
        print("  --query TEXT       Process a single query")
        print("  --batch FILE       Process a batch file")
        print("  --ui               Launch the Streamlit UI")
        print("  --model NAME       Override the embedding model")
        print("  --provider NAME    LLM provider (azure or openai)")
        print("  --azure-model NAME Azure OpenAI model deployment name")
        print("  --config FILE      Specify configuration file")
        print("\nFor more information, use --help")

if __name__ == "__main__":
    main()