import os
import sys
import argparse
import pandas as pd
import time
from tqdm import tqdm
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions and MedicalDiseaseNameSearchSystem
from utils.logging_config import setup_logging
from src.main_application import MedicalDiseaseNameSearchSystem

# Set up logging
logger = setup_logging()

def process_batch_file(
    system, 
    input_file, 
    output_file,
    query_column="query",
    additional_columns=None,
    batch_size=None,
    show_progress=True
):
    """
    Process a batch of queries from a CSV file.
    
    Args:
        system: MedicalDiseaseNameSearchSystem instance
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        query_column: Name of the column containing queries
        additional_columns: List of additional columns to include in the output
        batch_size: Number of queries to process in each batch (None for all)
        show_progress: Whether to show a progress bar
    
    Returns:
        dict: Processing statistics
    """
    logger.info(f"Processing batch file: {input_file}")
    
    try:
        # Read input CSV
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} queries from {input_file}")
        
        # Validate input
        if query_column not in df.columns:
            error_msg = f"Column '{query_column}' not found in input file"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get additional columns if specified
        if additional_columns is None:
            additional_columns = []
        
        # Create output dataframe
        results = []
        errors = 0
        start_time = time.time()
        
        # Process queries
        iterator = tqdm(df.iterrows(), total=len(df)) if show_progress else df.iterrows()
        
        for i, row in iterator:
            try:
                # Get query
                query = row[query_column]
                
                # Process query
                result = system.convert_medical_expression(query)
                
                # Prepare result dictionary
                result_dict = {
                    'query': query,
                    'standard_diagnosis': result['standard_diagnosis'],
                    'confidence_score': result['confidence_score'],
                    'needs_human_review': result.get('needs_human_review', False)
                }
                
                # Add alternatives if available
                if result.get('alternative_diagnoses'):
                    result_dict['alternative_diagnoses'] = ', '.join(result['alternative_diagnoses'])
                
                # Add reasoning if available
                if result.get('reasoning'):
                    result_dict['reasoning'] = result['reasoning']
                
                # Add additional columns
                for col in additional_columns:
                    if col in row:
                        result_dict[col] = row[col]
                
                # Add result to list
                results.append(result_dict)
                
            except Exception as e:
                logger.error(f"Error processing query {i}: {e}")
                errors += 1
                
                # Add error result
                results.append({
                    'query': row[query_column],
                    'standard_diagnosis': 'ERROR',
                    'confidence_score': 0.0,
                    'needs_human_review': True,
                    'error': str(e)
                })
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(output_file, index=False)
        
        # Calculate statistics
        total_time = time.time() - start_time
        avg_time = total_time / len(df) if len(df) > 0 else 0
        
        stats = {
            'total_queries': len(df),
            'successful_queries': len(df) - errors,
            'failed_queries': errors,
            'total_time': total_time,
            'avg_time_per_query': avg_time
        }
        
        logger.info(f"Batch processing completed. Results saved to {output_file}")
        logger.info(f"Statistics: {stats}")
        
        return stats
    
    except Exception as e:
        logger.error(f"Error processing batch file: {e}")
        raise

def main():
    """Main entry point for batch processing script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Batch process medical expressions")
    
    parser.add_argument("input_file", type=str, help="Path to input CSV file")
    parser.add_argument("--output_file", type=str, help="Path to output CSV file")
    parser.add_argument("--query_column", type=str, default="query", 
                      help="Name of the column containing queries")
    parser.add_argument("--model", type=str, 
                      default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                      help="Medical embedding model to use")
    parser.add_argument("--vector_store", type=str, 
                      default="./data/vector_stores/sapbert_faiss",
                      help="Path to vector store")
    parser.add_argument("--llm_model", type=str, default="gpt-3.5-turbo",
                      help="LLM model to use")
    parser.add_argument("--batch_size", type=int, default=None,
                      help="Number of queries to process in each batch")
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output_file is None:
        base_name = os.path.basename(args.input_file)
        name_parts = os.path.splitext(base_name)
        args.output_file = f"{name_parts[0]}_results{name_parts[1]}"
    
    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Initialize system
    try:
        system = MedicalDiseaseNameSearchSystem(
            embedding_model_name=args.model,
            vector_db_path=args.vector_store,
            llm_model_name=args.llm_model
        )
        
        # Process batch
        stats = process_batch_file(
            system=system,
            input_file=args.input_file,
            output_file=args.output_file,
            query_column=args.query_column,
            batch_size=args.batch_size
        )
        
        # Print statistics
        print("\nBatch Processing Statistics")
        print(f"Total queries: {stats['total_queries']}")
        print(f"Successful queries: {stats['successful_queries']}")
        print(f"Failed queries: {stats['failed_queries']}")
        print(f"Total processing time: {stats['total_time']:.2f} seconds")
        print(f"Average time per query: {stats['avg_time_per_query']:.2f} seconds")
        print(f"\nResults saved to: {args.output_file}")
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()