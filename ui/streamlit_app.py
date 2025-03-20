import os
import streamlit as st
import pandas as pd
import time
import yaml
import argparse
from typing import Dict, Any
import sys

# Add parent directory to path so we can import modules
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Custom imports
from src.main_application import MedicalDiseaseNameSearchSystem

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Streamlit UI for Medical Disease Name Search System")
    parser.add_argument("--config", type=str, default="../config/app_config.yaml",
                        help="Path to configuration file")
    # Remove streamlit's own arguments
    streamlit_args = ["--server.port", "--server.address", "--browser.serverAddress",
                      "--browser.gatherUsageStats", "--logger.level", "--logger.messageFormat"]
    parsed_args, _ = parser.parse_known_args()
    return parsed_args

# Function to load configuration
def load_config(config_path):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        # Return default configuration
        return {
            "embedding_model": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
            "vector_store_path": "../data/vector_stores/sapbert_faiss",
            "llm_model": "gpt-3.5-turbo",
            "confidence_threshold": 0.7
        }

# Main UI function
def create_ui():
    """Create a Streamlit UI for the Medical Disease Name Search System."""
    # Page configuration
    st.set_page_config(
        page_title="Medical Disease Name Search System",
        page_icon="🏥",
        layout="wide"
    )
    
    # Title and description
    st.title("🏥 Medical Disease Name Search System")
    st.subheader("Convert non-standard medical expressions to standardized disease names")
    
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API key input
        api_key = st.text_input("API Key (for LLM)", type="password")
        
        # Embedding model selection
        embedding_model = st.selectbox(
            "Medical Embedding Model",
            options=[
                "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                "emilyalsentzer/Bio_ClinicalBERT",
                "allenai/scibert_scivocab_uncased"
            ],
            index=0
        )
        
        # LLM model selection
        llm_model = st.selectbox(
            "LLM Model",
            options=[
                "gpt-3.5-turbo",
                "gpt-4"
            ],
            index=0
        )
        
        # Vector DB path
        vector_db_path = st.text_input(
            "Vector Database Path",
            value=config["vector_store_path"]
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=config["confidence_threshold"],
            step=0.05
        )
        
        # Initialize button
        if st.button("Initialize System"):
            if not api_key:
                st.error("Please enter an API key")
            else:
                with st.spinner("Initializing system..."):
                    # Create or load system
                    try:
                        # Store API key in environment variable
                        os.environ["OPENAI_API_KEY"] = api_key
                        
                        # Create the system
                        system = MedicalDiseaseNameSearchSystem(
                            embedding_model_name=embedding_model,
                            vector_db_path=vector_db_path,
                            llm_model_name=llm_model,
                            confidence_threshold=confidence_threshold
                        )
                        
                        # Store in session state
                        st.session_state.system = system
                        st.success("System initialized successfully!")
                    except Exception as e:
                        st.error(f"Error initializing system: {e}")
        
        # Data upload section
        st.header("Data Management")
        
        uploaded_file = st.file_uploader(
            "Upload Medical Data (CSV)",
            type=["csv"]
        )
        
        if uploaded_file is not None:
            if st.button("Process Data"):
                if 'system' not in st.session_state:
                    st.error("Please initialize the system first")
                else:
                    with st.spinner("Processing data..."):
                        # Save uploaded file temporarily
                        temp_path = "temp_medical_data.csv"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                            
                        # Load data into system
                        try:
                            st.session_state.system.load_data(temp_path)
                            st.success("Data processed successfully!")
                        except Exception as e:
                            st.error(f"Error processing data: {e}")
                        
                        # Clean up
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
    
    # Main content area - Query input
    input_col, results_col = st.columns([1, 1])
    
    with input_col:
        st.header("Query Input")
        
        # Single query mode
        query = st.text_area(
            "Enter non-standard medical expression",
            height=100,
            placeholder="e.g., high blood sugar, chest pain with shortness of breath"
        )
        
        process_button = st.button("Process Query")
        
        # Batch mode
        st.subheader("Batch Processing")
        
        batch_file = st.file_uploader(
            "Upload Queries (CSV with 'query' column)",
            type=["csv"]
        )
        
        batch_button = st.button("Process Batch")
    
    with results_col:
        st.header("Results")
        
        # Handle single query processing
        if process_button and query:
            if 'system' not in st.session_state:
                st.error("Please initialize the system first")
            else:
                with st.spinner("Processing query..."):
                    try:
                        start_time = time.time()
                        result = st.session_state.system.convert_medical_expression(
                            query,
                            confidence_threshold=confidence_threshold
                        )
                        elapsed_time = time.time() - start_time
                        
                        # Display results
                        st.subheader("Standardized Result")
                        
                        # Create two columns for results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Standard Disease Name", result["standard_diagnosis"])
                            
                        with col2:
                            confidence = result["confidence_score"]
                            st.metric("Confidence", f"{confidence:.2f}")
                            
                            # Display confidence with color
                            if confidence >= 0.9:
                                st.success("High confidence")
                            elif confidence >= 0.7:
                                st.info("Medium confidence")
                            else:
                                st.warning("Low confidence")
                        
                        # Show alternatives if available
                        if result.get("alternative_diagnoses"):
                            st.subheader("Alternative Diagnoses")
                            for alt in result["alternative_diagnoses"]:
                                st.write(f"- {alt}")
                        
                        # Show reasoning if available
                        if result.get("reasoning"):
                            st.subheader("Reasoning")
                            st.write(result["reasoning"])
                        
                        # Show processing time
                        st.caption(f"Processed in {elapsed_time:.2f} seconds")
                        
                        # Human review indicator
                        if result.get("needs_human_review", False):
                            st.warning("⚠️ This case requires human expert review")
                            
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
        
        # Handle batch processing
        if batch_button and batch_file:
            if 'system' not in st.session_state:
                st.error("Please initialize the system first")
            else:
                with st.spinner("Processing batch queries..."):
                    try:
                        # Read batch file
                        batch_df = pd.read_csv(batch_file)
                        
                        if 'query' not in batch_df.columns:
                            st.error("CSV must contain a 'query' column")
                        else:
                            # Process each query
                            results = []
                            
                            progress_bar = st.progress(0)
                            
                            for i, row in batch_df.iterrows():
                                query = row['query']
                                result = st.session_state.system.convert_medical_expression(
                                    query,
                                    confidence_threshold=confidence_threshold
                                )
                                
                                results.append({
                                    'query': query,
                                    'standard_diagnosis': result['standard_diagnosis'],
                                    'confidence_score': result['confidence_score'],
                                    'needs_human_review': result.get('needs_human_review', False)
                                })
                                
                                # Update progress
                                progress_bar.progress((i + 1) / len(batch_df))
                            
                            # Create results dataframe
                            results_df = pd.DataFrame(results)
                            
                            # Display results
                            st.subheader("Batch Results")
                            st.dataframe(results_df)
                            
                            # Download link
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="medical_standardization_results.csv",
                                mime="text/csv",
                            )
                            
                    except Exception as e:
                        st.error(f"Error processing batch: {e}")
    
    # System information
    st.header("System Information")
    
    if 'system' in st.session_state:
        st.json({
            "embedding_model": embedding_model,
            "llm_model": llm_model,
            "vector_db_path": vector_db_path,
            "confidence_threshold": confidence_threshold,
            "system_status": "Initialized"
        })
    else:
        st.info("System not initialized")

if __name__ == "__main__":
    create_ui()