import os
import streamlit as st
import pandas as pd
import time
import yaml
import argparse
from typing import Dict, Any
import sys
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
# Add parent directory to path so we can import modules
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Apply PyTorch workaround for Streamlit
# This prevents Streamlit from watching PyTorch classes which causes the runtime error
import torch, torchvision, torchaudio
# Save original __getattr__ function
original_getattr = torch._classes.__getattr__

def safe_getattr(self, attr):
    """A patched getattr function that handles special attributes safely"""
    if attr in ('__path__', '__file__', '__loader__', '__package__', '__spec__'):
        # Return an empty list for __path__ to prevent Streamlit from watching this module
        if attr == '__path__':
            class EmptyPath:
                _path = []
            return EmptyPath()
        # Return None for other special attributes
        return None
    # Call the original function for normal attributes
    return original_getattr(self, attr)

# Apply the patch
torch._classes.__getattr__ = lambda attr: safe_getattr(torch._classes, attr)

# Now it's safe to import our custom modules
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
            "llm": {
                "provider": "azure",
                "model": "gpt-35-turbo",
                "api_version": "2024-02-15-preview",
                "azure_endpoint": "https://formaigpt.openai.azure.com"
            },
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
    
    # Main tabs for different sections
    tabs = st.tabs(["Query", "Batch Processing", "Expert Review", "Analytics", "Settings"])

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # LLM Provider selection
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["azure", "openai"],
            index=0 if config["llm"].get("provider", "azure") == "azure" else 1
        )
        
        # API key input based on provider
        if llm_provider == "azure":
            api_key = st.text_input("Azure OpenAI API Key", type="password", 
                                   help="Leave blank to use default key from wrapper")
            api_key_env = "AZURE_OPENAI_API_KEY"
            
            # Azure-specific settings
            azure_model = st.text_input(
                "Azure OpenAI Model Deployment Name",
                value=config["llm"].get("model", "gpt-35-turbo")
            )
            
            azure_endpoint = st.text_input(
                "Azure OpenAI Endpoint",
                value=config["llm"].get("azure_endpoint", "https://formaigpt.openai.azure.com")
            )
            
            azure_api_version = st.text_input(
                "Azure OpenAI API Version",
                value=config["llm"].get("api_version", "2024-02-15-preview")
            )
        else:
            api_key = st.text_input("OpenAI API Key", type="password")
            api_key_env = "OPENAI_API_KEY"
            
            # OpenAI model selection
            openai_model = st.selectbox(
                "OpenAI Model",
                options=[
                    "gpt-3.5-turbo",
                    "gpt-4"
                ],
                index=0
            )
        
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
            if llm_provider == "openai" and not api_key:
                st.error("Please enter an OpenAI API key")
            else:
                with st.spinner("Initializing system..."):
                    # Store API key in environment variable
                    if api_key:
                        os.environ[api_key_env] = api_key
                    
                    # Create the system
                    try:
                        # Configuration for system initialization
                        system_config = {
                            "embedding_model_name": embedding_model,
                            "vector_db_path": vector_db_path,
                            "api_key": api_key or None
                        }
                        
                        # Add provider-specific parameters
                        if llm_provider == "azure":
                            system_config.update({
                                "llm_model_name": azure_model,
                                "api_version": azure_api_version,
                                "azure_endpoint": azure_endpoint
                            })
                        else:
                            system_config["llm_model_name"] = openai_model
                        
                        # Initialize the system
                        system = MedicalDiseaseNameSearchSystem(**system_config)
                        
                        # Store in session state
                        st.session_state.system = system
                        st.session_state.llm_provider = llm_provider
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
    
    # Tab 1: Query Input (Main functionality)
    with tabs[0]:
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
        
        with results_col:
            st.header("Results")
            
            # Handle query processing (existing code)
            if process_button and query:
                # ... existing code for processing query ...
                
                # Add ontology information if available
                if 'system' in st.session_state and hasattr(st.session_state.system, 'get_medical_ontologies'):
                    try:
                        ontology_data = st.session_state.system.get_medical_ontologies(result["standard_diagnosis"])
                        
                        # Display ontology data in expander
                        with st.expander("Medical Ontology Information"):
                            # ICD-10 codes
                            if ontology_data["icd10"]:
                                st.subheader("ICD-10 Codes")
                                for code in ontology_data["icd10"]:
                                    st.write(f"**{code['code']}**: {code['description']}")
                            else:
                                st.info("No ICD-10 codes found")
                            
                            # SNOMED CT codes
                            if ontology_data["snomed_ct"]:
                                st.subheader("SNOMED CT Codes")
                                for code in ontology_data["snomed_ct"]:
                                    st.write(f"**{code['concept_id']}**: {code['term']}")
                            else:
                                st.info("No SNOMED CT codes found")
                    except Exception as e:
                        st.warning(f"Could not load ontology data: {e}")
                
                # Add expert feedback form
                with st.expander("Provide Expert Feedback"):
                    st.write("Help improve the system by providing expert feedback:")
                    
                    # Correct diagnosis field
                    expert_diagnosis = st.text_input(
                        "Correct diagnosis (if different from system diagnosis)",
                        value=result["standard_diagnosis"]
                    )
                    
                    # Is correct checkbox
                    is_correct = st.checkbox(
                        "System diagnosis is correct",
                        value=expert_diagnosis == result["standard_diagnosis"]
                    )
                    
                    # Expert notes
                    expert_notes = st.text_area(
                        "Additional notes or comments"
                    )
                    
                    # Submit button
                    if st.button("Submit Feedback"):
                        if 'system' in st.session_state and hasattr(st.session_state.system, 'record_expert_feedback'):
                            try:
                                st.session_state.system.record_expert_feedback(
                                    query=query,
                                    system_diagnosis=result["standard_diagnosis"],
                                    expert_diagnosis=expert_diagnosis,
                                    is_correct=is_correct,
                                    expert_notes=expert_notes
                                )
                                st.success("Feedback submitted successfully!")
                            except Exception as e:
                                st.error(f"Error submitting feedback: {e}")
                        else:
                            st.error("System not initialized properly for feedback")
    
    # Tab 2: Batch Processing
    with tabs[1]:
        st.header("Batch Processing")
        
        batch_file = st.file_uploader(
            "Upload Queries (CSV with 'query' column)",
            type=["csv"],
            key="batch_uploader"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider(
                "Confidence Threshold for Batch",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05
            )
        
        with col2:
            auto_queue_review = st.checkbox(
                "Automatically queue low confidence results for expert review",
                value=True
            )
        
        batch_button = st.button("Process Batch", key="batch_process_button")
        
        # Batch processing
        if batch_button and batch_file:
            # ... existing batch processing code ...
            
            # Add option to queue all low confidence items for review
            if auto_queue_review and 'system' in st.session_state:
                try:
                    low_confidence = results_df[results_df["confidence_score"] < confidence_threshold]
                    
                    if not low_confidence.empty:
                        st.info(f"Queueing {len(low_confidence)} items for expert review")
                        
                        for _, row in low_confidence.iterrows():
                            st.session_state.system.feedback_manager.queue_for_expert_review(
                                query=row["query"],
                                system_diagnosis=row["standard_diagnosis"],
                                confidence_score=row["confidence_score"]
                            )
                except Exception as e:
                    st.warning(f"Could not queue items for review: {e}")
    
    # Tab 3: Expert Review Queue
    with tabs[2]:
        st.header("Expert Review Queue")
        
        if 'system' not in st.session_state:
            st.warning("Please initialize the system to access the expert review queue")
        else:
            # Refresh button
            if st.button("Refresh Queue"):
                st.experimental_rerun()
            
            try:
                # Get review queue
                high_priority = st.session_state.system.feedback_manager.get_expert_review_queue("high_priority")
                medium_priority = st.session_state.system.feedback_manager.get_expert_review_queue("medium_priority")
                
                # Display high priority queue
                st.subheader(f"High Priority Cases ({len(high_priority)})")
                
                if high_priority:
                    for i, case in enumerate(high_priority):
                        with st.expander(f"Case {i+1}: {case['query'][:50]}..."):
                            st.write(f"**Query:** {case['query']}")
                            st.write(f"**System Diagnosis:** {case['system_diagnosis']}")
                            st.write(f"**Confidence:** {case['confidence_score']:.2f}")
                            
                            if case.get('alternative_diagnoses'):
                                st.write("**Alternative Diagnoses:**")
                                for alt in case['alternative_diagnoses']:
                                    st.write(f"- {alt}")
                            
                            # Feedback form
                            st.divider()
                            st.write("### Expert Feedback")
                            
                            expert_diagnosis = st.text_input(
                                "Correct diagnosis",
                                key=f"expert_diag_{i}_high"
                            )
                            
                            is_correct = st.checkbox(
                                "System diagnosis is correct",
                                key=f"is_correct_{i}_high"
                            )
                            
                            expert_notes = st.text_area(
                                "Notes",
                                key=f"notes_{i}_high"
                            )
                            
                            if st.button("Submit", key=f"submit_{i}_high"):
                                st.session_state.system.record_expert_feedback(
                                    query=case['query'],
                                    system_diagnosis=case['system_diagnosis'],
                                    expert_diagnosis=expert_diagnosis if not is_correct else case['system_diagnosis'],
                                    is_correct=is_correct,
                                    expert_notes=expert_notes
                                )
                                st.success("Feedback submitted!")
                                st.experimental_rerun()
                else:
                    st.info("No high priority cases in queue")
                
                # Display medium priority queue (collapsed by default)
                with st.expander(f"Medium Priority Cases ({len(medium_priority)})"):
                    if medium_priority:
                        for i, case in enumerate(medium_priority):
                            with st.expander(f"Case {i+1}: {case['query'][:50]}..."):
                                # Similar feedback form as high priority
                                # ... (similar code as above) ...
                                pass
                    else:
                        st.info("No medium priority cases in queue")
                        
            except Exception as e:
                st.error(f"Error accessing review queue: {e}")
    
    # Tab 4: Analytics Dashboard
    with tabs[3]:
        st.header("System Analytics")
        
        if 'system' not in st.session_state:
            st.warning("Please initialize the system to view analytics")
        else:
            try:
                # Get feedback statistics
                stats = st.session_state.system.feedback_manager.get_feedback_statistics()
                suggestions = st.session_state.system.feedback_manager.get_improvement_suggestions()
                
                # Create dashboard
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Accuracy Metrics")
                    
                    # Overall accuracy
                    st.metric(
                        "Overall Accuracy",
                        f"{stats['overall_accuracy']:.1%}",
                        help="Percentage of correct diagnoses based on expert feedback"
                    )
                    
                    # Create accuracy by confidence chart
                    confidence_data = {
                        "Confidence Band": ["Low", "Medium", "High"],
                        "Accuracy": [
                            stats['accuracy_by_confidence']['low'],
                            stats['accuracy_by_confidence']['medium'],
                            stats['accuracy_by_confidence']['high']
                        ]
                    }
                    
                    fig = px.bar(
                        confidence_data,
                        x="Confidence Band",
                        y="Accuracy",
                        title="Accuracy by Confidence Band",
                        color="Confidence Band",
                        color_discrete_map={
                            "Low": "yellow",
                            "Medium": "blue",
                            "High": "green"
                        }
                    )
                    
                    fig.update_layout(yaxis_tickformat=".0%")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("System Usage")
                    
                    # Create usage metrics
                    total_entries = stats['total_feedback_entries']
                    total_correct = stats['correct_diagnoses']
                    total_incorrect = total_entries - total_correct
                    
                    # Pie chart
                    fig = px.pie(
                        values=[total_correct, total_incorrect],
                        names=["Correct", "Incorrect"],
                        title="Diagnosis Accuracy",
                        color_discrete_sequence=["green", "red"]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Improvement suggestions
                st.subheader("Improvement Suggestions")
                
                if suggestions:
                    for i, suggestion in enumerate(suggestions):
                        with st.expander(f"{i+1}. {suggestion['suggestion']} ({suggestion['count']} instances)"):
                            st.write("**System Diagnosis:** ", suggestion['system_diagnosis'])
                            st.write("**Expert Diagnoses:** ", ", ".join(suggestion['expert_diagnoses']))
                            st.write("**Example Queries:**")
                            for query in suggestion['example_queries']:
                                st.write(f"- {query}")
                else:
                    st.info("No improvement suggestions available yet. This requires feedback data.")
                
            except Exception as e:
                st.error(f"Error generating analytics: {e}")
    
    # Tab 5: Settings
    with tabs[4]:
        st.header("System Settings")
        
        # Configuration settings
        st.subheader("Search Configuration")
        
        # Hybrid search weights
        st.write("Hybrid Search Weights")
        vector_weight = st.slider("Vector Search Weight", 0.0, 1.0, 0.6, 0.1)
        keyword_weight = st.slider("Keyword Search Weight", 0.0, 1.0, 0.3, 0.1)
        fuzzy_weight = st.slider("Fuzzy Matching Weight", 0.0, 1.0, 0.1, 0.1)
        
        # Normalize weights
        total_weight = vector_weight + keyword_weight + fuzzy_weight
        if total_weight > 0:
            vector_weight /= total_weight
            keyword_weight /= total_weight
            fuzzy_weight /= total_weight
        
        # Display normalized weights
        st.info(f"Normalized weights: Vector ({vector_weight:.2f}), Keyword ({keyword_weight:.2f}), Fuzzy ({fuzzy_weight:.2f})")
        
        # Ontology settings
        st.subheader("Medical Ontology Settings")
        
        icd10_path = st.text_input(
            "ICD-10 Database Path",
            value=config.get("data", {}).get("icd10_database", "./data/icd10_codes.csv")
        )
        
        snomed_path = st.text_input(
            "SNOMED CT Folder Path",
            value=config.get("data", {}).get("snomed_ct_folder", "./data/snomed_ct/")
        )
        
        # Save settings button
        if st.button("Save Settings"):
            # Update configuration
            config["hybrid_search"] = {
                "vector_weight": vector_weight,
                "keyword_weight": keyword_weight,
                "fuzzy_weight": fuzzy_weight
            }
            
            config["data"] = {
                "icd10_database": icd10_path,
                "snomed_ct_folder": snomed_path
            }
            
            # Save to file
            try:
                with open(args.config, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                st.success("Settings saved successfully!")
            except Exception as e:
                st.error(f"Error saving settings: {e}")
        
        # System information
        with st.expander("System Information"):
            if 'system' in st.session_state:
                # System info (existing code)
                # ...
                pass
            else:
                st.info("System not initialized")

if __name__ == "__main__":
    create_ui()