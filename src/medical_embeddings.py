import os
from langchain.embeddings import HuggingFaceEmbeddings

def setup_medical_embeddings(model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext", 
                            device="cpu"):
    """
    Initialize a medical domain-specific embedding model.
    
    Args:
        model_name (str): Name of the medical embedding model to use
        device (str): Device to run the model on ('cpu' or 'cuda')
        
    Returns:
        HuggingFaceEmbeddings: Initialized embedding model
    """
    print(f"Loading medical embedding model: {model_name}")
    
    # Configure the embedding model
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings

# Example usage:
# Select the appropriate model based on your specific needs:
# 1. SapBERT: Good for medical entity normalization
# 2. BiomedBERT: General biomedical domain
# 3. ClinicalBERT: Focused on clinical notes
# 4. SciBERT: Scientific/biomedical literature

# medical_embeddings = setup_medical_embeddings(
#     model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
#     device="cuda" if torch.cuda.is_available() else "cpu"
# )