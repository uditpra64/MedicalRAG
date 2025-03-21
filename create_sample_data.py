import os
import pandas as pd

def create_sample_icd10_data(filename="data/icd10_codes.csv"):
    """Create a sample ICD-10 data file for testing."""
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
        {"code": "C50", "description": "Malignant neoplasm of breast"}
    ]
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create and save dataframe
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    
    print(f"Created sample ICD-10 data with {len(df)} entries")
    return filename

if __name__ == "__main__":
    create_sample_icd10_data()