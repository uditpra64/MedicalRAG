import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class MedicalOntologyManager:
    """
    Manages medical ontologies like ICD-10 and SNOMED CT.
    Provides mapping and lookup services for medical terms.
    """
    
    def __init__(self, 
                 icd10_path: Optional[str] = None,
                 snomed_ct_path: Optional[str] = None):
        """
        Initialize the medical ontology manager.
        
        Args:
            icd10_path: Path to ICD-10 data file (CSV)
            snomed_ct_path: Path to SNOMED CT data folder
        """
        self.icd10_data = None
        self.snomed_data = None
        
        # Load ICD-10 if path provided
        if icd10_path and os.path.exists(icd10_path):
            self._load_icd10(icd10_path)
        
        # Load SNOMED CT if path provided
        if snomed_ct_path and os.path.exists(snomed_ct_path):
            self._load_snomed_ct(snomed_ct_path)
    
    def _load_icd10(self, file_path: str) -> bool:
        """
        Load ICD-10 codes from CSV file.
        
        Args:
            file_path: Path to ICD-10 CSV file
            
        Returns:
            bool: Success status
        """
        try:
            self.icd10_data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(self.icd10_data)} ICD-10 codes")
            return True
        except Exception as e:
            logger.error(f"Error loading ICD-10 data: {e}")
            return False
    
    def _load_snomed_ct(self, folder_path: str) -> bool:
        """
        Load SNOMED CT data from folder.
        
        Args:
            folder_path: Path to SNOMED CT data folder
            
        Returns:
            bool: Success status
        """
        try:
            # SNOMED CT typically has multiple files - concepts, descriptions, relationships
            concept_file = os.path.join(folder_path, "sct2_Concept_Full.txt")
            description_file = os.path.join(folder_path, "sct2_Description_Full.txt")
            
            if os.path.exists(concept_file) and os.path.exists(description_file):
                # Load concept file
                concepts = pd.read_csv(
                    concept_file, 
                    sep='\t',
                    encoding='utf-8'
                )
                
                # Load descriptions
                descriptions = pd.read_csv(
                    description_file,
                    sep='\t',
                    encoding='utf-8'
                )
                
                # Store both dataframes
                self.snomed_data = {
                    "concepts": concepts,
                    "descriptions": descriptions
                }
                
                logger.info(f"Loaded SNOMED CT data: {len(concepts)} concepts, {len(descriptions)} descriptions")
                return True
            else:
                logger.warning(f"SNOMED CT files not found in {folder_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading SNOMED CT data: {e}")
            return False
    
    def get_icd10_code(self, disease_name: str) -> List[Dict[str, str]]:
        """
        Get ICD-10 codes for a disease name.
        
        Args:
            disease_name: Disease name to look up
            
        Returns:
            List of matching ICD-10 code dictionaries
        """
        if self.icd10_data is None:
            logger.warning("ICD-10 data not loaded")
            return []
        
        try:
            # Search for the disease name in the description column
            # This is a simplified approach - would need refinement in production
            matches = self.icd10_data[
                self.icd10_data['description'].str.contains(disease_name, case=False, na=False)
            ]
            
            # Convert to list of dictionaries
            result = matches.to_dict('records')
            logger.debug(f"Found {len(result)} ICD-10 matches for '{disease_name}'")
            
            return result
        
        except Exception as e:
            logger.error(f"Error searching ICD-10 codes: {e}")
            return []
    
    def get_snomed_code(self, disease_name: str) -> List[Dict[str, str]]:
        """
        Get SNOMED CT codes for a disease name.
        
        Args:
            disease_name: Disease name to look up
            
        Returns:
            List of matching SNOMED CT code dictionaries
        """
        if self.snomed_data is None:
            logger.warning("SNOMED CT data not loaded")
            return []
        
        try:
            # Search for the disease name in the description table
            # This is a simplified approach - would need refinement in production
            descriptions = self.snomed_data["descriptions"]
            
            matches = descriptions[
                descriptions['term'].str.contains(disease_name, case=False, na=False)
            ]
            
            # Get corresponding concepts
            concept_ids = matches['conceptId'].unique()
            concepts = self.snomed_data["concepts"]
            matching_concepts = concepts[concepts['id'].isin(concept_ids)]
            
            # Combine the data
            result = []
            for _, concept in matching_concepts.iterrows():
                concept_matches = matches[matches['conceptId'] == concept['id']]
                for _, desc in concept_matches.iterrows():
                    result.append({
                        'concept_id': concept['id'],
                        'term': desc['term'],
                        'description_type': desc['typeId'],
                        'active': concept['active']
                    })
            
            logger.debug(f"Found {len(result)} SNOMED CT matches for '{disease_name}'")
            return result
        
        except Exception as e:
            logger.error(f"Error searching SNOMED CT codes: {e}")
            return []
    
    def get_ontology_data(self, disease_name: str) -> Dict[str, List]:
        """
        Get comprehensive ontology data for a disease name.
        
        Args:
            disease_name: Disease name to look up
            
        Returns:
            Dictionary containing all matching ontology data
        """
        return {
            "icd10": self.get_icd10_code(disease_name),
            "snomed_ct": self.get_snomed_code(disease_name)
        }


# Usage in main_application.py:
"""
from src.medical_ontology import MedicalOntologyManager

# In MedicalDiseaseNameSearchSystem.__init__:
self.ontology_manager = MedicalOntologyManager(
    icd10_path=config.get("data", {}).get("icd10_database"),
    snomed_ct_path=config.get("data", {}).get("snomed_ct_folder")
)

# Update the get_medical_ontologies method:
def get_medical_ontologies(self, disease_name: str) -> Dict[str, Any]:
    return self.ontology_manager.get_ontology_data(disease_name)
"""