import os
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class FeedbackManager:
    """
    Manages user feedback to improve the medical search system over time.
    Records feedback, provides analysis, and enables retraining based on feedback.
    """
    
    def __init__(self, feedback_dir: str = "data/feedback"):
        """
        Initialize the feedback manager.
        
        Args:
            feedback_dir: Directory to store feedback data
        """
        self.feedback_dir = feedback_dir
        self.feedback_file = os.path.join(feedback_dir, "feedback_data.csv")
        self.active_learning_file = os.path.join(feedback_dir, "active_learning_queue.json")
        
        # Create feedback directory if it doesn't exist
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Initialize feedback dataframe
        self._initialize_feedback_store()
        
        # Initialize active learning queue
        self._initialize_active_learning_queue()
    
    def _initialize_feedback_store(self):
        """Initialize the feedback store CSV file if it doesn't exist."""
        if not os.path.exists(self.feedback_file):
            # Create empty dataframe with required columns
            columns = [
                "timestamp", "query", "system_diagnosis", "expert_diagnosis", 
                "confidence_score", "is_correct", "expert_notes"
            ]
            
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.feedback_file, index=False)
            logger.info(f"Created new feedback store at {self.feedback_file}")
    
    def _initialize_active_learning_queue(self):
        """Initialize the active learning queue if it doesn't exist."""
        if not os.path.exists(self.active_learning_file):
            # Create empty queue
            queue = {
                "high_priority": [],  # Low confidence matches
                "medium_priority": [], # Moderate confidence matches
                "review_complete": []  # Reviewed items
            }
            
            with open(self.active_learning_file, "w") as f:
                json.dump(queue, f, indent=2)
            
            logger.info(f"Created new active learning queue at {self.active_learning_file}")
    
    def record_feedback(
        self,
        query: str,
        system_diagnosis: str,
        expert_diagnosis: str,
        confidence_score: float,
        is_correct: bool,
        expert_notes: Optional[str] = None
    ) -> bool:
        """
        Record expert feedback for a diagnosis.
        
        Args:
            query: Original query
            system_diagnosis: Diagnosis provided by the system
            expert_diagnosis: Correct diagnosis from expert
            confidence_score: System's confidence score
            is_correct: Whether the system diagnosis was correct
            expert_notes: Additional notes from expert
            
        Returns:
            bool: Success status
        """
        try:
            # Create feedback entry
            feedback = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "system_diagnosis": system_diagnosis,
                "expert_diagnosis": expert_diagnosis,
                "confidence_score": confidence_score,
                "is_correct": is_correct,
                "expert_notes": expert_notes or ""
            }
            
            # Append to dataframe
            feedback_df = pd.read_csv(self.feedback_file)
            feedback_df = pd.concat([feedback_df, pd.DataFrame([feedback])], ignore_index=True)
            feedback_df.to_csv(self.feedback_file, index=False)
            
            # Update active learning queue
            self._remove_from_active_learning_queue(query)
            
            logger.info(f"Recorded feedback for query: '{query}'")
            return True
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return False
    
    def queue_for_expert_review(
        self,
        query: str,
        system_diagnosis: str,
        confidence_score: float,
        alternative_diagnoses: Optional[List[str]] = None,
        priority: str = None
    ) -> bool:
        """
        Queue a case for expert review in the active learning system.
        
        Args:
            query: Original query
            system_diagnosis: Diagnosis provided by the system
            confidence_score: System's confidence score
            alternative_diagnoses: List of alternative diagnoses
            priority: Priority level ("high_priority", "medium_priority", or None for auto-determination)
            
        Returns:
            bool: Success status
        """
        try:
            # Determine priority based on confidence score if not explicitly specified
            if priority is None:
                if confidence_score < 0.7:
                    priority = "high_priority"
                else:
                    priority = "medium_priority"
            
            # Create queue entry
            entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "system_diagnosis": system_diagnosis,
                "confidence_score": confidence_score,
                "alternative_diagnoses": alternative_diagnoses or []
            }
            
            # Add to queue
            with open(self.active_learning_file, "r") as f:
                queue = json.load(f)
            
            # Check if already in queue or review_complete
            existing_queries = [item["query"] for item in queue.get(priority, [])]
            completed_queries = queue.get("review_complete", [])
            
            if query not in existing_queries and query not in completed_queries:
                # Make sure the priority queue exists
                if priority not in queue:
                    queue[priority] = []
                    
                queue[priority].append(entry)
                    
                # Save updated queue
                with open(self.active_learning_file, "w") as f:
                    json.dump(queue, f, indent=2)
                
                logger.info(f"Added query to {priority} review queue: '{query}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding to active learning queue: {e}")
            return False
    
    def _remove_from_active_learning_queue(self, query: str):
        """
        Remove a reviewed item from the active learning queue.
        
        Args:
            query: Query to remove
        """
        try:
            with open(self.active_learning_file, "r") as f:
                queue = json.load(f)
            
            # Find and remove from priority queues
            for priority in ["high_priority", "medium_priority"]:
                queue[priority] = [item for item in queue[priority] if item["query"] != query]
            
            # Add to review_complete list
            queue["review_complete"].append(query)
            
            # Save updated queue
            with open(self.active_learning_file, "w") as f:
                json.dump(queue, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating active learning queue: {e}")
    
    def get_expert_review_queue(self, priority: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the current expert review queue.
        
        Args:
            priority: Priority level ("high_priority", "medium_priority", or None for all)
            
        Returns:
            List of queue entries
        """
        try:
            with open(self.active_learning_file, "r") as f:
                queue = json.load(f)
            
            if priority:
                return queue.get(priority, [])
            else:
                # Combine all priority levels
                return queue.get("high_priority", []) + queue.get("medium_priority", [])
                
        except Exception as e:
            logger.error(f"Error getting review queue: {e}")
            return []
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the feedback data.
        
        Returns:
            dict: Statistics about feedback data
        """
        try:
            feedback_df = pd.read_csv(self.feedback_file)
            
            # Calculate statistics
            total_entries = len(feedback_df)
            correct_entries = len(feedback_df[feedback_df["is_correct"]])
            accuracy = correct_entries / total_entries if total_entries > 0 else 0
            
            # Accuracy by confidence band
            feedback_df["confidence_band"] = pd.cut(
                feedback_df["confidence_score"],
                bins=[0, 0.7, 0.9, 1.0],
                labels=["low", "medium", "high"]
            )
            
            accuracy_by_confidence = {}
            for band in ["low", "medium", "high"]:
                band_df = feedback_df[feedback_df["confidence_band"] == band]
                band_total = len(band_df)
                band_correct = len(band_df[band_df["is_correct"]])
                accuracy_by_confidence[band] = band_correct / band_total if band_total > 0 else 0
            
            return {
                "total_feedback_entries": total_entries,
                "correct_diagnoses": correct_entries,
                "overall_accuracy": accuracy,
                "accuracy_by_confidence": accuracy_by_confidence,
                "last_update": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating feedback statistics: {e}")
            return {
                "error": str(e),
                "total_feedback_entries": 0
            }
    
    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """
        Generate improvement suggestions based on feedback data.
        
        Returns:
            list: Improvement suggestions
        """
        try:
            feedback_df = pd.read_csv(self.feedback_file)
            
            # Focus on incorrect diagnoses
            incorrect_df = feedback_df[~feedback_df["is_correct"]]
            
            # Group by system diagnosis to find common errors
            error_groups = incorrect_df.groupby("system_diagnosis").agg({
                "query": list,
                "expert_diagnosis": list
            }).reset_index()
            
            # Generate suggestions
            suggestions = []
            for _, row in error_groups.iterrows():
                if len(row["query"]) >= 2:  # Require at least 2 errors of the same type
                    suggestions.append({
                        "system_diagnosis": row["system_diagnosis"],
                        "example_queries": row["query"][:3],  # Show up to 3 examples
                        "expert_diagnoses": list(set(row["expert_diagnosis"])),
                        "count": len(row["query"]),
                        "suggestion": f"Improve detection of '{row['system_diagnosis']}'"
                    })
            
            # Sort by error count (descending)
            suggestions.sort(key=lambda x: x["count"], reverse=True)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {e}")
            return []


# Add to main_application.py:
"""
from src.feedback_manager import FeedbackManager

# In MedicalDiseaseNameSearchSystem.__init__:
self.feedback_manager = FeedbackManager()

# Add method to record feedback:
def record_expert_feedback(self, query, system_diagnosis, expert_diagnosis, is_correct, expert_notes=None):
    '''Record expert feedback for a diagnosis.'''
    return self.feedback_manager.record_feedback(
        query=query,
        system_diagnosis=system_diagnosis,
        expert_diagnosis=expert_diagnosis,
        confidence_score=self.last_confidence_score,
        is_correct=is_correct,
        expert_notes=expert_notes
    )

# Modify convert_medical_expression to use feedback system:
def convert_medical_expression(self, expression, confidence_threshold=None):
    '''Convert a non-standard medical expression to a standardized disease name.'''
    # ... existing code ...
    
    # Store last confidence score for feedback
    self.last_confidence_score = parsed_result['confidence_score']
    
    # Queue for expert review if confidence is low or needs human review
    if parsed_result['confidence_score'] < 0.9 or parsed_result.get('needs_human_review', False):
        self.feedback_manager.queue_for_expert_review(
            query=expression,
            system_diagnosis=parsed_result['standard_diagnosis'],
            confidence_score=parsed_result['confidence_score'],
            alternative_diagnoses=parsed_result.get('alternative_diagnoses', [])
        )
    
    return parsed_result
"""