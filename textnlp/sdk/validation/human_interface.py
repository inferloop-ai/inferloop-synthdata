# sdk/validation/human_interface.py
from typing import List, Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)

class HumanValidator:
    """Interface for human evaluation of synthetic text"""
    
    def __init__(self):
        self.evaluations = []
    
    def create_evaluation_task(self, text: str, criteria: List[str], task_id: str) -> Dict:
        """Create a human evaluation task"""
        task = {
            "task_id": task_id,
            "text": text,
            "criteria": criteria,
            "status": "pending",
            "scores": {},
            "comments": ""
        }
        
        self.evaluations.append(task)
        return task
    
    def submit_evaluation(self, task_id: str, scores: Dict[str, int], comments: str = "") -> bool:
        """Submit human evaluation scores"""
        for evaluation in self.evaluations:
            if evaluation["task_id"] == task_id:
                evaluation["scores"] = scores
                evaluation["comments"] = comments
                evaluation["status"] = "completed"
                logger.info(f"Evaluation {task_id} completed")
                return True
        
        logger.error(f"Task {task_id} not found")
        return False
    
    def get_completed_evaluations(self) -> List[Dict]:
        """Get all completed evaluations"""
        return [e for e in self.evaluations if e["status"] == "completed"]
    
    def export_evaluations(self, filepath: str):
        """Export evaluations to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.evaluations, f, indent=2)
