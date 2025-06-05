# sdk/formatter.py
import json
import csv
import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataFormatter:
    """Format synthetic data into various output formats"""
    
    @staticmethod
    def to_jsonl(data: List[Dict], filepath: str):
        """Save data to JSONL format"""
        try:
            with open(filepath, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save JSONL: {e}")
            raise
    
    @staticmethod
    def to_csv(data: List[Dict], filepath: str):
        """Save data to CSV format"""
        try:
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            raise
    
    @staticmethod
    def to_markdown(data: List[Dict], filepath: str):
        """Save data to Markdown format"""
        try:
            with open(filepath, 'w') as f:
                f.write("# Synthetic Data Output\n\n")
                
                for i, item in enumerate(data, 1):
                    f.write(f"## Sample {i}\n\n")
                    for key, value in item.items():
                        f.write(f"**{key}:** {value}\n\n")
                    f.write("---\n\n")
            
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save Markdown: {e}")
            raise
    
    @staticmethod
    def format_for_training(prompts: List[str], responses: List[str]) -> List[Dict]:
        """Format prompt-response pairs for training"""
        return [
            {
                "prompt": prompt,
                "response": response,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            for prompt, response in zip(prompts, responses)
        ]

