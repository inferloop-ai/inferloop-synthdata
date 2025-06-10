#!/usr/bin/env python3
"""
Robotics Manipulation Tasks Example
Demonstrates synthetic video generation for robot training
"""

import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoboticsVideoGenerator:
    def __init__(self, api_url="http://localhost:8080"):
        self.api_url = api_url
        
    def generate_manipulation_dataset(self):
        """Generate a complete manipulation training dataset"""
        
        config = {
            "vertical": "robotics",
            "data_sources": [
                {
                    "source_type": "reference",
                    "scenarios": ["grasping", "assembly", "sorting"],
                    "quality_filters": {
                        "min_precision_mm": 0.1,
                        "min_success_rate": 0.9
                    }
                }
            ],
            "generation_config": {
                "engine": "unity",
                "robot_type": "robotic_arm",
                "tasks": [
                    "object_grasping",
                    "precision_placement", 
                    "assembly_operations"
                ],
                "objects": [
                    "bottles", "boxes", "tools", "electronic_components"
                ],
                "environments": ["factory_floor", "laboratory"],
                "duration_seconds": 300,
                "variations": 50
            },
            "quality_requirements": {
                "min_label_accuracy": 0.98,
                "max_frame_lag_ms": 50,
                "manipulation_precision_mm": 0.1,
                "physics_accuracy": 0.95
            },
            "delivery_config": {
                "format": "mp4",
                "include_annotations": True,
                "include_joint_states": True,
                "delivery_method": "sdk"
            }
        }
        
        # Start pipeline
        logger.info("Starting robotics manipulation pipeline...")
        response = requests.post(
            f"{self.api_url}/api/v1/pipeline/start",
            json=config
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to start pipeline: {response.text}")
            return None
            
        pipeline_id = response.json()["pipeline_id"]
        logger.info(f"Pipeline started with ID: {pipeline_id}")
        
        # Monitor progress
        return self.monitor_pipeline(pipeline_id)
    
    def monitor_pipeline(self, pipeline_id):
        """Monitor pipeline progress"""
        while True:
            response = requests.get(
                f"{self.api_url}/api/v1/pipeline/status/{pipeline_id}"
            )
            
            if response.status_code != 200:
                logger.error("Failed to get pipeline status")
                break
                
            status = response.json()
            current_status = status["status"]
            progress = status["progress"]
            stage = status["current_stage"]
            
            logger.info(f"Status: {current_status} | Stage: {stage} | Progress: {progress}%")
            
            if current_status in ["completed", "failed"]:
                break
                
            time.sleep(15)
        
        if current_status == "completed":
            logger.info("🎉 Robotics dataset generation completed!")
            return status
        else:
            logger.error("❌ Pipeline failed")
            return None

def main():
    generator = RoboticsVideoGenerator()
    result = generator.generate_manipulation_dataset()
    
    if result:
        print("✅ Robotics training dataset ready!")
        print(f"📊 Generated videos with manipulation precision: {result['metadata'].get('precision', 'N/A')}")
    else:
        print("❌ Dataset generation failed")

if __name__ == "__main__":
    main()
