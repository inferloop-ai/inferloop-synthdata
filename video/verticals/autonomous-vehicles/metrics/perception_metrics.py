"""
Autonomous Vehicles Perception Metrics
Evaluates the quality of perception systems in synthetic video data
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import cv2
import json
import logging

logger = logging.getLogger(__name__)

class PerceptionMetrics:
    """
    Metrics for evaluating perception system performance in autonomous vehicle scenarios
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize perception metrics calculator
        
        Args:
            config_path: Path to configuration file
        """
        self.config = {}
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Default thresholds
        self.thresholds = {
            "object_detection_iou": 0.7,
            "lane_detection_accuracy": 0.9,
            "traffic_sign_recognition": 0.85,
            "depth_estimation_error": 0.1,  # 10% relative error
            "semantic_segmentation_iou": 0.75
        }
        
        # Override with config if provided
        if 'thresholds' in self.config:
            self.thresholds.update(self.config['thresholds'])
            
        logger.info(f"Initialized perception metrics with thresholds: {self.thresholds}")
    
    def calculate_object_detection_metrics(self, 
                                          ground_truth: List[Dict[str, Any]], 
                                          predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate object detection metrics (precision, recall, F1, mAP)
        
        Args:
            ground_truth: List of ground truth objects with bounding boxes
            predictions: List of predicted objects with bounding boxes
            
        Returns:
            Dictionary of metrics
        """
        if not ground_truth or not predictions:
            logger.warning("Empty ground truth or predictions for object detection metrics")
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "mAP": 0.0
            }
        
        # Calculate IoU for each prediction with ground truth
        matches = []
        for gt in ground_truth:
            best_iou = 0.0
            best_pred = None
            
            for pred in predictions:
                if gt['class'] == pred['class']:
                    iou = self._calculate_iou(gt['bbox'], pred['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_pred = pred
            
            if best_iou >= self.thresholds["object_detection_iou"]:
                matches.append((gt, best_pred, best_iou))
        
        # Calculate metrics
        tp = len(matches)
        fp = len(predictions) - tp
        fn = len(ground_truth) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Simple mAP calculation
        ap = precision * recall  # Simplified AP
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mAP": ap
        }
    
    def calculate_lane_detection_metrics(self,
                                        ground_truth_lanes: List[np.ndarray],
                                        predicted_lanes: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate lane detection metrics
        
        Args:
            ground_truth_lanes: List of ground truth lane points
            predicted_lanes: List of predicted lane points
            
        Returns:
            Dictionary of metrics
        """
        if not ground_truth_lanes or not predicted_lanes:
            logger.warning("Empty ground truth or predictions for lane detection metrics")
            return {
                "accuracy": 0.0,
                "completeness": 0.0,
                "f1_score": 0.0
            }
        
        # Match lanes and calculate distances
        total_distance = 0.0
        matched_points = 0
        
        for gt_lane in ground_truth_lanes:
            best_distance = float('inf')
            best_pred = None
            
            for pred_lane in predicted_lanes:
                distance = self._calculate_lane_distance(gt_lane, pred_lane)
                if distance < best_distance:
                    best_distance = distance
                    best_pred = pred_lane
            
            if best_pred is not None:
                total_distance += best_distance
                matched_points += len(gt_lane)
        
        avg_distance = total_distance / matched_points if matched_points > 0 else float('inf')
        accuracy = max(0.0, 1.0 - avg_distance / 100.0)  # Normalize distance
        
        # Calculate completeness (percentage of ground truth points matched)
        total_gt_points = sum(len(lane) for lane in ground_truth_lanes)
        completeness = matched_points / total_gt_points if total_gt_points > 0 else 0.0
        
        # Calculate F1 score
        f1 = 2 * accuracy * completeness / (accuracy + completeness) if (accuracy + completeness) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "completeness": completeness,
            "f1_score": f1
        }
    
    def calculate_traffic_sign_recognition(self,
                                          ground_truth: List[Dict[str, Any]],
                                          predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate traffic sign recognition metrics
        
        Args:
            ground_truth: List of ground truth traffic signs
            predictions: List of predicted traffic signs
            
        Returns:
            Dictionary of metrics
        """
        if not ground_truth or not predictions:
            logger.warning("Empty ground truth or predictions for traffic sign recognition")
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }
        
        # Match predictions to ground truth
        correct = 0
        for gt in ground_truth:
            for pred in predictions:
                # Check if prediction matches ground truth (position and type)
                if (self._calculate_iou(gt['bbox'], pred['bbox']) >= self.thresholds["traffic_sign_recognition"] and
                    gt['sign_type'] == pred['sign_type']):
                    correct += 1
                    break
        
        # Calculate metrics
        accuracy = correct / len(ground_truth) if len(ground_truth) > 0 else 0.0
        precision = correct / len(predictions) if len(predictions) > 0 else 0.0
        recall = correct / len(ground_truth) if len(ground_truth) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            box1: [x1, y1, x2, y2] format
            box2: [x1, y1, x2, y2] format
            
        Returns:
            IoU value
        """
        # Calculate intersection area
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _calculate_lane_distance(self, lane1: np.ndarray, lane2: np.ndarray) -> float:
        """
        Calculate average distance between two lane curves
        
        Args:
            lane1: Array of (x,y) points for first lane
            lane2: Array of (x,y) points for second lane
            
        Returns:
            Average distance
        """
        # Interpolate lanes to have the same number of points
        max_points = max(len(lane1), len(lane2))
        
        if len(lane1) < max_points:
            lane1 = self._interpolate_lane(lane1, max_points)
        
        if len(lane2) < max_points:
            lane2 = self._interpolate_lane(lane2, max_points)
        
        # Calculate point-wise distances
        distances = np.sqrt(np.sum((lane1 - lane2) ** 2, axis=1))
        return np.mean(distances)
    
    def _interpolate_lane(self, lane: np.ndarray, num_points: int) -> np.ndarray:
        """
        Interpolate lane points to have specified number of points
        
        Args:
            lane: Array of (x,y) points
            num_points: Desired number of points
            
        Returns:
            Interpolated lane points
        """
        if len(lane) <= 1:
            return lane
        
        # Create parameter t (0 to 1)
        t = np.linspace(0, 1, len(lane))
        t_new = np.linspace(0, 1, num_points)
        
        # Interpolate x and y separately
        x = np.interp(t_new, t, lane[:, 0])
        y = np.interp(t_new, t, lane[:, 1])
        
        return np.column_stack((x, y))


# Example usage
if __name__ == "__main__":
    # Example data
    metrics = PerceptionMetrics()
    
    # Object detection example
    gt_objects = [
        {"class": "car", "bbox": [100, 150, 200, 250]},
        {"class": "pedestrian", "bbox": [300, 200, 350, 300]}
    ]
    
    pred_objects = [
        {"class": "car", "bbox": [105, 155, 210, 260]},
        {"class": "pedestrian", "bbox": [290, 210, 345, 310]}
    ]
    
    od_metrics = metrics.calculate_object_detection_metrics(gt_objects, pred_objects)
    print(f"Object Detection Metrics: {od_metrics}")
