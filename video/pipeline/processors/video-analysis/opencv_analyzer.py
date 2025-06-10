"""
OpenCV-based Video Analysis Module
Part of the Inferloop SynthData Video Pipeline

This module provides video analysis capabilities using OpenCV,
focusing on frame extraction, object detection, and motion analysis.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenCVAnalyzer:
    """
    Video analysis using OpenCV for the synthetic data pipeline.
    Provides frame extraction, object detection, and motion analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the OpenCV analyzer with optional configuration.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path) if config_path else {}
        
        # Default configuration values
        self.frame_sample_rate = self.config.get('frame_sample_rate', 5)  # Process every Nth frame
        self.detection_confidence = self.config.get('detection_confidence', 0.5)
        self.output_format = self.config.get('output_format', 'json')
        
        # Initialize object detection if enabled
        if self.config.get('enable_object_detection', True):
            self._init_object_detection()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def _init_object_detection(self):
        """Initialize object detection models"""
        # Use OpenCV's DNN module with pre-trained models
        try:
            # Load YOLO model for object detection
            model_path = self.config.get('yolo_model_path', 'models/yolov4.weights')
            config_path = self.config.get('yolo_config_path', 'models/yolov4.cfg')
            
            self.net = cv2.dnn.readNetFromDarknet(config_path, model_path)
            
            # Use CUDA if available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            
            # Load class names
            classes_path = self.config.get('classes_path', 'models/coco.names')
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
                
            logger.info("Object detection model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize object detection: {e}")
            self.net = None
            self.classes = []
    
    def analyze_video(self, video_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze video file and extract metrics and features
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save analysis results (optional)
            
        Returns:
            Dictionary with analysis results
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Analyzing video: {video_path} ({width}x{height}, {fps} fps, {frame_count} frames)")
        
        # Initialize results
        results = {
            "video_path": video_path,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": frame_count / fps if fps > 0 else 0,
            "frames_analyzed": 0,
            "detected_objects": {},
            "motion_analysis": {
                "average_motion": 0.0,
                "motion_peaks": []
            },
            "frame_metrics": []
        }
        
        # Process frames
        frame_idx = 0
        prev_frame = None
        total_motion = 0.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every Nth frame
            if frame_idx % self.frame_sample_rate == 0:
                # Analyze frame
                frame_results = self._analyze_frame(frame, frame_idx)
                
                # Calculate motion if we have a previous frame
                if prev_frame is not None:
                    motion_score = self._calculate_motion(prev_frame, frame)
                    frame_results["motion_score"] = motion_score
                    total_motion += motion_score
                    
                    # Detect motion peaks
                    if motion_score > self.config.get('motion_peak_threshold', 0.1):
                        results["motion_analysis"]["motion_peaks"].append({
                            "frame": frame_idx,
                            "timestamp": frame_idx / fps,
                            "score": motion_score
                        })
                
                results["frame_metrics"].append(frame_results)
                results["frames_analyzed"] += 1
                
                # Update object detection counts
                for obj in frame_results.get("objects", []):
                    obj_class = obj["class"]
                    if obj_class in results["detected_objects"]:
                        results["detected_objects"][obj_class] += 1
                    else:
                        results["detected_objects"][obj_class] = 1
                
                # Save analyzed frame if output directory is provided
                if output_dir:
                    output_path = Path(output_dir) / f"frame_{frame_idx:06d}.jpg"
                    self._save_analyzed_frame(frame, frame_results, str(output_path))
            
            # Keep previous frame for motion analysis
            prev_frame = frame.copy()
            frame_idx += 1
        
        # Calculate average motion
        if results["frames_analyzed"] > 0:
            results["motion_analysis"]["average_motion"] = total_motion / results["frames_analyzed"]
        
        # Release video capture
        cap.release()
        
        # Save results if output directory is provided
        if output_dir:
            output_path = Path(output_dir) / "analysis_results.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def _analyze_frame(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Analyze a single frame"""
        result = {
            "frame_idx": frame_idx,
            "brightness": self._calculate_brightness(frame),
            "contrast": self._calculate_contrast(frame),
            "sharpness": self._calculate_sharpness(frame),
            "objects": []
        }
        
        # Perform object detection if model is loaded
        if hasattr(self, 'net') and self.net is not None:
            objects = self._detect_objects(frame)
            result["objects"] = objects
        
        return result
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in frame using YOLO"""
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Forward pass
        outputs = self.net.forward(output_layers)
        
        # Process detections
        detected_objects = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.detection_confidence:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    detected_objects.append({
                        "class": self.classes[class_id] if class_id < len(self.classes) else "unknown",
                        "confidence": float(confidence),
                        "bbox": [x, y, w, h]
                    })
        
        return detected_objects
    
    def _calculate_brightness(self, frame: np.ndarray) -> float:
        """Calculate average brightness of frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    
    def _calculate_contrast(self, frame: np.ndarray) -> float:
        """Calculate contrast of frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))
    
    def _calculate_sharpness(self, frame: np.ndarray) -> float:
        """Calculate sharpness of frame using Laplacian variance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    def _calculate_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Calculate motion between frames using absolute difference"""
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        
        # Normalize by frame size
        height, width = diff.shape
        motion_score = np.sum(diff) / (255.0 * height * width)
        
        return float(motion_score)
    
    def _save_analyzed_frame(self, frame: np.ndarray, frame_results: Dict[str, Any], output_path: str):
        """Save frame with analysis overlays"""
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Draw object detection boxes
        for obj in frame_results.get("objects", []):
            x, y, w, h = obj["bbox"]
            label = f"{obj['class']} ({obj['confidence']:.2f})"
            
            # Draw rectangle
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(output_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add metrics text
        metrics_text = [
            f"Brightness: {frame_results['brightness']:.2f}",
            f"Contrast: {frame_results['contrast']:.2f}",
            f"Sharpness: {frame_results['sharpness']:.2f}"
        ]
        
        if "motion_score" in frame_results:
            metrics_text.append(f"Motion: {frame_results['motion_score']:.4f}")
        
        # Draw metrics
        for i, text in enumerate(metrics_text):
            cv2.putText(output_frame, text, (10, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save frame
        cv2.imwrite(output_path, output_frame)

# Example usage
if __name__ == "__main__":
    analyzer = OpenCVAnalyzer()
    results = analyzer.analyze_video("sample_video.mp4", "output_analysis")
    print(f"Analysis complete. Analyzed {results['frames_analyzed']} frames.")
    print(f"Detected objects: {results['detected_objects']}")
