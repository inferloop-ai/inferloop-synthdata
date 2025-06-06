import torch
import clip
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class SemanticProfiler:
    """Extract semantic information from images using YOLO and CLIP."""
    
    def __init__(self, yolo_model: str = "yolov8n.pt", clip_model: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize YOLO for object detection
        try:
            self.yolo = YOLO(yolo_model)
            logger.info(f"Loaded YOLO model: {yolo_model}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.yolo = None
            
        # Initialize CLIP for scene understanding
        try:
            self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)
            logger.info(f"Loaded CLIP model: {clip_model}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.clip_model = None
            
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Common scene categories for CLIP classification
        self.scene_categories = [
            "indoor scene", "outdoor scene", "urban environment", "natural landscape",
            "daytime scene", "nighttime scene", "clear weather", "rainy weather",
            "crowded area", "empty space", "residential area", "commercial area"
        ]
        
    def profile_single_image(self, image: np.ndarray) -> Dict:
        """Extract semantic features from a single image."""
        features = {}
        
        # Object detection with YOLO
        if self.yolo is not None:
            features.update(self._detect_objects(image))
            
        # Scene classification with CLIP
        if self.clip_model is not None:
            features.update(self._classify_scene(image))
            
        # Face detection
        features.update(self._detect_faces(image))
        
        return features
    
    def _detect_objects(self, image: np.ndarray) -> Dict:
        """Detect objects using YOLO and return class distribution."""
        try:
            results = self.yolo(image, verbose=False)
            
            if not results or len(results) == 0:
                return {'object_count': 0, 'detected_classes': []}
            
            # Extract detections
            boxes = results[0].boxes
            if boxes is None:
                return {'object_count': 0, 'detected_classes': []}
                
            # Get class names and counts
            classes = boxes.cls.cpu().numpy() if len(boxes.cls) > 0 else []
            class_names = [self.yolo.names[int(cls)] for cls in classes]
            
            # Count occurrences of each class
            class_counts = {}
            for class_name in class_names:
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Calculate confidence scores
            confidences = boxes.conf.cpu().numpy() if len(boxes.conf) > 0 else []
            avg_confidence = float(np.mean(confidences)) if len(confidences) > 0 else 0.0
            
            return {
                'object_count': len(class_names),
                'detected_classes': class_names,
                'class_distribution': class_counts,
                'avg_detection_confidence': avg_confidence,
                'top_classes': list(dict(sorted(class_counts.items(), 
                                              key=lambda x: x[1], reverse=True)[:5]).keys())
            }
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return {'object_count': 0, 'detected_classes': []}
    
    def _classify_scene(self, image: np.ndarray) -> Dict:
        """Classify scene using CLIP."""
        try:
            # Preprocess image for CLIP
            pil_image = self._numpy_to_pil(image)
            image_tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Tokenize scene categories
            text_tokens = clip.tokenize(self.scene_categories).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                logits_per_image, logits_per_text = self.clip_model(image_tensor, text_tokens)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            # Create scene classification results
            scene_scores = {}
            for i, category in enumerate(self.scene_categories):
                scene_scores[f"scene_{category.replace(' ', '_')}"] = float(probs[i])
            
            # Get top scene predictions
            top_scenes = sorted(zip(self.scene_categories, probs), 
                              key=lambda x: x[1], reverse=True)[:3]
            
            return {
                'scene_scores': scene_scores,
                'top_scene': top_scenes[0][0],
                'top_scene_confidence': float(top_scenes[0][1]),
                'scene_predictions': [(scene, float(score)) for scene, score in top_scenes]
            }
            
        except Exception as e:
            logger.error(f"Scene classification failed: {e}")
            return {'scene_scores': {}, 'top_scene': 'unknown'}
    
    def _detect_faces(self, image: np.ndarray) -> Dict:
        """Detect faces in the image."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            return {
                'face_count': len(faces),
                'has_faces': len(faces) > 0,
                'face_density': len(faces) / (image.shape[0] * image.shape[1]) * 1000000  # faces per megapixel
            }
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return {'face_count': 0, 'has_faces': False}
    
    def _numpy_to_pil(self, image: np.ndarray):
        """Convert numpy array to PIL Image."""
        from PIL import Image
        if len(image.shape) == 3:
            # BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image_rgb)
        else:
            return Image.fromarray(image)
    
    def profile_batch(self, images: List[np.ndarray]) -> Dict:
        """Profile a batch of images and return aggregated semantic features."""
        all_features = []
        
        for img in images:
            features = self.profile_single_image(img)
            all_features.append(features)
        
        # Aggregate numeric features
        aggregated = {}
        
        if all_features:
            # Object detection aggregation
            total_objects = sum(f.get('object_count', 0) for f in all_features)
            all_classes = []
            for f in all_features:
                all_classes.extend(f.get('detected_classes', []))
            
            # Count all detected classes across batch
            batch_class_distribution = {}
            for class_name in all_classes:
                batch_class_distribution[class_name] = batch_class_distribution.get(class_name, 0) + 1
            
            aggregated.update({
                'total_objects_detected': total_objects,
                'avg_objects_per_image': total_objects / len(all_features),
                'unique_classes_detected': len(set(all_classes)),
                'batch_class_distribution': batch_class_distribution
            })
            
            # Scene aggregation
            scene_scores_sum = {}
            for f in all_features:
                for scene_key, score in f.get('scene_scores', {}).items():
                    scene_scores_sum[scene_key] = scene_scores_sum.get(scene_key, 0) + score
            
            # Average scene scores
            if scene_scores_sum:
                aggregated['avg_scene_scores'] = {
                    k: v / len(all_features) for k, v in scene_scores_sum.items()
                }
            
            # Face detection aggregation
            total_faces = sum(f.get('face_count', 0) for f in all_features)
            images_with_faces = sum(1 for f in all_features if f.get('has_faces', False))
            
            aggregated.update({
                'total_faces_detected': total_faces,
                'avg_faces_per_image': total_faces / len(all_features),
                'pct_images_with_faces': (images_with_faces / len(all_features)) * 100
            })
        
        return aggregated

if __name__ == "__main__":
    # Test the semantic profiler
    profiler = SemanticProfiler()
    
    # Create a test image
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    features = profiler.profile_single_image(test_img)
    
    print("Semantic Profile Features:")
    for key, value in features.items():
        print(f"{key}: {value}")
