import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class PrivacyValidator:
    """Validate images for privacy concerns including faces, text, and PII."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize text detector (if available)
        try:
            import pytesseract
            self.ocr_available = True
        except ImportError:
            logger.warning("pytesseract not available - text detection disabled")
            self.ocr_available = False
        
        # PII patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'license_plate': r'\b[A-Z]{1,3}[-\s]?\d{1,4}[-\s]?[A-Z]{0,3}\b'
        }
    
    def validate_batch(self,
                      image_paths: List[str],
                      check_faces: bool = True,
                      check_text: bool = True,
                      anonymize: bool = False) -> Dict:
        """Validate a batch of images for privacy concerns."""
        
        logger.info(f"Privacy validation for {len(image_paths)} images")
        
        results = {
            'num_images': len(image_paths),
            'face_detection': {},
            'text_detection': {},
            'privacy_issues': [],
            'privacy_score': 0,
            'individual_results': []
        }
        
        if anonymize:
            results['anonymized_images'] = {}
        
        # Process each image
        face_stats = {'images_with_faces': 0, 'total_faces': 0, 'face_details': []}
        text_stats = {'images_with_text': 0, 'potential_pii': 0, 'text_details': []}
        
        for i, image_path in enumerate(image_paths):
            try:
                image_result = self._validate_single_image(
                    image_path, check_faces, check_text, anonymize
                )
                
                results['individual_results'].append(image_result)
                
                # Aggregate face detection
                if check_faces and 'faces' in image_result:
                    face_data = image_result['faces']
                    if face_data['num_faces'] > 0:
                        face_stats['images_with_faces'] += 1
                        face_stats['total_faces'] += face_data['num_faces']
                        face_stats['face_details'].append({
                            'image': image_path,
                            'num_faces': face_data['num_faces'],
                            'face_sizes': face_data.get('face_sizes', [])
                        })
                
                # Aggregate text detection
                if check_text and 'text' in image_result:
                    text_data = image_result['text']
                    if text_data['has_text']:
                        text_stats['images_with_text'] += 1
                    if text_data['has_pii']:
                        text_stats['potential_pii'] += 1
                        text_stats['text_details'].append({
                            'image': image_path,
                            'pii_types': text_data.get('pii_types', [])
                        })
                
                # Collect anonymized image if requested
                if anonymize and 'anonymized_image' in image_result:
                    results['anonymized_images'][image_path] = image_result['anonymized_image']
                
            except Exception as e:
                logger.error(f"Failed to validate {image_path}: {e}")
                continue
        
        # Calculate aggregate statistics
        if check_faces:
            face_stats['avg_faces_per_image'] = (
                face_stats['total_faces'] / len(image_paths) if image_paths else 0
            )
            results['face_detection'] = face_stats
        
        if check_text:
            results['text_detection'] = text_stats
        
        # Calculate privacy score
        results['privacy_score'] = self._calculate_privacy_score(results)
        
        # Generate privacy issues summary
        results['privacy_issues'] = self._generate_privacy_issues(results)
        
        return results
    
    def _validate_single_image(self,
                              image_path: str,
                              check_faces: bool,
                              check_text: bool,
                              anonymize: bool) -> Dict:
        """Validate a single image for privacy concerns."""
        
        result = {'image_path': image_path}
        
        # Load image
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            pil_image = Image.open(image_path)
            
        except Exception as e:
            result['error'] = str(e)
            return result
        
        # Face detection
        if check_faces:
            result['faces'] = self._detect_faces(image)
        
        # Text detection
        if check_text:
            result['text'] = self._detect_text_and_pii(pil_image)
        
        # Generate anonymized version if requested
        if anonymize:
            result['anonymized_image'] = self._anonymize_image(image, pil_image, result)
        
        return result
    
    def _detect_faces(self, image: np.ndarray) -> Dict:
        """Detect faces in the image."""
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            face_data = {
                'num_faces': len(faces),
                'face_locations': faces.tolist() if len(faces) > 0 else [],
                'face_sizes': []
            }
            
            # Calculate face sizes
            for (x, y, w, h) in faces:
                face_area = w * h
                image_area = image.shape[0] * image.shape[1]
                face_size_ratio = face_area / image_area
                face_data['face_sizes'].append({
                    'width': w,
                    'height': h,
                    'area_ratio': face_size_ratio
                })
            
            return face_data
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return {'num_faces': 0, 'error': str(e)}
    
    def _detect_text_and_pii(self, image: Image.Image) -> Dict:
        """Detect text and potential PII in the image."""
        
        text_data = {
            'has_text': False,
            'extracted_text': '',
            'has_pii': False,
            'pii_types': [],
            'pii_matches': {}
        }
        
        if not self.ocr_available:
            text_data['error'] = 'OCR not available'
            return text_data
        
        try:
            import pytesseract
            
            # Extract text using OCR
            extracted_text = pytesseract.image_to_string(image)
            
            if extracted_text.strip():
                text_data['has_text'] = True
                text_data['extracted_text'] = extracted_text.strip()
                
                # Check for PII patterns
                pii_found = self._check_for_pii(extracted_text)
                if pii_found:
                    text_data['has_pii'] = True
                    text_data['pii_types'] = list(pii_found.keys())
                    text_data['pii_matches'] = pii_found
            
            return text_data
            
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            text_data['error'] = str(e)
            return text_data
    
    def _check_for_pii(self, text: str) -> Dict:
        """Check text for PII patterns."""
        
        pii_found = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                pii_found[pii_type] = matches
        
        return pii_found
    
    def _anonymize_image(self, cv_image: np.ndarray, pil_image: Image.Image, validation_result: Dict) -> Image.Image:
        """Create an anonymized version of the image."""
        
        anonymized = cv_image.copy()
        
        # Blur faces if detected
        if 'faces' in validation_result and validation_result['faces']['num_faces'] > 0:
            faces = validation_result['faces']['face_locations']
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_region = anonymized[y:y+h, x:x+w]
                
                # Apply strong blur
                blurred_face = cv2.GaussianBlur(face_region, (51, 51), 0)
                
                # Replace face region with blurred version
                anonymized[y:y+h, x:x+w] = blurred_face
        
        # Convert back to PIL Image
        anonymized_rgb = cv2.cvtColor(anonymized, cv2.COLOR_BGR2RGB)
        return Image.fromarray(anonymized_rgb)
    
    def _calculate_privacy_score(self, results: Dict) -> float:
        """Calculate overall privacy score (0-100, higher is better)."""
        
        score = 100.0
        num_images = results['num_images']
        
        if num_images == 0:
            return score
        
        # Face detection penalty
        if 'face_detection' in results:
            face_stats = results['face_detection']
            images_with_faces = face_stats.get('images_with_faces', 0)
            face_ratio = images_with_faces / num_images
            
            # Penalize based on face presence
            face_penalty = face_ratio * 40  # Up to 40 points penalty
            score -= face_penalty
        
        # Text/PII detection penalty
        if 'text_detection' in results:
            text_stats = results['text_detection']
            images_with_pii = text_stats.get('potential_pii', 0)
            pii_ratio = images_with_pii / num_images
            
            # Heavy penalty for PII
            pii_penalty = pii_ratio * 50  # Up to 50 points penalty
            score -= pii_penalty
            
            # Light penalty for any text
            images_with_text = text_stats.get('images_with_text', 0)
            text_ratio = images_with_text / num_images
            text_penalty = text_ratio * 10  # Up to 10 points penalty
            score -= text_penalty
        
        return max(0.0, min(100.0, score))
    
    def _generate_privacy_issues(self, results: Dict) -> List[str]:
        """Generate a list of privacy issues found."""
        
        issues = []
        
        # Face-related issues
        if 'face_detection' in results:
            face_stats = results['face_detection']
            if face_stats.get('images_with_faces', 0) > 0:
                total_faces = face_stats.get('total_faces', 0)
                images_with_faces = face_stats.get('images_with_faces', 0)
                
                issues.append(f"Found {total_faces} faces across {images_with_faces} images")
                
                # Check for large faces (potential privacy concern)
                face_details = face_stats.get('face_details', [])
                large_faces = 0
                for detail in face_details:
                    for face_size in detail.get('face_sizes', []):
                        if face_size.get('area_ratio', 0) > 0.1:  # Face > 10% of image
                            large_faces += 1
                
                if large_faces > 0:
                    issues.append(f"Found {large_faces} large/prominent faces")
        
        # Text-related issues
        if 'text_detection' in results:
            text_stats = results['text_detection']
            
            if text_stats.get('potential_pii', 0) > 0:
                pii_count = text_stats.get('potential_pii', 0)
                issues.append(f"Found potential PII in {pii_count} images")
                
                # Detail PII types found
                text_details = text_stats.get('text_details', [])
                all_pii_types = set()
                for detail in text_details:
                    all_pii_types.update(detail.get('pii_types', []))
                
                if all_pii_types:
                    issues.append(f"PII types detected: {', '.join(sorted(all_pii_types))}")
            
            if text_stats.get('images_with_text', 0) > 0:
                text_count = text_stats.get('images_with_text', 0)
                issues.append(f"Found readable text in {text_count} images")
        
        if not issues:
            issues.append("No significant privacy concerns detected")
        
        return issues
    
    def anonymize_dataset(self,
                         input_dir: str,
                         output_dir: str,
                         blur_faces: bool = True,
                         blur_text: bool = True) -> Dict:
        """Anonymize an entire dataset."""
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(input_path.glob(ext))
        
        if not image_files:
            raise ValueError(f"No images found in {input_dir}")
        
        logger.info(f"Anonymizing {len(image_files)} images...")
        
        processed_count = 0
        
        for image_file in image_files:
            try:
                # Validate and anonymize
                result = self._validate_single_image(
                    str(image_file),
                    check_faces=blur_faces,
                    check_text=blur_text,
                    anonymize=True
                )
                
                # Save anonymized image
                if 'anonymized_image' in result:
                    output_file = output_path / image_file.name
                    result['anonymized_image'].save(output_file)
                    processed_count += 1
                else:
                    # Copy original if no anonymization needed
                    output_file = output_path / image_file.name
                    Image.open(image_file).save(output_file)
                    processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to anonymize {image_file}: {e}")
                continue
        
        summary = {
            'input_directory': str(input_path),
            'output_directory': str(output_path),
            'total_images': len(image_files),
            'processed_images': processed_count,
            'settings': {
                'blur_faces': blur_faces,
                'blur_text': blur_text
            }
        }
        
        logger.info(f"Anonymization complete: {processed_count}/{len(image_files)} images processed")
        
        return summary

if __name__ == "__main__":
    # Test the privacy validator
    try:
        validator = PrivacyValidator()
        
        # Test with a dummy image (would need real images in practice)
        test_dir = Path("./data/generated")
        if test_dir.exists():
            image_files = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
            if image_files:
                results = validator.validate_batch(
                    [str(f) for f in image_files[:3]],
                    check_faces=True,
                    check_text=True,
                    anonymize=False
                )
                
                print("Privacy validation results:")
                print(f"Privacy Score: {results['privacy_score']:.2f}/100")
                print(f"Issues: {results['privacy_issues']}")
                
            else:
                print("No test images found")
        else:
            print("Test directory not found")
            
    except Exception as e:
        print(f"Test failed: {e}")
