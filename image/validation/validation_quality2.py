import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

class ImageQualityValidator:
    """Validate synthetic image quality using various metrics"""
    
    def __init__(self, config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize pre-trained model for feature extraction
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT).to(self.device)
        self.model.eval()
        
        # Remove the classification layer
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def check_blurriness(self, image_path):
        """Check if image is blurry using Laplacian variance"""
        try:
            img = np.array(Image.open(image_path).convert('L'))
            return cv2.Laplacian(img, cv2.CV_64F).var()
        except Exception as e:
            print(f"Error analyzing {image_path}: {e}")
            return 0
    
    def extract_features(self, image_path):
        """Extract features from image using pre-trained ResNet"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_t = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(img_t)
            
            return features.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def validate_image(self, image_path):
        """Run all validation checks on an image and return quality score"""
        results = {
            "path": image_path,
            "blur_score": self.check_blurriness(image_path),
            "has_features": self.extract_features(image_path) is not None
        }
        
        # Higher blur score means less blurry
        quality_score = min(100, results["blur_score"] / 2)
        results["quality_score"] = quality_score
        
        return results
    
    def validate_directory(self, directory):
        """Validate all images in a directory"""
        image_paths = [os.path.join(directory, f) for f in os.listdir(directory)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        results = []
        for img_path in image_paths:
            results.append(self.validate_image(img_path))
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Validate synthetic image quality")
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Path to image or directory of images")
    parser.add_argument("--output", "-o", type=str, default="validation_results.json",
                        help="Output file for validation results")
    parser.add_argument("--threshold", "-t", type=float, default=50.0,
                        help="Quality threshold (0-100)")
    
    args = parser.parse_args()
    
    validator = ImageQualityValidator()
    
    if os.path.isdir(args.input):
        results = validator.validate_directory(args.input)
        print(f"Validated {len(results)} images")
    else:
        results = [validator.validate_image(args.input)]
        print(f"Validated image with score: {results[0]['quality_score']:.2f}")
    
    # Write results to file
    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    try:
        import cv2
    except ImportError:
        print("OpenCV (cv2) is required for blur detection. Please install with:")
        print("pip install opencv-python")
        exit(1)
        
    main()