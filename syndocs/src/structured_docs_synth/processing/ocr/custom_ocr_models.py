#!/usr/bin/env python3
"""
Custom OCR Models implementation with support for various architectures
"""

import logging
import time
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError, ValidationError
from .ocr_pipeline import BoundingBox, OCRWord, OCRLine, OCRPage


class ModelArchitecture(Enum):
    """Available custom OCR model architectures"""
    CRNN = "crnn"
    ATTENTION_OCR = "attention_ocr"
    PARSEQ = "parseq"
    TROCR_CUSTOM = "trocr_custom"
    EASY_OCR = "easy_ocr"
    PADDLE_OCR = "paddle_ocr"


@dataclass
class CustomOCREngineResult:
    """Custom OCR extraction result"""
    text: str
    confidence: float
    words: List[OCRWord]
    lines: List[OCRLine]
    processing_time: float = 0.0
    model_info: Dict[str, Any] = None


class CustomOCRConfig(BaseModel):
    """Custom OCR engine configuration"""
    
    architecture: ModelArchitecture = Field(ModelArchitecture.CRNN, description="Model architecture")
    model_path: Optional[str] = Field(None, description="Path to trained model")
    config_path: Optional[str] = Field(None, description="Path to model config")
    vocab_path: Optional[str] = Field(None, description="Path to vocabulary file")
    device: str = Field("auto", description="Device to run model on")
    batch_size: int = Field(8, description="Batch size for processing")
    
    # Image preprocessing
    image_height: int = Field(32, description="Input image height")
    image_width: int = Field(128, description="Input image width")
    normalize: bool = Field(True, description="Normalize image pixels")
    
    # Text processing
    max_text_length: int = Field(25, description="Maximum text length")
    character_encoding: str = Field("utf-8", description="Character encoding")
    
    # Model-specific settings
    model_settings: Dict[str, Any] = Field(default_factory=dict, description="Architecture-specific settings")


class BaseOCRModel(ABC):
    """Base class for custom OCR models"""
    
    def __init__(self, config: CustomOCRConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.model = None
        self.vocab = None
        self.device = self._get_device()
    
    def _get_device(self):
        """Determine the device to use"""
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the OCR model"""
        pass
    
    @abstractmethod
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        pass
    
    @abstractmethod
    def predict(self, image_tensor: torch.Tensor) -> Tuple[str, float]:
        """Predict text from image tensor"""
        pass
    
    def extract_text(self, image: Image.Image) -> Tuple[str, float]:
        """Extract text from image"""
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Predict text
            text, confidence = self.predict(image_tensor)
            
            return text, confidence
            
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            return "", 0.0


class CRNNModel(BaseOCRModel):
    """CRNN (Convolutional Recurrent Neural Network) OCR Model"""
    
    def load_model(self):
        """Load CRNN model"""
        try:
            if self.config.model_path and Path(self.config.model_path).exists():
                # Load custom trained model
                self.model = torch.load(self.config.model_path, map_location=self.device)
                self.logger.info(f"Loaded custom CRNN model from {self.config.model_path}")
            else:
                # Create default CRNN architecture
                self.model = self._create_default_crnn()
                self.logger.info("Created default CRNN model")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Load vocabulary
            self._load_vocabulary()
            
        except Exception as e:
            raise ProcessingError(f"Failed to load CRNN model: {e}")
    
    def _create_default_crnn(self):
        """Create default CRNN architecture"""
        class SimpleCRNN(nn.Module):
            def __init__(self, vocab_size=37, hidden_size=256):
                super().__init__()
                # CNN feature extractor
                self.cnn = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d((2, 1), (2, 1)),
                    nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d((2, 1), (2, 1))
                )
                
                # RNN sequence processor
                self.rnn = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True)
                self.classifier = nn.Linear(hidden_size * 2, vocab_size)
            
            def forward(self, x):
                # CNN features
                features = self.cnn(x)
                b, c, h, w = features.size()
                features = features.view(b, c * h, w).permute(2, 0, 1)
                
                # RNN processing
                rnn_out, _ = self.rnn(features)
                output = self.classifier(rnn_out)
                
                return output
        
        return SimpleCRNN()
    
    def _load_vocabulary(self):
        """Load character vocabulary"""
        if self.config.vocab_path and Path(self.config.vocab_path).exists():
            with open(self.config.vocab_path, 'r', encoding=self.config.character_encoding) as f:
                self.vocab = json.load(f)
        else:
            # Default English vocabulary
            self.vocab = {
                'chars': list('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'),
                'blank_token': 0
            }
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for CRNN"""
        try:
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize((self.config.image_width, self.config.image_height), Image.Resampling.LANCZOS)
            
            # Convert to tensor
            image_array = np.array(image).astype(np.float32)
            
            if self.config.normalize:
                image_array = image_array / 255.0
                image_array = (image_array - 0.5) / 0.5  # Normalize to [-1, 1]
            
            # Convert to tensor and add batch dimension
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            raise ProcessingError(f"Image preprocessing error: {e}")
    
    def predict(self, image_tensor: torch.Tensor) -> Tuple[str, float]:
        """Predict text using CRNN"""
        try:
            with torch.no_grad():
                outputs = self.model(image_tensor)
                
                # CTC decoding (simplified)
                predicted_ids = torch.argmax(outputs, dim=2)
                
                # Decode to text
                text = self._decode_predictions(predicted_ids[0])
                
                # Calculate confidence (simplified)
                confidence = torch.softmax(outputs, dim=2).max().item()
                
                return text, confidence
                
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return "", 0.0
    
    def _decode_predictions(self, predicted_ids):
        """Decode predicted character IDs to text"""
        chars = self.vocab['chars']
        blank_token = self.vocab['blank_token']
        
        # Remove blanks and consecutive duplicates
        result = []
        prev_char = None
        
        for char_id in predicted_ids:
            char_id = char_id.item()
            if char_id != blank_token and char_id != prev_char:
                if char_id < len(chars):
                    result.append(chars[char_id])
            prev_char = char_id
        
        return ''.join(result)


class EasyOCRModel(BaseOCRModel):
    """EasyOCR wrapper for custom OCR"""
    
    def load_model(self):
        """Load EasyOCR model"""
        try:
            import easyocr
            
            # Initialize EasyOCR reader
            languages = self.config.model_settings.get('languages', ['en'])
            gpu = self.device == 'cuda'
            
            self.model = easyocr.Reader(
                languages,
                gpu=gpu,
                model_storage_directory=self.config.model_settings.get('model_dir', None)
            )
            
            self.logger.info(f"Loaded EasyOCR model with languages: {languages}")
            
        except ImportError:
            raise ProcessingError("EasyOCR library not installed")
        except Exception as e:
            raise ProcessingError(f"Failed to load EasyOCR model: {e}")
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for EasyOCR"""
        return np.array(image)
    
    def predict(self, image_array: np.ndarray) -> Tuple[str, float]:
        """Predict text using EasyOCR"""
        try:
            results = self.model.readtext(image_array)
            
            if results:
                # Combine all detected text
                texts = []
                confidences = []
                
                for (bbox, text, confidence) in results:
                    texts.append(text)
                    confidences.append(confidence)
                
                combined_text = ' '.join(texts)
                avg_confidence = sum(confidences) / len(confidences)
                
                return combined_text, avg_confidence
            else:
                return "", 0.0
                
        except Exception as e:
            self.logger.error(f"EasyOCR prediction failed: {e}")
            return "", 0.0


class PaddleOCRModel(BaseOCRModel):
    """PaddleOCR wrapper for custom OCR"""
    
    def load_model(self):
        """Load PaddleOCR model"""
        try:
            from paddleocr import PaddleOCR
            
            # Initialize PaddleOCR
            lang = self.config.model_settings.get('lang', 'en')
            use_gpu = self.device == 'cuda'
            
            self.model = PaddleOCR(
                lang=lang,
                use_gpu=use_gpu,
                show_log=False
            )
            
            self.logger.info(f"Loaded PaddleOCR model with language: {lang}")
            
        except ImportError:
            raise ProcessingError("PaddleOCR library not installed")
        except Exception as e:
            raise ProcessingError(f"Failed to load PaddleOCR model: {e}")
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for PaddleOCR"""
        return np.array(image)
    
    def predict(self, image_array: np.ndarray) -> Tuple[str, float]:
        """Predict text using PaddleOCR"""
        try:
            results = self.model.ocr(image_array)
            
            if results and results[0]:
                # Combine all detected text
                texts = []
                confidences = []
                
                for line in results[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    texts.append(text)
                    confidences.append(confidence)
                
                combined_text = ' '.join(texts)
                avg_confidence = sum(confidences) / len(confidences)
                
                return combined_text, avg_confidence
            else:
                return "", 0.0
                
        except Exception as e:
            self.logger.error(f"PaddleOCR prediction failed: {e}")
            return "", 0.0


class CustomOCREngine:
    """
    Custom OCR Engine supporting multiple architectures
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.config = CustomOCRConfig(**(config or {}))
        
        # Initialize the specific model
        self.model = self._create_model()
        self.model.load_model()
        
        self.logger.info(f"Custom OCR engine initialized with architecture: {self.config.architecture.value}")
    
    def _create_model(self) -> BaseOCRModel:
        """Create the appropriate model based on architecture"""
        architecture_map = {
            ModelArchitecture.CRNN: CRNNModel,
            ModelArchitecture.EASY_OCR: EasyOCRModel,
            ModelArchitecture.PADDLE_OCR: PaddleOCRModel,
        }
        
        model_class = architecture_map.get(self.config.architecture)
        if not model_class:
            raise ProcessingError(f"Unsupported architecture: {self.config.architecture}")
        
        return model_class(self.config)
    
    def extract_text(self, image: Image.Image) -> CustomOCREngineResult:
        """Extract text from image using custom OCR model"""
        start_time = time.time()
        
        try:
            # Extract text using the model
            text, confidence = self.model.extract_text(image)
            
            # Create simplified word and line objects
            words = []
            lines = []
            
            if text:
                # Split text into words (simplified)
                word_parts = text.split()
                word_width = image.width // max(len(word_parts), 1)
                
                for word_idx, word_text in enumerate(word_parts):
                    word_bbox = BoundingBox(
                        x=word_idx * word_width,
                        y=0,
                        width=word_width,
                        height=image.height
                    )
                    
                    word = OCRWord(
                        text=word_text,
                        confidence=confidence * 100,
                        bbox=word_bbox,
                        page_num=0,
                        line_num=0,
                        word_num=word_idx
                    )
                    words.append(word)
                
                # Create single line object
                line_bbox = BoundingBox(0, 0, image.width, image.height)
                line = OCRLine(
                    text=text,
                    confidence=confidence * 100,
                    bbox=line_bbox,
                    words=words,
                    page_num=0,
                    line_num=0
                )
                lines.append(line)
            
            processing_time = time.time() - start_time
            
            result = CustomOCREngineResult(
                text=text,
                confidence=confidence,
                words=words,
                lines=lines,
                processing_time=processing_time,
                model_info={
                    'architecture': self.config.architecture.value,
                    'model_path': self.config.model_path,
                    'device': self.model.device
                }
            )
            
            self.logger.debug(f"Custom OCR extraction completed in {processing_time:.2f}s, confidence: {confidence:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Custom OCR text extraction failed: {e}")
            raise ProcessingError(f"Custom OCR extraction error: {e}")
    
    def extract_text_batch(self, images: List[Image.Image]) -> List[CustomOCREngineResult]:
        """Extract text from multiple images"""
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.extract_text(image)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch processing failed for image {i}: {e}")
                # Add empty result
                results.append(CustomOCREngineResult(
                    text="",
                    confidence=0.0,
                    words=[],
                    lines=[],
                    processing_time=0.0
                ))
        
        return results
    
    def get_supported_architectures(self) -> List[str]:
        """Get list of supported architectures"""
        return [arch.value for arch in ModelArchitecture]
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get custom OCR engine information"""
        try:
            info = {
                'engine_name': 'CustomOCR',
                'architecture': self.config.architecture.value,
                'model_path': self.config.model_path,
                'device': self.model.device,
                'supported_architectures': self.get_supported_architectures(),
                'configuration': {
                    'batch_size': self.config.batch_size,
                    'image_height': self.config.image_height,
                    'image_width': self.config.image_width,
                    'max_text_length': self.config.max_text_length,
                    'normalize': self.config.normalize
                }
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting engine info: {e}")
            return {'engine_name': 'CustomOCR', 'error': str(e)}
    
    def cleanup(self):
        """Cleanup engine resources"""
        try:
            if hasattr(self.model, 'cleanup'):
                self.model.cleanup()
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.debug("Custom OCR engine cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")


# Factory functions for easy engine creation
def create_custom_ocr_engine(architecture: str = "crnn", **config_kwargs) -> CustomOCREngine:
    """
    Factory function to create custom OCR engine
    
    Args:
        architecture: Model architecture ('crnn', 'easy_ocr', 'paddle_ocr')
        **config_kwargs: Additional configuration parameters
    
    Returns:
        Configured CustomOCREngine instance
    """
    config = {
        'architecture': ModelArchitecture(architecture),
        **config_kwargs
    }
    
    return CustomOCREngine(config)


def create_easyocr_engine(languages: List[str] = None, **config_kwargs) -> CustomOCREngine:
    """Create EasyOCR engine"""
    languages = languages or ['en']
    
    config = {
        'architecture': ModelArchitecture.EASY_OCR,
        'model_settings': {'languages': languages},
        **config_kwargs
    }
    
    return CustomOCREngine(config)


def create_paddleocr_engine(lang: str = 'en', **config_kwargs) -> CustomOCREngine:
    """Create PaddleOCR engine"""
    config = {
        'architecture': ModelArchitecture.PADDLE_OCR,
        'model_settings': {'lang': lang},
        **config_kwargs
    }
    
    return CustomOCREngine(config)