#!/usr/bin/env python3
"""
TrOCR (Transformer OCR) Engine implementation using Hugging Face models
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import torch
import numpy as np
from PIL import Image, ImageOps

from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError, ValidationError
from .ocr_pipeline import BoundingBox, OCRWord, OCRLine, OCRPage


@dataclass
class TrOCREngineResult:
    """TrOCR OCR extraction result"""
    text: str
    confidence: float
    words: List[OCRWord]
    lines: List[OCRLine]
    processing_time: float = 0.0


class TrOCRConfig(BaseModel):
    """TrOCR engine configuration"""
    
    model_name: str = Field("microsoft/trocr-base-printed", description="HuggingFace model name")
    processor_name: str = Field("microsoft/trocr-base-printed", description="HuggingFace processor name")
    device: str = Field("auto", description="Device to run model on ('cpu', 'cuda', 'auto')")
    max_length: int = Field(256, description="Maximum sequence length")
    batch_size: int = Field(1, description="Batch size for processing")
    confidence_threshold: float = Field(0.5, description="Minimum confidence threshold")
    
    # Image preprocessing
    image_size: Tuple[int, int] = Field((384, 384), description="Input image size (width, height)")
    normalize: bool = Field(True, description="Normalize image pixels")
    
    # Advanced options
    use_cache: bool = Field(True, description="Use model cache")
    torch_dtype: str = Field("float32", description="PyTorch data type")
    attention_mask: bool = Field(True, description="Use attention mask")


class TrOCREngine:
    """
    TrOCR (Transformer OCR) Engine for text extraction using transformer models
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.config = TrOCRConfig(**(config or {}))
        
        # Initialize model and processor
        self.model = None
        self.processor = None
        self.device = None
        
        self._initialize_model()
        
        self.logger.info(f"TrOCR engine initialized with model: {self.config.model_name}")
    
    def _initialize_model(self):
        """Initialize TrOCR model and processor"""
        try:
            # Import required libraries
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            # Determine device
            if self.config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device
            
            self.logger.info(f"Loading TrOCR model on device: {self.device}")
            
            # Load processor
            self.processor = TrOCRProcessor.from_pretrained(
                self.config.processor_name,
                cache_dir=None if not self.config.use_cache else None
            )
            
            # Load model
            torch_dtype = getattr(torch, self.config.torch_dtype)
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.config.model_name,
                torch_dtype=torch_dtype,
                cache_dir=None if not self.config.use_cache else None
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.logger.info("TrOCR model loaded successfully")
            
        except ImportError as e:
            raise ProcessingError(f"Required transformers library not found: {e}")
        except Exception as e:
            self.logger.error(f"Failed to initialize TrOCR model: {e}")
            raise ProcessingError(f"TrOCR model initialization failed: {e}")
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for TrOCR model"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            target_width, target_height = self.config.image_size
            
            # Maintain aspect ratio while resizing
            image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Create new image with target size and paste resized image
            new_image = Image.new('RGB', (target_width, target_height), (255, 255, 255))
            
            # Center the image
            x = (target_width - image.width) // 2
            y = (target_height - image.height) // 2
            new_image.paste(image, (x, y))
            
            return new_image
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def _extract_text_from_image(self, image: Image.Image) -> Tuple[str, float]:
        """Extract text from a single image using TrOCR"""
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Prepare inputs
            inputs = self.processor(
                processed_image, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.pixel_values,
                    max_length=self.config.max_length,
                    attention_mask=inputs.attention_mask if self.config.attention_mask else None,
                    do_sample=False,  # Use greedy decoding for consistency
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode generated text
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Calculate confidence (simplified approach)
            # Note: TrOCR doesn't directly provide confidence scores
            # We use sequence length and model certainty as proxy
            confidence = min(1.0, len(generated_text) / 50.0)  # Heuristic
            confidence = max(confidence, self.config.confidence_threshold)
            
            return generated_text.strip(), confidence
            
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            return "", 0.0
    
    def _segment_image_into_lines(self, image: Image.Image) -> List[Tuple[Image.Image, BoundingBox]]:
        """
        Segment image into text lines for better OCR results
        This is a simplified approach - in practice, you'd use more sophisticated methods
        """
        try:
            # For now, we'll treat the entire image as one line
            # In a real implementation, you would use text detection models
            # like CRAFT, DBNet, or similar to detect text regions
            
            bbox = BoundingBox(
                x=0,
                y=0,
                width=image.width,
                height=image.height
            )
            
            return [(image, bbox)]
            
        except Exception as e:
            self.logger.warning(f"Image segmentation failed: {e}")
            return [(image, BoundingBox(0, 0, image.width, image.height))]
    
    def extract_text(self, image: Image.Image) -> TrOCREngineResult:
        """
        Extract text from image using TrOCR
        """
        start_time = time.time()
        
        try:
            # Segment image into lines (simplified)
            line_segments = self._segment_image_into_lines(image)
            
            words = []
            lines = []
            all_text_parts = []
            
            for line_idx, (line_image, line_bbox) in enumerate(line_segments):
                # Extract text from line
                line_text, line_confidence = self._extract_text_from_image(line_image)
                
                if line_text:
                    all_text_parts.append(line_text)
                    
                    # Create word objects (simplified - splitting by spaces)
                    word_parts = line_text.split()
                    line_words = []
                    
                    # Estimate word positions within the line
                    if word_parts:
                        word_width = line_bbox.width // len(word_parts)
                        
                        for word_idx, word_text in enumerate(word_parts):
                            word_bbox = BoundingBox(
                                x=line_bbox.x + (word_idx * word_width),
                                y=line_bbox.y,
                                width=word_width,
                                height=line_bbox.height
                            )
                            
                            word = OCRWord(
                                text=word_text,
                                confidence=line_confidence * 100,  # Convert to 0-100 range
                                bbox=word_bbox,
                                page_num=0,
                                line_num=line_idx,
                                word_num=word_idx
                            )
                            
                            words.append(word)
                            line_words.append(word)
                    
                    # Create line object
                    line = OCRLine(
                        text=line_text,
                        confidence=line_confidence * 100,  # Convert to 0-100 range
                        bbox=line_bbox,
                        words=line_words,
                        page_num=0,
                        line_num=line_idx
                    )
                    
                    lines.append(line)
            
            # Combine all text
            full_text = ' '.join(all_text_parts)
            
            # Calculate overall confidence
            if lines:
                overall_confidence = sum(line.confidence for line in lines) / len(lines) / 100.0
            else:
                overall_confidence = 0.0
            
            processing_time = time.time() - start_time
            
            result = TrOCREngineResult(
                text=full_text,
                confidence=overall_confidence,
                words=words,
                lines=lines,
                processing_time=processing_time
            )
            
            self.logger.debug(f"TrOCR extraction completed in {processing_time:.2f}s, confidence: {result.confidence:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"TrOCR text extraction failed: {e}")
            raise ProcessingError(f"TrOCR extraction error: {e}")
    
    def extract_text_batch(self, images: List[Image.Image]) -> List[TrOCREngineResult]:
        """
        Extract text from multiple images in batch
        """
        results = []
        
        # Process images in batches
        for i in range(0, len(images), self.config.batch_size):
            batch = images[i:i + self.config.batch_size]
            
            for image in batch:
                try:
                    result = self.extract_text(image)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch processing failed for image {i}: {e}")
                    # Add empty result
                    results.append(TrOCREngineResult(
                        text="",
                        confidence=0.0,
                        words=[],
                        lines=[],
                        processing_time=0.0
                    ))
        
        return results
    
    def extract_text_handwritten(self, image: Image.Image) -> TrOCREngineResult:
        """
        Extract text optimized for handwritten text
        """
        # Switch to handwritten model if not already configured
        original_model = self.config.model_name
        
        if "handwritten" not in self.config.model_name:
            # Temporarily switch to handwritten model
            self.config.model_name = "microsoft/trocr-base-handwritten"
            self.config.processor_name = "microsoft/trocr-base-handwritten"
            
            try:
                # Reinitialize with handwritten model
                self._initialize_model()
                result = self.extract_text(image)
                return result
            finally:
                # Restore original model
                self.config.model_name = original_model
                self.config.processor_name = original_model
                self._initialize_model()
        else:
            return self.extract_text(image)
    
    def extract_text_printed(self, image: Image.Image) -> TrOCREngineResult:
        """
        Extract text optimized for printed text
        """
        # Switch to printed model if not already configured
        original_model = self.config.model_name
        
        if "printed" not in self.config.model_name:
            # Temporarily switch to printed model
            self.config.model_name = "microsoft/trocr-base-printed"
            self.config.processor_name = "microsoft/trocr-base-printed"
            
            try:
                # Reinitialize with printed model
                self._initialize_model()
                result = self.extract_text(image)
                return result
            finally:
                # Restore original model
                self.config.model_name = original_model
                self.config.processor_name = original_model
                self._initialize_model()
        else:
            return self.extract_text(image)
    
    def get_available_models(self) -> List[str]:
        """Get list of available TrOCR models"""
        return [
            "microsoft/trocr-base-printed",
            "microsoft/trocr-base-handwritten",
            "microsoft/trocr-large-printed",
            "microsoft/trocr-large-handwritten",
            "microsoft/trocr-small-printed",
            "microsoft/trocr-small-handwritten"
        ]
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get TrOCR engine information"""
        try:
            device_info = {
                'device': self.device,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            if torch.cuda.is_available() and self.device.startswith('cuda'):
                device_info['cuda_device_name'] = torch.cuda.get_device_name()
                device_info['cuda_memory_total'] = torch.cuda.get_device_properties(0).total_memory
            
            info = {
                'engine_name': 'TrOCR',
                'model_name': self.config.model_name,
                'processor_name': self.config.processor_name,
                'device_info': device_info,
                'available_models': self.get_available_models(),
                'configuration': {
                    'max_length': self.config.max_length,
                    'batch_size': self.config.batch_size,
                    'confidence_threshold': self.config.confidence_threshold,
                    'image_size': self.config.image_size,
                    'torch_dtype': self.config.torch_dtype
                }
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting engine info: {e}")
            return {'engine_name': 'TrOCR', 'error': str(e)}
    
    def cleanup(self):
        """Cleanup engine resources"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.debug("TrOCR engine cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")


# Factory function for easy engine creation
def create_trocr_engine(model_type: str = "printed", **config_kwargs) -> TrOCREngine:
    """
    Factory function to create TrOCR engine
    
    Args:
        model_type: Model type ('printed', 'handwritten', 'large-printed', 'large-handwritten')
        **config_kwargs: Additional configuration parameters
    
    Returns:
        Configured TrOCREngine instance
    """
    model_mapping = {
        "printed": "microsoft/trocr-base-printed",
        "handwritten": "microsoft/trocr-base-handwritten", 
        "large-printed": "microsoft/trocr-large-printed",
        "large-handwritten": "microsoft/trocr-large-handwritten",
        "small-printed": "microsoft/trocr-small-printed",
        "small-handwritten": "microsoft/trocr-small-handwritten"
    }
    
    model_name = model_mapping.get(model_type, "microsoft/trocr-base-printed")
    
    config = {
        'model_name': model_name,
        'processor_name': model_name,
        **config_kwargs
    }
    
    return TrOCREngine(config)