#!/usr/bin/env python3
"""
OCR Pipeline for document processing and text extraction
"""

import logging
import asyncio
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
import numpy as np
from pydantic import BaseModel, Field, validator

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError, ValidationError


class OCREngine(Enum):
    """Available OCR engines"""
    TESSERACT = "tesseract"
    TROCR = "trocr"
    CUSTOM = "custom"
    ENSEMBLE = "ensemble"


class OCRConfidence(Enum):
    """OCR confidence levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class BoundingBox:
    """Bounding box coordinates"""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height
    
    def to_dict(self) -> Dict[str, int]:
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height
        }


@dataclass
class OCRWord:
    """Individual word detected by OCR"""
    text: str
    confidence: float
    bbox: BoundingBox
    page_num: int = 0
    line_num: int = 0
    word_num: int = 0


@dataclass
class OCRLine:
    """Line of text detected by OCR"""
    text: str
    confidence: float
    bbox: BoundingBox
    words: List[OCRWord] = field(default_factory=list)
    page_num: int = 0
    line_num: int = 0


@dataclass
class OCRPage:
    """Page of OCR results"""
    page_num: int
    width: int
    height: int
    text: str
    confidence: float
    lines: List[OCRLine] = field(default_factory=list)
    words: List[OCRWord] = field(default_factory=list)
    processing_time: float = 0.0


class OCRResult(BaseModel):
    """Complete OCR processing result"""
    
    document_id: str = Field(..., description="Unique document identifier")
    file_path: str = Field(..., description="Source file path")
    total_pages: int = Field(..., description="Total number of pages")
    pages: List[OCRPage] = Field(default_factory=list, description="OCR results per page")
    engine_used: str = Field(..., description="OCR engine used")
    processing_time: float = Field(0.0, description="Total processing time in seconds")
    confidence_score: float = Field(0.0, description="Overall confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('confidence_score')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence score must be between 0 and 1')
        return v


class OCRConfig(BaseModel):
    """OCR pipeline configuration"""
    
    engine: OCREngine = Field(OCREngine.TESSERACT, description="Primary OCR engine")
    fallback_engines: List[OCREngine] = Field(default_factory=list, description="Fallback engines")
    confidence_threshold: float = Field(0.7, description="Minimum confidence threshold")
    max_workers: int = Field(4, description="Maximum parallel workers")
    preprocessing_enabled: bool = Field(True, description="Enable image preprocessing")
    postprocessing_enabled: bool = Field(True, description="Enable text postprocessing")
    language_codes: List[str] = Field(default_factory=lambda: ['eng'], description="OCR language codes")
    dpi: int = Field(300, description="Target DPI for processing")
    
    # Engine-specific configurations
    tesseract_config: Dict[str, Any] = Field(default_factory=dict)
    trocr_config: Dict[str, Any] = Field(default_factory=dict)
    custom_config: Dict[str, Any] = Field(default_factory=dict)


class OCRPipeline:
    """
    Main OCR Pipeline for processing documents with multiple engines
    """
    
    def __init__(self, config: Optional[OCRConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or OCRConfig()
        
        # Initialize engines
        self.engines = {}
        self._initialize_engines()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        self.logger.info(f"OCR Pipeline initialized with engine: {self.config.engine.value}")
    
    def _initialize_engines(self):
        """Initialize OCR engines"""
        try:
            if self.config.engine == OCREngine.TESSERACT:
                from .tesseract_engine import TesseractEngine
                self.engines[OCREngine.TESSERACT] = TesseractEngine(self.config.tesseract_config)
            
            if self.config.engine == OCREngine.TROCR:
                from .trocr_engine import TrOCREngine
                self.engines[OCREngine.TROCR] = TrOCREngine(self.config.trocr_config)
            
            if self.config.engine == OCREngine.CUSTOM:
                from .custom_ocr_models import CustomOCREngine
                self.engines[OCREngine.CUSTOM] = CustomOCREngine(self.config.custom_config)
            
            # Initialize fallback engines
            for engine_type in self.config.fallback_engines:
                if engine_type not in self.engines:
                    if engine_type == OCREngine.TESSERACT:
                        from .tesseract_engine import TesseractEngine
                        self.engines[engine_type] = TesseractEngine(self.config.tesseract_config)
                    elif engine_type == OCREngine.TROCR:
                        from .trocr_engine import TrOCREngine
                        self.engines[engine_type] = TrOCREngine(self.config.trocr_config)
                    elif engine_type == OCREngine.CUSTOM:
                        from .custom_ocr_models import CustomOCREngine
                        self.engines[engine_type] = CustomOCREngine(self.config.custom_config)
            
        except ImportError as e:
            self.logger.error(f"Failed to initialize OCR engine: {e}")
            raise ProcessingError(f"OCR engine initialization failed: {e}")
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results
        """
        if not self.config.preprocessing_enabled:
            return image
        
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if DPI is specified
            if self.config.dpi != 300:
                current_dpi = image.info.get('dpi', (72, 72))[0]
                scale_factor = self.config.dpi / current_dpi
                new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array for processing
            img_array = np.array(image)
            
            # Basic image enhancement
            # Increase contrast slightly
            img_array = np.clip(img_array * 1.1, 0, 255).astype(np.uint8)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(img_array)
            
            return processed_image
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def postprocess_text(self, text: str) -> str:
        """
        Postprocess extracted text
        """
        if not self.config.postprocessing_enabled:
            return text
        
        try:
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Fix common OCR errors
            corrections = {
                '0': 'O',  # Common digit/letter confusion
                '1': 'I',
                '5': 'S',
                '@': 'a',
                '|': 'l',
                '¢': 'c',
                '§': 's'
            }
            
            # Apply corrections selectively (only for obvious errors)
            for wrong, correct in corrections.items():
                # Only replace if it creates a valid word context
                text = text.replace(f' {wrong} ', f' {correct} ')
            
            return text
            
        except Exception as e:
            self.logger.warning(f"Text postprocessing failed: {e}")
            return text
    
    def process_single_page(self, image: Image.Image, page_num: int = 0) -> OCRPage:
        """
        Process a single page/image
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Get primary engine
            primary_engine = self.engines[self.config.engine]
            
            # Process with primary engine
            result = primary_engine.extract_text(processed_image)
            
            # Check confidence and use fallback if needed
            if result.confidence < self.config.confidence_threshold and self.config.fallback_engines:
                self.logger.info(f"Primary engine confidence {result.confidence:.2f} below threshold, trying fallback")
                
                for fallback_engine_type in self.config.fallback_engines:
                    fallback_engine = self.engines[fallback_engine_type]
                    fallback_result = fallback_engine.extract_text(processed_image)
                    
                    if fallback_result.confidence > result.confidence:
                        self.logger.info(f"Fallback engine {fallback_engine_type.value} improved confidence: {fallback_result.confidence:.2f}")
                        result = fallback_result
                        break
            
            # Postprocess text
            processed_text = self.postprocess_text(result.text)
            
            # Create OCR page result
            processing_time = time.time() - start_time
            
            ocr_page = OCRPage(
                page_num=page_num,
                width=image.width,
                height=image.height,
                text=processed_text,
                confidence=result.confidence,
                lines=result.lines if hasattr(result, 'lines') else [],
                words=result.words if hasattr(result, 'words') else [],
                processing_time=processing_time
            )
            
            return ocr_page
            
        except Exception as e:
            self.logger.error(f"Error processing page {page_num}: {e}")
            raise ProcessingError(f"Page processing failed: {e}")
    
    def process_document(self, file_path: Union[str, Path], document_id: Optional[str] = None) -> OCRResult:
        """
        Process a complete document (single or multi-page)
        """
        file_path = Path(file_path)
        document_id = document_id or f"doc_{int(time.time())}"
        
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")
        
        start_time = time.time()
        
        try:
            # Load document pages
            pages_data = self._load_document_pages(file_path)
            
            # Process pages in parallel
            ocr_pages = []
            if len(pages_data) == 1:
                # Single page - process directly
                ocr_page = self.process_single_page(pages_data[0], 0)
                ocr_pages.append(ocr_page)
            else:
                # Multi-page - process in parallel
                futures = []
                for i, page_image in enumerate(pages_data):
                    future = self.executor.submit(self.process_single_page, page_image, i)
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        ocr_page = future.result()
                        ocr_pages.append(ocr_page)
                    except Exception as e:
                        self.logger.error(f"Page processing failed: {e}")
                        continue
                
                # Sort pages by page number
                ocr_pages.sort(key=lambda x: x.page_num)
            
            # Calculate overall confidence
            if ocr_pages:
                overall_confidence = sum(page.confidence for page in ocr_pages) / len(ocr_pages)
            else:
                overall_confidence = 0.0
            
            processing_time = time.time() - start_time
            
            # Create final result
            result = OCRResult(
                document_id=document_id,
                file_path=str(file_path),
                total_pages=len(ocr_pages),
                pages=ocr_pages,
                engine_used=self.config.engine.value,
                processing_time=processing_time,
                confidence_score=overall_confidence,
                metadata={
                    'language_codes': self.config.language_codes,
                    'dpi': self.config.dpi,
                    'preprocessing_enabled': self.config.preprocessing_enabled,
                    'postprocessing_enabled': self.config.postprocessing_enabled
                }
            )
            
            self.logger.info(f"Document processed successfully: {document_id} ({len(ocr_pages)} pages, {processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            raise ProcessingError(f"Document processing failed: {e}")
    
    def _load_document_pages(self, file_path: Path) -> List[Image.Image]:
        """
        Load document pages as PIL Images
        """
        try:
            suffix = file_path.suffix.lower()
            
            if suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                # Single image file
                image = Image.open(file_path)
                return [image]
            
            elif suffix == '.pdf':
                # PDF file - requires PDF processing
                try:
                    import pdf2image
                    images = pdf2image.convert_from_path(
                        file_path,
                        dpi=self.config.dpi,
                        fmt='RGB'
                    )
                    return images
                except ImportError:
                    raise ProcessingError("pdf2image library required for PDF processing")
            
            else:
                raise ValidationError(f"Unsupported file format: {suffix}")
                
        except Exception as e:
            self.logger.error(f"Failed to load document pages: {e}")
            raise ProcessingError(f"Document loading failed: {e}")
    
    async def process_document_async(self, file_path: Union[str, Path], document_id: Optional[str] = None) -> OCRResult:
        """
        Asynchronous document processing
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.process_document,
            file_path,
            document_id
        )
        return result
    
    def process_batch(self, file_paths: List[Union[str, Path]], 
                     document_ids: Optional[List[str]] = None) -> List[OCRResult]:
        """
        Process multiple documents in batch
        """
        if document_ids and len(document_ids) != len(file_paths):
            raise ValidationError("Number of document IDs must match number of file paths")
        
        results = []
        futures = []
        
        # Submit all jobs
        for i, file_path in enumerate(file_paths):
            doc_id = document_ids[i] if document_ids else None
            future = self.executor.submit(self.process_document, file_path, doc_id)
            futures.append((future, file_path))
        
        # Collect results
        for future, file_path in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch processing failed for {file_path}: {e}")
                continue
        
        self.logger.info(f"Batch processing completed: {len(results)}/{len(file_paths)} documents processed")
        
        return results
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats
        """
        return ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about configured engines
        """
        info = {
            'primary_engine': self.config.engine.value,
            'fallback_engines': [engine.value for engine in self.config.fallback_engines],
            'available_engines': list(self.engines.keys()),
            'configuration': {
                'confidence_threshold': self.config.confidence_threshold,
                'max_workers': self.config.max_workers,
                'preprocessing_enabled': self.config.preprocessing_enabled,
                'postprocessing_enabled': self.config.postprocessing_enabled,
                'language_codes': self.config.language_codes,
                'dpi': self.config.dpi
            }
        }
        return info
    
    def cleanup(self):
        """
        Cleanup resources
        """
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        # Cleanup engines
        for engine in self.engines.values():
            if hasattr(engine, 'cleanup'):
                engine.cleanup()
        
        self.logger.info("OCR Pipeline cleanup completed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Factory function for easy pipeline creation
def create_ocr_pipeline(engine: str = "tesseract", **config_kwargs) -> OCRPipeline:
    """
    Factory function to create OCR pipeline with specific engine
    
    Args:
        engine: OCR engine name ('tesseract', 'trocr', 'custom', 'ensemble')
        **config_kwargs: Additional configuration parameters
    
    Returns:
        Configured OCRPipeline instance
    """
    config = OCRConfig(
        engine=OCREngine(engine),
        **config_kwargs
    )
    
    return OCRPipeline(config)