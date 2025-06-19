"""
OCR processing module for document text extraction
"""

from .ocr_pipeline import (
    OCRPipeline, OCRConfig, OCRResult, OCRPage, OCRLine, OCRWord, 
    BoundingBox, OCREngine, OCRConfidence, create_ocr_pipeline
)
from .tesseract_engine import TesseractEngine, TesseractConfig, create_tesseract_engine
from .trocr_engine import TrOCREngine, TrOCRConfig, create_trocr_engine
from .custom_ocr_models import (
    CustomOCREngine, CustomOCRConfig, ModelArchitecture,
    create_custom_ocr_engine, create_easyocr_engine, create_paddleocr_engine
)

__all__ = [
    # Pipeline
    'OCRPipeline',
    'OCRConfig', 
    'OCRResult',
    'OCRPage',
    'OCRLine', 
    'OCRWord',
    'BoundingBox',
    'OCREngine',
    'OCRConfidence',
    'create_ocr_pipeline',
    
    # Tesseract
    'TesseractEngine',
    'TesseractConfig',
    'create_tesseract_engine',
    
    # TrOCR
    'TrOCREngine', 
    'TrOCRConfig',
    'create_trocr_engine',
    
    # Custom OCR
    'CustomOCREngine',
    'CustomOCRConfig',
    'ModelArchitecture',
    'create_custom_ocr_engine',
    'create_easyocr_engine',
    'create_paddleocr_engine'
]