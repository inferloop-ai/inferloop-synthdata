#!/usr/bin/env python3
"""
Tesseract OCR Engine implementation
"""

import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import json
import time

from PIL import Image
import numpy as np
from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError, ValidationError
from .ocr_pipeline import BoundingBox, OCRWord, OCRLine, OCRPage


@dataclass
class TesseractEngineResult:
    """Tesseract OCR extraction result"""
    text: str
    confidence: float
    words: List[OCRWord]
    lines: List[OCRLine]
    processing_time: float = 0.0


class TesseractConfig(BaseModel):
    """Tesseract engine configuration"""
    
    tesseract_cmd: str = Field("tesseract", description="Tesseract command path")
    language: str = Field("eng", description="OCR language code")
    oem: int = Field(3, description="OCR Engine Mode (0-3)")
    psm: int = Field(6, description="Page Segmentation Mode (0-13)")
    config_string: str = Field("", description="Additional Tesseract config")
    preserve_interword_spaces: bool = Field(True, description="Preserve spaces between words")
    timeout: int = Field(300, description="Processing timeout in seconds")
    
    # Advanced options
    user_words_file: Optional[str] = Field(None, description="Custom words file path")
    user_patterns_file: Optional[str] = Field(None, description="Custom patterns file path")
    tessdata_dir: Optional[str] = Field(None, description="Custom tessdata directory")


class TesseractEngine:
    """
    Tesseract OCR Engine for text extraction from images
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.config = TesseractConfig(**(config or {}))
        
        # Validate Tesseract installation
        self._validate_tesseract()
        
        self.logger.info(f"Tesseract engine initialized with language: {self.config.language}")
    
    def _validate_tesseract(self):
        """Validate Tesseract installation and configuration"""
        try:
            # Check if Tesseract is installed
            result = subprocess.run(
                [self.config.tesseract_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                raise ProcessingError(f"Tesseract validation failed: {result.stderr}")
            
            version_info = result.stdout.split('\n')[0]
            self.logger.info(f"Tesseract version: {version_info}")
            
            # Check if language data is available
            lang_result = subprocess.run(
                [self.config.tesseract_cmd, "--list-langs"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if lang_result.returncode == 0:
                available_languages = lang_result.stdout.strip().split('\n')[1:]  # Skip header
                if self.config.language not in available_languages:
                    self.logger.warning(f"Language {self.config.language} may not be available. Available: {available_languages}")
            
        except subprocess.TimeoutExpired:
            raise ProcessingError("Tesseract validation timeout")
        except FileNotFoundError:
            raise ProcessingError(f"Tesseract not found at: {self.config.tesseract_cmd}")
        except Exception as e:
            raise ProcessingError(f"Tesseract validation error: {e}")
    
    def _prepare_tesseract_command(self, temp_image_path: str, output_format: str = "txt") -> List[str]:
        """Prepare Tesseract command with all options"""
        cmd = [
            self.config.tesseract_cmd,
            temp_image_path,
            "stdout"  # Output to stdout
        ]
        
        # Language
        cmd.extend(["-l", self.config.language])
        
        # OCR Engine Mode
        cmd.extend(["--oem", str(self.config.oem)])
        
        # Page Segmentation Mode
        cmd.extend(["--psm", str(self.config.psm)])
        
        # Custom tessdata directory
        if self.config.tessdata_dir:
            cmd.extend(["--tessdata-dir", self.config.tessdata_dir])
        
        # User words file
        if self.config.user_words_file and os.path.exists(self.config.user_words_file):
            cmd.extend(["-c", f"user_words_file={self.config.user_words_file}"])
        
        # User patterns file
        if self.config.user_patterns_file and os.path.exists(self.config.user_patterns_file):
            cmd.extend(["-c", f"user_patterns_file={self.config.user_patterns_file}"])
        
        # Preserve interword spaces
        if self.config.preserve_interword_spaces:
            cmd.extend(["-c", "preserve_interword_spaces=1"])
        
        # Additional config string
        if self.config.config_string:
            cmd.extend(["-c", self.config.config_string])
        
        return cmd
    
    def extract_text(self, image: Image.Image) -> TesseractEngineResult:
        """
        Extract text from image using Tesseract OCR
        """
        start_time = time.time()
        
        try:
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Convert image to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Save image
                image.save(temp_path, 'PNG')
            
            try:
                # Extract plain text
                text_result = self._extract_plain_text(temp_path)
                
                # Extract detailed information (words, boxes, confidences)
                detailed_result = self._extract_detailed_info(temp_path)
                
                # Combine results
                words, lines = detailed_result
                
                # Calculate overall confidence
                if words:
                    overall_confidence = sum(word.confidence for word in words) / len(words)
                else:
                    overall_confidence = 0.0
                
                processing_time = time.time() - start_time
                
                result = TesseractEngineResult(
                    text=text_result,
                    confidence=overall_confidence / 100.0,  # Convert to 0-1 range
                    words=words,
                    lines=lines,
                    processing_time=processing_time
                )
                
                self.logger.debug(f"Tesseract extraction completed in {processing_time:.2f}s, confidence: {result.confidence:.2f}")
                
                return result
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                
        except Exception as e:
            self.logger.error(f"Tesseract text extraction failed: {e}")
            raise ProcessingError(f"Tesseract extraction error: {e}")
    
    def _extract_plain_text(self, image_path: str) -> str:
        """Extract plain text using Tesseract"""
        try:
            cmd = self._prepare_tesseract_command(image_path, "txt")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            if result.returncode != 0:
                raise ProcessingError(f"Tesseract text extraction failed: {result.stderr}")
            
            return result.stdout.strip()
            
        except subprocess.TimeoutExpired:
            raise ProcessingError("Tesseract text extraction timeout")
        except Exception as e:
            raise ProcessingError(f"Text extraction error: {e}")
    
    def _extract_detailed_info(self, image_path: str) -> Tuple[List[OCRWord], List[OCRLine]]:
        """Extract detailed word and line information with bounding boxes"""
        try:
            # Get TSV output with bounding boxes and confidence
            cmd = self._prepare_tesseract_command(image_path, "tsv")
            cmd.append("tsv")  # Output format
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            if result.returncode != 0:
                self.logger.warning(f"Tesseract detailed extraction warning: {result.stderr}")
                return [], []
            
            return self._parse_tsv_output(result.stdout)
            
        except subprocess.TimeoutExpired:
            raise ProcessingError("Tesseract detailed extraction timeout")
        except Exception as e:
            self.logger.warning(f"Detailed extraction error: {e}")
            return [], []
    
    def _parse_tsv_output(self, tsv_output: str) -> Tuple[List[OCRWord], List[OCRLine]]:
        """Parse Tesseract TSV output to extract words and lines with bounding boxes"""
        words = []
        lines = []
        current_line_words = []
        current_line_num = -1
        
        try:
            lines_data = tsv_output.strip().split('\n')
            if len(lines_data) < 2:  # No data beyond header
                return [], []
            
            # Skip header line
            for line in lines_data[1:]:
                parts = line.split('\t')
                
                if len(parts) < 12:
                    continue
                
                try:
                    level = int(parts[0])
                    page_num = int(parts[1])
                    block_num = int(parts[2])
                    par_num = int(parts[3])
                    line_num = int(parts[4])
                    word_num = int(parts[5])
                    left = int(parts[6])
                    top = int(parts[7])
                    width = int(parts[8])
                    height = int(parts[9])
                    conf = int(parts[10])
                    text = parts[11] if len(parts) > 11 else ""
                    
                    # Skip empty text or very low confidence
                    if not text.strip() or conf < 0:
                        continue
                    
                    bbox = BoundingBox(x=left, y=top, width=width, height=height)
                    
                    # Level 5 is word level
                    if level == 5 and text.strip():
                        word = OCRWord(
                            text=text,
                            confidence=float(conf),
                            bbox=bbox,
                            page_num=page_num,
                            line_num=line_num,
                            word_num=word_num
                        )
                        words.append(word)
                        
                        # Group words by line
                        if line_num != current_line_num:
                            # Save previous line if exists
                            if current_line_words and current_line_num >= 0:
                                line_text = ' '.join(w.text for w in current_line_words)
                                line_confidence = sum(w.confidence for w in current_line_words) / len(current_line_words)
                                line_bbox = self._calculate_line_bbox(current_line_words)
                                
                                line_obj = OCRLine(
                                    text=line_text,
                                    confidence=line_confidence,
                                    bbox=line_bbox,
                                    words=current_line_words.copy(),
                                    page_num=page_num,
                                    line_num=current_line_num
                                )
                                lines.append(line_obj)
                            
                            # Start new line
                            current_line_words = [word]
                            current_line_num = line_num
                        else:
                            current_line_words.append(word)
                
                except (ValueError, IndexError) as e:
                    self.logger.debug(f"Error parsing TSV line: {line}, error: {e}")
                    continue
            
            # Handle last line
            if current_line_words and current_line_num >= 0:
                line_text = ' '.join(w.text for w in current_line_words)
                line_confidence = sum(w.confidence for w in current_line_words) / len(current_line_words)
                line_bbox = self._calculate_line_bbox(current_line_words)
                
                line_obj = OCRLine(
                    text=line_text,
                    confidence=line_confidence,
                    bbox=line_bbox,
                    words=current_line_words.copy(),
                    page_num=0,
                    line_num=current_line_num
                )
                lines.append(line_obj)
            
            return words, lines
            
        except Exception as e:
            self.logger.error(f"Error parsing TSV output: {e}")
            return [], []
    
    def _calculate_line_bbox(self, words: List[OCRWord]) -> BoundingBox:
        """Calculate bounding box for a line based on constituent words"""
        if not words:
            return BoundingBox(0, 0, 0, 0)
        
        min_x = min(word.bbox.x for word in words)
        min_y = min(word.bbox.y for word in words)
        max_x = max(word.bbox.x + word.bbox.width for word in words)
        max_y = max(word.bbox.y + word.bbox.height for word in words)
        
        return BoundingBox(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y
        )
    
    def extract_text_with_layout(self, image: Image.Image) -> TesseractEngineResult:
        """
        Extract text while preserving layout information
        """
        # Use PSM 6 (uniform block of text) for better layout preservation
        original_psm = self.config.psm
        self.config.psm = 6
        
        try:
            result = self.extract_text(image)
            return result
        finally:
            # Restore original PSM
            self.config.psm = original_psm
    
    def extract_text_oriented(self, image: Image.Image) -> TesseractEngineResult:
        """
        Extract text from images with automatic orientation detection
        """
        # Use PSM 1 (automatic page segmentation with OSD)
        original_psm = self.config.psm
        self.config.psm = 1
        
        try:
            result = self.extract_text(image)
            return result
        finally:
            # Restore original PSM
            self.config.psm = original_psm
    
    def get_available_languages(self) -> List[str]:
        """Get list of available Tesseract languages"""
        try:
            result = subprocess.run(
                [self.config.tesseract_cmd, "--list-langs"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[1:]  # Skip header
            else:
                return []
                
        except Exception as e:
            self.logger.warning(f"Could not get available languages: {e}")
            return []
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get Tesseract engine information"""
        try:
            # Get version
            version_result = subprocess.run(
                [self.config.tesseract_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            version = "Unknown"
            if version_result.returncode == 0:
                version = version_result.stdout.split('\n')[0]
            
            # Get available languages
            languages = self.get_available_languages()
            
            info = {
                'engine_name': 'Tesseract',
                'version': version,
                'available_languages': languages,
                'current_language': self.config.language,
                'configuration': {
                    'oem': self.config.oem,
                    'psm': self.config.psm,
                    'timeout': self.config.timeout,
                    'preserve_interword_spaces': self.config.preserve_interword_spaces
                }
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting engine info: {e}")
            return {'engine_name': 'Tesseract', 'error': str(e)}
    
    def cleanup(self):
        """Cleanup engine resources"""
        # Tesseract doesn't require explicit cleanup
        self.logger.debug("Tesseract engine cleanup completed")


# Factory function for easy engine creation
def create_tesseract_engine(language: str = "eng", **config_kwargs) -> TesseractEngine:
    """
    Factory function to create Tesseract engine
    
    Args:
        language: OCR language code (default: 'eng')
        **config_kwargs: Additional configuration parameters
    
    Returns:
        Configured TesseractEngine instance
    """
    config = {
        'language': language,
        **config_kwargs
    }
    
    return TesseractEngine(config)