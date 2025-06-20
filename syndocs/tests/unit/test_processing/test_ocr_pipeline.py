"""
Unit tests for OCR processing pipeline.

Tests the OCR pipeline for text extraction from document images,
including preprocessing, recognition, and post-processing steps.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from PIL import Image
import io

from structured_docs_synth.processing.ocr.ocr_pipeline import (
    OCRPipeline,
    OCRConfig,
    OCRResult,
    TextBlock,
    OCREngine,
    PreprocessingStep
)
from structured_docs_synth.core.exceptions import ProcessingError


class TestOCRConfig:
    """Test OCR configuration."""
    
    def test_default_config(self):
        """Test default OCR configuration."""
        config = OCRConfig()
        
        assert config.engine == 'tesseract'
        assert config.language == 'eng'
        assert config.dpi == 300
        assert config.preprocessing_enabled is True
        assert config.confidence_threshold == 0.6
        assert config.enable_gpu is False
        assert config.max_image_size == (4096, 4096)
    
    def test_custom_config(self):
        """Test custom OCR configuration."""
        config = OCRConfig(
            engine='trocr',
            language='fra',
            dpi=600,
            enable_gpu=True,
            confidence_threshold=0.8
        )
        
        assert config.engine == 'trocr'
        assert config.language == 'fra'
        assert config.dpi == 600
        assert config.enable_gpu is True
        assert config.confidence_threshold == 0.8


class TestTextBlock:
    """Test TextBlock data structure."""
    
    def test_text_block_creation(self):
        """Test creating text block."""
        block = TextBlock(
            text="Hello World",
            bbox=(10, 20, 100, 40),
            confidence=0.95,
            line_number=1,
            word_number=1
        )
        
        assert block.text == "Hello World"
        assert block.bbox == (10, 20, 100, 40)
        assert block.confidence == 0.95
        assert block.line_number == 1
        assert block.word_number == 1
    
    def test_text_block_properties(self):
        """Test text block computed properties."""
        block = TextBlock(
            text="Test",
            bbox=(10, 20, 110, 50),
            confidence=0.9
        )
        
        assert block.width == 100
        assert block.height == 30
        assert block.center_x == 60
        assert block.center_y == 35
        assert block.area == 3000
    
    def test_text_block_overlap(self):
        """Test bounding box overlap calculation."""
        block1 = TextBlock("A", bbox=(0, 0, 100, 100))
        block2 = TextBlock("B", bbox=(50, 50, 150, 150))
        block3 = TextBlock("C", bbox=(200, 200, 300, 300))
        
        assert block1.overlaps_with(block2) is True
        assert block1.overlaps_with(block3) is False
        assert block2.overlaps_with(block3) is False


class TestOCRPipeline:
    """Test OCR pipeline functionality."""
    
    @pytest.fixture
    def ocr_config(self):
        """Provide OCR configuration."""
        return OCRConfig(
            engine='tesseract',
            language='eng',
            preprocessing_enabled=True
        )
    
    @pytest.fixture
    def ocr_pipeline(self, ocr_config):
        """Provide OCR pipeline instance."""
        return OCRPipeline(ocr_config)
    
    @pytest.fixture
    def sample_image(self):
        """Create sample test image."""
        # Create a simple white image with black text
        img = Image.new('RGB', (800, 600), color='white')
        return img
    
    def test_pipeline_initialization(self, ocr_pipeline, ocr_config):
        """Test pipeline initialization."""
        assert ocr_pipeline.config == ocr_config
        assert ocr_pipeline.engine is not None
        assert len(ocr_pipeline.preprocessing_steps) > 0
    
    @pytest.mark.asyncio
    async def test_process_image_success(self, ocr_pipeline, sample_image):
        """Test successful image processing."""
        # Mock OCR engine
        mock_result = OCRResult(
            text="Sample document text",
            blocks=[
                TextBlock("Sample", bbox=(10, 10, 100, 30), confidence=0.95),
                TextBlock("document", bbox=(110, 10, 200, 30), confidence=0.92),
                TextBlock("text", bbox=(210, 10, 250, 30), confidence=0.88)
            ],
            confidence=0.92,
            language='eng'
        )
        
        ocr_pipeline.engine.recognize = AsyncMock(return_value=mock_result)
        
        # Process image
        result = await ocr_pipeline.process_image(sample_image)
        
        assert result.text == "Sample document text"
        assert len(result.blocks) == 3
        assert result.confidence == 0.92
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_process_image_with_preprocessing(self, ocr_pipeline, sample_image):
        """Test image preprocessing steps."""
        # Mock preprocessing
        preprocessed_img = sample_image.copy()
        
        with patch.object(ocr_pipeline, '_preprocess_image') as mock_preprocess:
            mock_preprocess.return_value = preprocessed_img
            
            # Mock OCR
            ocr_pipeline.engine.recognize = AsyncMock(
                return_value=OCRResult("Preprocessed text", [], 0.9)
            )
            
            result = await ocr_pipeline.process_image(sample_image)
            
            mock_preprocess.assert_called_once()
            assert result.text == "Preprocessed text"
    
    @pytest.mark.asyncio
    async def test_process_pdf(self, ocr_pipeline, temp_dir):
        """Test PDF processing."""
        # Create mock PDF file
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 mock content")
        
        # Mock PDF to image conversion
        mock_images = [
            Image.new('RGB', (800, 600), 'white'),
            Image.new('RGB', (800, 600), 'white')
        ]
        
        with patch('pdf2image.convert_from_path', return_value=mock_images):
            # Mock OCR for each page
            ocr_pipeline.engine.recognize = AsyncMock(side_effect=[
                OCRResult("Page 1 text", [], 0.9),
                OCRResult("Page 2 text", [], 0.85)
            ])
            
            results = await ocr_pipeline.process_pdf(pdf_path)
            
            assert len(results) == 2
            assert results[0].text == "Page 1 text"
            assert results[1].text == "Page 2 text"
    
    @pytest.mark.asyncio
    async def test_batch_process(self, ocr_pipeline):
        """Test batch image processing."""
        images = [
            Image.new('RGB', (400, 300), 'white'),
            Image.new('RGB', (400, 300), 'white'),
            Image.new('RGB', (400, 300), 'white')
        ]
        
        # Mock OCR results
        ocr_pipeline.engine.recognize = AsyncMock(side_effect=[
            OCRResult(f"Text {i}", [], 0.9) for i in range(3)
        ])
        
        results = await ocr_pipeline.batch_process(images, max_concurrent=2)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        assert [r.result.text for r in results] == ["Text 0", "Text 1", "Text 2"]
    
    def test_preprocess_image_resize(self, ocr_pipeline):
        """Test image resizing preprocessing."""
        # Large image
        large_image = Image.new('RGB', (8000, 6000), 'white')
        
        processed = ocr_pipeline._preprocess_image(large_image)
        
        # Should be resized to max size
        assert processed.size[0] <= ocr_pipeline.config.max_image_size[0]
        assert processed.size[1] <= ocr_pipeline.config.max_image_size[1]
    
    def test_preprocess_image_dpi(self, ocr_pipeline):
        """Test DPI adjustment preprocessing."""
        # Low DPI image
        low_dpi_image = Image.new('RGB', (800, 600), 'white')
        low_dpi_image.info['dpi'] = (72, 72)
        
        processed = ocr_pipeline._preprocess_image(low_dpi_image)
        
        # Should be resized based on DPI
        expected_scale = ocr_pipeline.config.dpi / 72
        expected_size = (
            int(800 * expected_scale),
            int(600 * expected_scale)
        )
        assert processed.size == expected_size
    
    def test_preprocess_grayscale_conversion(self, ocr_pipeline):
        """Test grayscale conversion."""
        color_image = Image.new('RGB', (400, 300), (255, 0, 0))  # Red image
        
        ocr_pipeline.config.convert_to_grayscale = True
        processed = ocr_pipeline._preprocess_image(color_image)
        
        assert processed.mode == 'L'  # Grayscale
    
    @pytest.mark.asyncio
    async def test_text_detection_regions(self, ocr_pipeline):
        """Test text region detection."""
        image = Image.new('RGB', (800, 600), 'white')
        
        # Mock text detection
        mock_regions = [
            {'bbox': (10, 10, 200, 50), 'confidence': 0.9},
            {'bbox': (10, 100, 200, 150), 'confidence': 0.85}
        ]
        
        with patch.object(ocr_pipeline, '_detect_text_regions', return_value=mock_regions):
            regions = await ocr_pipeline.detect_text_regions(image)
            
            assert len(regions) == 2
            assert regions[0]['confidence'] == 0.9
    
    @pytest.mark.asyncio
    async def test_apply_confidence_threshold(self, ocr_pipeline, sample_image):
        """Test confidence threshold filtering."""
        # Mock OCR with mixed confidence blocks
        mock_result = OCRResult(
            text="High confidence low confidence",
            blocks=[
                TextBlock("High", bbox=(0, 0, 50, 20), confidence=0.9),
                TextBlock("confidence", bbox=(60, 0, 150, 20), confidence=0.85),
                TextBlock("low", bbox=(160, 0, 200, 20), confidence=0.4),
                TextBlock("confidence", bbox=(210, 0, 300, 20), confidence=0.3)
            ],
            confidence=0.65
        )
        
        ocr_pipeline.engine.recognize = AsyncMock(return_value=mock_result)
        ocr_pipeline.config.confidence_threshold = 0.6
        
        result = await ocr_pipeline.process_image(sample_image)
        
        # Should filter out low confidence blocks
        assert len(result.blocks) == 2
        assert all(block.confidence >= 0.6 for block in result.blocks)
    
    @pytest.mark.asyncio
    async def test_language_detection(self, ocr_pipeline, sample_image):
        """Test automatic language detection."""
        ocr_pipeline.config.auto_detect_language = True
        
        # Mock language detection
        with patch.object(ocr_pipeline, '_detect_language', return_value='fra'):
            # Mock OCR
            ocr_pipeline.engine.recognize = AsyncMock(
                return_value=OCRResult("Texte français", [], 0.9, language='fra')
            )
            
            result = await ocr_pipeline.process_image(sample_image)
            
            assert result.language == 'fra'
    
    @pytest.mark.asyncio
    async def test_error_handling(self, ocr_pipeline, sample_image):
        """Test error handling in pipeline."""
        # Mock OCR engine failure
        ocr_pipeline.engine.recognize = AsyncMock(
            side_effect=Exception("OCR engine error")
        )
        
        with pytest.raises(ProcessingError, match="OCR processing failed"):
            await ocr_pipeline.process_image(sample_image)
    
    @pytest.mark.asyncio
    async def test_post_processing(self, ocr_pipeline, sample_image):
        """Test post-processing steps."""
        # Mock OCR result with issues
        mock_result = OCRResult(
            text="  Multiple   spaces  and\n\nextra lines  ",
            blocks=[
                TextBlock("Multiple", bbox=(0, 0, 80, 20), confidence=0.9),
                TextBlock("spaces", bbox=(120, 0, 180, 20), confidence=0.9)
            ],
            confidence=0.9
        )
        
        ocr_pipeline.engine.recognize = AsyncMock(return_value=mock_result)
        ocr_pipeline.config.enable_post_processing = True
        
        result = await ocr_pipeline.process_image(sample_image)
        
        # Text should be cleaned
        assert "  Multiple   spaces" not in result.text
        assert "\n\n" not in result.text
    
    def test_merge_adjacent_blocks(self, ocr_pipeline):
        """Test merging of adjacent text blocks."""
        blocks = [
            TextBlock("Hello", bbox=(0, 0, 50, 20), confidence=0.9),
            TextBlock("World", bbox=(55, 0, 100, 20), confidence=0.9),  # Adjacent
            TextBlock("Far", bbox=(200, 0, 230, 20), confidence=0.9),  # Not adjacent
        ]
        
        merged = ocr_pipeline._merge_adjacent_blocks(blocks, max_gap=10)
        
        assert len(merged) == 2
        assert merged[0].text == "Hello World"
        assert merged[1].text == "Far"
    
    @pytest.mark.asyncio
    async def test_table_detection(self, ocr_pipeline, sample_image):
        """Test table structure detection."""
        # Mock table detection
        mock_tables = [
            {
                'bbox': (50, 50, 500, 300),
                'rows': 5,
                'columns': 3,
                'cells': []
            }
        ]
        
        with patch.object(ocr_pipeline, '_detect_tables', return_value=mock_tables):
            tables = await ocr_pipeline.detect_tables(sample_image)
            
            assert len(tables) == 1
            assert tables[0]['rows'] == 5
            assert tables[0]['columns'] == 3
    
    def test_confidence_calculation(self, ocr_pipeline):
        """Test overall confidence calculation."""
        blocks = [
            TextBlock("A", bbox=(0, 0, 10, 10), confidence=0.9),
            TextBlock("B", bbox=(20, 0, 30, 10), confidence=0.8),
            TextBlock("C", bbox=(40, 0, 50, 10), confidence=0.7)
        ]
        
        # Weighted average based on text length
        overall_confidence = ocr_pipeline._calculate_overall_confidence(blocks)
        
        assert 0.7 <= overall_confidence <= 0.9
    
    @pytest.mark.parametrize("image_format", ["PNG", "JPEG", "TIFF", "BMP"])
    @pytest.mark.asyncio
    async def test_multiple_image_formats(self, ocr_pipeline, image_format, temp_dir):
        """Test processing different image formats."""
        # Create image in specified format
        img = Image.new('RGB', (400, 300), 'white')
        img_path = temp_dir / f"test.{image_format.lower()}"
        img.save(img_path, format=image_format)
        
        # Mock OCR
        ocr_pipeline.engine.recognize = AsyncMock(
            return_value=OCRResult(f"{image_format} text", [], 0.9)
        )
        
        # Load and process
        loaded_img = Image.open(img_path)
        result = await ocr_pipeline.process_image(loaded_img)
        
        assert result.text == f"{image_format} text"