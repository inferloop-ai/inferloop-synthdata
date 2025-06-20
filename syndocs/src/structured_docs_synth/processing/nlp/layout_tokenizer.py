#!/usr/bin/env python3
"""
Document Layout Tokenizer for structured document analysis
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError, ValidationError


class TokenType(Enum):
    """Document layout token types"""
    TEXT = "text"
    TITLE = "title"
    HEADER = "header"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_CELL = "table_cell"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    SIGNATURE = "signature"
    LOGO = "logo"
    FORM_FIELD = "form_field"


@dataclass
class LayoutToken:
    """Document layout token with spatial information"""
    text: str
    token_type: TokenType
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    page_num: int = 0
    confidence: float = 1.0
    font_size: Optional[float] = None
    font_family: Optional[str] = None
    is_bold: bool = False
    is_italic: bool = False
    color: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayoutTokenizerResult:
    """Layout tokenization result"""
    tokens: List[LayoutToken]
    page_structure: Dict[int, List[LayoutToken]]
    reading_order: List[int]  # Token indices in reading order
    processing_time: float = 0.0


class LayoutTokenizerConfig(BaseModel):
    """Layout tokenizer configuration"""
    
    # Tokenization settings
    min_token_length: int = Field(1, description="Minimum token length")
    merge_adjacent_tokens: bool = Field(True, description="Merge adjacent tokens")
    detect_reading_order: bool = Field(True, description="Detect reading order")
    
    # Layout analysis
    line_height_threshold: float = Field(1.5, description="Line height threshold for grouping")
    column_detection: bool = Field(True, description="Enable column detection")
    table_detection: bool = Field(True, description="Enable table detection")
    
    # Font analysis
    analyze_fonts: bool = Field(True, description="Analyze font properties")
    title_font_size_threshold: float = Field(14.0, description="Font size threshold for titles")


class LayoutTokenizer:
    """
    Document layout tokenizer for structured analysis
    """
    
    def __init__(self, config: Optional[LayoutTokenizerConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or LayoutTokenizerConfig()
        
        self.logger.info("Layout Tokenizer initialized")
    
    def tokenize_layout(self, ocr_result) -> LayoutTokenizerResult:
        """Tokenize document layout from OCR result"""
        start_time = time.time()
        
        try:
            tokens = []
            page_structure = {}
            
            # Process each page
            for page in ocr_result.pages:
                page_tokens = self._tokenize_page(page)
                tokens.extend(page_tokens)
                page_structure[page.page_num] = page_tokens
            
            # Detect reading order
            reading_order = []
            if self.config.detect_reading_order:
                reading_order = self._detect_reading_order(tokens)
            
            processing_time = time.time() - start_time
            
            result = LayoutTokenizerResult(
                tokens=tokens,
                page_structure=page_structure,
                reading_order=reading_order,
                processing_time=processing_time
            )
            
            self.logger.debug(f"Layout tokenization completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Layout tokenization failed: {e}")
            raise ProcessingError(f"Layout tokenization error: {e}")
    
    def _tokenize_page(self, page) -> List[LayoutToken]:
        """Tokenize a single page"""
        tokens = []
        
        # Group lines into logical blocks
        line_groups = self._group_lines(page.lines)
        
        for group in line_groups:
            # Determine token type based on layout features
            token_type = self._classify_token_type(group)
            
            # Create token
            combined_text = ' '.join([line.text for line in group])
            bbox = self._calculate_group_bbox(group)
            
            # Analyze font properties
            font_info = self._analyze_font_properties(group)
            
            token = LayoutToken(
                text=combined_text,
                token_type=token_type,
                bbox=bbox,
                page_num=page.page_num,
                confidence=sum(line.confidence for line in group) / len(group) / 100.0,
                font_size=font_info.get('size'),
                font_family=font_info.get('family'),
                is_bold=font_info.get('bold', False),
                is_italic=font_info.get('italic', False),
                metadata={'line_count': len(group)}
            )
            
            tokens.append(token)
        
        return tokens
    
    def _group_lines(self, lines) -> List[List]:
        """Group lines into logical blocks based on layout"""
        if not lines:
            return []
        
        groups = []
        current_group = [lines[0]]
        
        for i in range(1, len(lines)):
            current_line = lines[i]
            prev_line = lines[i-1]
            
            # Check if lines should be grouped together
            if self._should_group_lines(prev_line, current_line):
                current_group.append(current_line)
            else:
                groups.append(current_group)
                current_group = [current_line]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _should_group_lines(self, line1, line2) -> bool:
        """Determine if two lines should be grouped together"""
        # Calculate vertical distance
        v_distance = line2.bbox.y - (line1.bbox.y + line1.bbox.height)
        
        # Calculate line height
        line1_height = line1.bbox.height
        line2_height = line2.bbox.height
        avg_height = (line1_height + line2_height) / 2
        
        # Group if lines are close vertically
        if v_distance <= avg_height * self.config.line_height_threshold:
            # Check horizontal alignment for paragraphs
            h_overlap = min(line1.bbox.x + line1.bbox.width, line2.bbox.x + line2.bbox.width) - \
                       max(line1.bbox.x, line2.bbox.x)
            
            # Lines should have some horizontal overlap or be left-aligned
            if h_overlap > 0 or abs(line1.bbox.x - line2.bbox.x) < avg_height:
                return True
        
        return False
    
    def _classify_token_type(self, line_group) -> TokenType:
        """Classify token type based on layout features"""
        if not line_group:
            return TokenType.TEXT
        
        # Analyze first line for classification
        first_line = line_group[0]
        text = first_line.text.strip()
        
        # Check for titles (larger font, short text, centered)
        if (self.config.analyze_fonts and 
            len(text) < 100 and 
            len(line_group) == 1):
            return TokenType.TITLE
        
        # Check for headers (numbered sections, bold text)
        if self._is_header(text):
            return TokenType.HEADER
        
        # Check for list items
        if self._is_list_item(text):
            return TokenType.LIST_ITEM
        
        # Check for page numbers
        if self._is_page_number(text):
            return TokenType.PAGE_NUMBER
        
        # Default to paragraph for multi-line groups
        if len(line_group) > 1:
            return TokenType.PARAGRAPH
        
        return TokenType.TEXT
    
    def _is_header(self, text: str) -> bool:
        """Check if text is a header"""
        import re
        
        # Pattern for numbered headers (1., 1.1, A., etc.)
        header_patterns = [
            r'^\d+\.',
            r'^\d+\.\d+',
            r'^[A-Z]\.',
            r'^[IVX]+\.',
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _is_list_item(self, text: str) -> bool:
        """Check if text is a list item"""
        import re
        
        # Pattern for list items (bullet points, numbers, etc.)
        list_patterns = [
            r'^"',
            r'^-',
            r'^\*',
            r'^\d+\)',
            r'^[a-z]\)',
        ]
        
        for pattern in list_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _is_page_number(self, text: str) -> bool:
        """Check if text is a page number"""
        import re
        
        # Simple page number patterns
        if re.match(r'^\d+$', text.strip()) and len(text.strip()) <= 3:
            return True
        
        if re.match(r'^Page \d+', text, re.IGNORECASE):
            return True
        
        return False
    
    def _calculate_group_bbox(self, line_group) -> Tuple[int, int, int, int]:
        """Calculate bounding box for a group of lines"""
        if not line_group:
            return (0, 0, 0, 0)
        
        min_x = min(line.bbox.x for line in line_group)
        min_y = min(line.bbox.y for line in line_group)
        max_x = max(line.bbox.x + line.bbox.width for line in line_group)
        max_y = max(line.bbox.y + line.bbox.height for line in line_group)
        
        return (min_x, min_y, max_x, max_y)
    
    def _analyze_font_properties(self, line_group) -> Dict[str, Any]:
        """Analyze font properties of a line group"""
        # This is a simplified implementation
        # In practice, you'd extract font info from the OCR engine
        font_info = {
            'size': None,
            'family': None,
            'bold': False,
            'italic': False
        }
        
        # Estimate font size based on line height
        if line_group:
            avg_height = sum(line.bbox.height for line in line_group) / len(line_group)
            font_info['size'] = avg_height * 0.8  # Rough estimation
        
        return font_info
    
    def _detect_reading_order(self, tokens: List[LayoutToken]) -> List[int]:
        """Detect reading order of tokens"""
        if not tokens:
            return []
        
        # Group tokens by page
        page_tokens = {}
        for i, token in enumerate(tokens):
            page_num = token.page_num
            if page_num not in page_tokens:
                page_tokens[page_num] = []
            page_tokens[page_num].append((i, token))
        
        reading_order = []
        
        # Process each page
        for page_num in sorted(page_tokens.keys()):
            page_token_indices = self._sort_tokens_reading_order(page_tokens[page_num])
            reading_order.extend(page_token_indices)
        
        return reading_order
    
    def _sort_tokens_reading_order(self, page_tokens) -> List[int]:
        """Sort tokens in reading order for a single page"""
        # Simple top-to-bottom, left-to-right ordering
        sorted_tokens = sorted(page_tokens, key=lambda x: (x[1].bbox[1], x[1].bbox[0]))
        return [token_idx for token_idx, _ in sorted_tokens]
    
    def get_tokens_by_type(self, result: LayoutTokenizerResult, 
                          token_type: TokenType) -> List[LayoutToken]:
        """Get tokens of specific type"""
        return [token for token in result.tokens if token.token_type == token_type]
    
    def get_page_tokens(self, result: LayoutTokenizerResult, 
                       page_num: int) -> List[LayoutToken]:
        """Get tokens for specific page"""
        return result.page_structure.get(page_num, [])
    
    def extract_document_structure(self, result: LayoutTokenizerResult) -> Dict[str, Any]:
        """Extract document structure summary"""
        structure = {
            'total_tokens': len(result.tokens),
            'pages': len(result.page_structure),
            'token_types': {},
            'reading_order_length': len(result.reading_order)
        }
        
        # Count tokens by type
        for token in result.tokens:
            token_type = token.token_type.value
            if token_type not in structure['token_types']:
                structure['token_types'][token_type] = 0
            structure['token_types'][token_type] += 1
        
        return structure


def create_layout_tokenizer(**config_kwargs) -> LayoutTokenizer:
    """Factory function to create layout tokenizer"""
    config = LayoutTokenizerConfig(**config_kwargs)
    return LayoutTokenizer(config)