#!/usr/bin/env python3
"""
Structure Annotator for document structure annotation
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError, ValidationError
from ..nlp.layout_tokenizer import LayoutToken, TokenType


class StructureType(Enum):
    """Document structure types"""
    HEADER = "header"
    FOOTER = "footer"
    TITLE = "title"
    SUBTITLE = "subtitle"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_HEADER = "table_header"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    FIGURE = "figure"
    CAPTION = "caption"
    SIDEBAR = "sidebar"
    COLUMN = "column"
    SECTION = "section"


@dataclass
class StructureAnnotation:
    """Document structure annotation"""
    id: str
    structure_type: StructureType
    bbox: Tuple[int, int, int, int]
    text: str = ""
    confidence: float = 1.0
    page_num: int = 0
    hierarchy_level: int = 0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentStructure:
    """Complete document structure"""
    annotations: List[StructureAnnotation]
    hierarchy: Dict[str, List[str]]  # parent_id -> children_ids
    reading_order: List[str]  # annotation IDs in reading order
    structure_tree: Dict[str, Any]  # Hierarchical structure
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructureAnnotationResult:
    """Structure annotation result"""
    document_structure: DocumentStructure
    structure_statistics: Dict[str, int]
    processing_time: float = 0.0


class StructureAnnotatorConfig(BaseModel):
    """Structure annotator configuration"""
    
    # Detection settings
    enable_hierarchy: bool = Field(True, description="Build hierarchical structure")
    detect_reading_order: bool = Field(True, description="Detect reading order")
    merge_similar_elements: bool = Field(True, description="Merge similar elements")
    
    # Classification settings
    title_font_threshold: float = Field(1.5, description="Font size ratio for title detection")
    header_position_threshold: float = Field(0.2, description="Relative position for header detection")
    footer_position_threshold: float = Field(0.8, description="Relative position for footer detection")
    
    # Layout analysis
    column_detection: bool = Field(True, description="Detect column layout")
    table_detection: bool = Field(True, description="Detect table structures")
    list_detection: bool = Field(True, description="Detect lists")
    
    # Quality settings
    min_text_length: int = Field(3, description="Minimum text length for annotation")
    confidence_threshold: float = Field(0.6, description="Minimum confidence threshold")


class StructureAnnotator:
    """
    Structure Annotator for document layout and hierarchy analysis
    """
    
    def __init__(self, config: Optional[StructureAnnotatorConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or StructureAnnotatorConfig()
        
        self.logger.info("Structure Annotator initialized")
    
    def annotate_structure(self, layout_tokens: List[LayoutToken], 
                          ocr_result=None) -> StructureAnnotationResult:
        """Annotate document structure from layout tokens"""
        start_time = time.time()
        
        try:
            # Create structure annotations
            annotations = self._create_structure_annotations(layout_tokens, ocr_result)
            
            # Build hierarchy if enabled
            hierarchy = {}
            structure_tree = {}
            if self.config.enable_hierarchy:
                hierarchy, structure_tree = self._build_hierarchy(annotations)
            
            # Detect reading order
            reading_order = []
            if self.config.detect_reading_order:
                reading_order = self._detect_reading_order(annotations)
            
            # Create document structure
            doc_structure = DocumentStructure(
                annotations=annotations,
                hierarchy=hierarchy,
                reading_order=reading_order,
                structure_tree=structure_tree,
                metadata={
                    'total_elements': len(annotations),
                    'hierarchy_enabled': self.config.enable_hierarchy
                }
            )
            
            # Calculate statistics
            structure_stats = self._calculate_structure_statistics(annotations)
            
            processing_time = time.time() - start_time
            
            result = StructureAnnotationResult(
                document_structure=doc_structure,
                structure_statistics=structure_stats,
                processing_time=processing_time
            )
            
            self.logger.debug(f"Structure annotation completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Structure annotation failed: {e}")
            raise ProcessingError(f"Structure annotation error: {e}")
    
    def _create_structure_annotations(self, layout_tokens: List[LayoutToken], 
                                    ocr_result=None) -> List[StructureAnnotation]:
        """Create structure annotations from layout tokens"""
        annotations = []
        
        # Get page dimensions for relative positioning
        page_heights = {}
        if ocr_result:
            for page in ocr_result.pages:
                page_heights[page.page_num] = page.height
        
        for i, token in enumerate(layout_tokens):
            if len(token.text.strip()) < self.config.min_text_length:
                continue
            
            # Classify structure type
            structure_type = self._classify_structure_type(token, page_heights)
            
            # Determine hierarchy level
            hierarchy_level = self._determine_hierarchy_level(token, structure_type)
            
            annotation = StructureAnnotation(
                id=f"struct_{i}",
                structure_type=structure_type,
                bbox=token.bbox,
                text=token.text,
                confidence=token.confidence,
                page_num=token.page_num,
                hierarchy_level=hierarchy_level,
                attributes={
                    'token_type': token.token_type.value,
                    'font_size': token.font_size,
                    'is_bold': token.is_bold,
                    'is_italic': token.is_italic
                }
            )
            
            annotations.append(annotation)
        
        return annotations
    
    def _classify_structure_type(self, token: LayoutToken, 
                               page_heights: Dict[int, int]) -> StructureType:
        """Classify token into structure type"""
        # Check token type first
        if token.token_type == TokenType.TITLE:
            return StructureType.TITLE
        elif token.token_type == TokenType.HEADER:
            return StructureType.HEADER
        elif token.token_type == TokenType.PARAGRAPH:
            return StructureType.PARAGRAPH
        elif token.token_type == TokenType.LIST_ITEM:
            return StructureType.LIST_ITEM
        elif token.token_type == TokenType.TABLE:
            return StructureType.TABLE
        elif token.token_type == TokenType.CAPTION:
            return StructureType.CAPTION
        elif token.token_type == TokenType.FOOTER:
            return StructureType.FOOTER
        
        # Additional classification based on position and formatting
        page_height = page_heights.get(token.page_num, 1000)
        relative_y = token.bbox[1] / page_height if page_height > 0 else 0
        
        # Check for header/footer based on position
        if relative_y <= self.config.header_position_threshold:
            if token.font_size and token.font_size > 12:
                return StructureType.HEADER
        elif relative_y >= self.config.footer_position_threshold:
            return StructureType.FOOTER
        
        # Check for title based on font size and formatting
        if (token.font_size and token.font_size > 14 and 
            (token.is_bold or len(token.text) < 100)):
            return StructureType.TITLE
        
        # Check for lists
        if self._is_list_item(token.text):
            return StructureType.LIST_ITEM
        
        # Default to paragraph
        return StructureType.PARAGRAPH
    
    def _is_list_item(self, text: str) -> bool:
        """Check if text represents a list item"""
        import re
        
        list_patterns = [
            r'^\s*["·\-\*]\s+',  # Bullet points
            r'^\s*\d+[.)>]\s+',   # Numbered lists
            r'^\s*[a-zA-Z][.)>]\s+',  # Lettered lists
            r'^\s*[ivxlcdm]+[.)>]\s+',  # Roman numerals
        ]
        
        for pattern in list_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _determine_hierarchy_level(self, token: LayoutToken, 
                                 structure_type: StructureType) -> int:
        """Determine hierarchy level for token"""
        # Base levels by structure type
        type_levels = {
            StructureType.TITLE: 1,
            StructureType.SUBTITLE: 2,
            StructureType.HEADER: 2,
            StructureType.SECTION: 3,
            StructureType.PARAGRAPH: 4,
            StructureType.LIST: 4,
            StructureType.LIST_ITEM: 5,
            StructureType.TABLE: 4,
            StructureType.TABLE_ROW: 5,
            StructureType.TABLE_CELL: 6,
            StructureType.CAPTION: 5,
            StructureType.FOOTER: 6
        }
        
        base_level = type_levels.get(structure_type, 4)
        
        # Adjust based on font size
        if token.font_size:
            if token.font_size >= 18:
                base_level = max(1, base_level - 2)
            elif token.font_size >= 14:
                base_level = max(1, base_level - 1)
        
        # Adjust based on formatting
        if token.is_bold and structure_type in [StructureType.PARAGRAPH, StructureType.HEADER]:
            base_level = max(1, base_level - 1)
        
        return base_level
    
    def _build_hierarchy(self, annotations: List[StructureAnnotation]) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
        """Build hierarchical structure"""
        hierarchy = {}
        structure_tree = {'root': {'children': [], 'type': 'root'}}
        
        # Sort annotations by page and position
        sorted_annotations = sorted(annotations, key=lambda a: (a.page_num, a.bbox[1], a.bbox[0]))
        
        # Build hierarchy based on levels
        level_stack = []  # Stack of (level, annotation_id)
        
        for annotation in sorted_annotations:
            current_level = annotation.hierarchy_level
            current_id = annotation.id
            
            # Find parent level
            parent_id = None
            while level_stack and level_stack[-1][0] >= current_level:
                level_stack.pop()
            
            if level_stack:
                parent_id = level_stack[-1][1]
                annotation.parent_id = parent_id
                
                # Add to hierarchy
                if parent_id not in hierarchy:
                    hierarchy[parent_id] = []
                hierarchy[parent_id].append(current_id)
            
            # Add to level stack
            level_stack.append((current_level, current_id))
            
            # Build structure tree
            if parent_id:
                # Find parent in structure tree and add child
                self._add_to_structure_tree(structure_tree, parent_id, current_id, annotation)
            else:
                # Add to root
                structure_tree['root']['children'].append({
                    'id': current_id,
                    'type': annotation.structure_type.value,
                    'text': annotation.text[:50] + '...' if len(annotation.text) > 50 else annotation.text,
                    'children': []
                })
        
        return hierarchy, structure_tree
    
    def _add_to_structure_tree(self, tree: Dict[str, Any], parent_id: str, 
                             child_id: str, annotation: StructureAnnotation):
        """Add child to structure tree"""
        # Simplified tree building - would need recursive search in practice
        pass
    
    def _detect_reading_order(self, annotations: List[StructureAnnotation]) -> List[str]:
        """Detect reading order of structure elements"""
        # Sort by page, then by position (top-to-bottom, left-to-right)
        sorted_annotations = sorted(
            annotations, 
            key=lambda a: (a.page_num, a.bbox[1], a.bbox[0])
        )
        
        return [ann.id for ann in sorted_annotations]
    
    def _calculate_structure_statistics(self, annotations: List[StructureAnnotation]) -> Dict[str, int]:
        """Calculate structure statistics"""
        stats = {
            'total_elements': len(annotations),
            'structure_types': {},
            'hierarchy_levels': {},
            'pages': set()
        }
        
        for annotation in annotations:
            # Count by structure type
            struct_type = annotation.structure_type.value
            if struct_type not in stats['structure_types']:
                stats['structure_types'][struct_type] = 0
            stats['structure_types'][struct_type] += 1
            
            # Count by hierarchy level
            level = annotation.hierarchy_level
            if level not in stats['hierarchy_levels']:
                stats['hierarchy_levels'][level] = 0
            stats['hierarchy_levels'][level] += 1
            
            # Track pages
            stats['pages'].add(annotation.page_num)
        
        stats['total_pages'] = len(stats['pages'])
        del stats['pages']  # Remove set from final stats
        
        return stats
    
    def get_elements_by_type(self, result: StructureAnnotationResult, 
                           structure_type: StructureType) -> List[StructureAnnotation]:
        """Get elements by structure type"""
        return [
            ann for ann in result.document_structure.annotations 
            if ann.structure_type == structure_type
        ]
    
    def get_hierarchy_children(self, result: StructureAnnotationResult, 
                             parent_id: str) -> List[StructureAnnotation]:
        """Get children of a parent element"""
        child_ids = result.document_structure.hierarchy.get(parent_id, [])
        
        id_to_annotation = {
            ann.id: ann for ann in result.document_structure.annotations
        }
        
        return [id_to_annotation[child_id] for child_id in child_ids if child_id in id_to_annotation]
    
    def export_structure(self, result: StructureAnnotationResult, 
                        format: str = "json") -> Dict[str, Any]:
        """Export structure in specified format"""
        export_data = {
            'document_structure': {
                'total_elements': len(result.document_structure.annotations),
                'hierarchy': result.document_structure.hierarchy,
                'reading_order': result.document_structure.reading_order,
                'annotations': []
            },
            'statistics': result.structure_statistics,
            'processing_time': result.processing_time
        }
        
        # Export annotations
        for annotation in result.document_structure.annotations:
            ann_data = {
                'id': annotation.id,
                'type': annotation.structure_type.value,
                'text': annotation.text,
                'bbox': annotation.bbox,
                'page_num': annotation.page_num,
                'hierarchy_level': annotation.hierarchy_level,
                'confidence': annotation.confidence
            }
            
            if annotation.parent_id:
                ann_data['parent_id'] = annotation.parent_id
            
            if annotation.children_ids:
                ann_data['children_ids'] = annotation.children_ids
            
            export_data['document_structure']['annotations'].append(ann_data)
        
        return export_data


# Factory function
def create_structure_annotator(**config_kwargs) -> StructureAnnotator:
    """Factory function to create structure annotator"""
    config = StructureAnnotatorConfig(**config_kwargs)
    return StructureAnnotator(config)