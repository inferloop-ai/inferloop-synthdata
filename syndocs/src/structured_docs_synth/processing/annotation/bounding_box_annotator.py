#!/usr/bin/env python3
"""
Bounding Box Annotator for layout and spatial annotation
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


class SpatialRelation(Enum):
    """Spatial relationships between bounding boxes"""
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    INSIDE = "inside"
    CONTAINS = "contains"
    OVERLAPS = "overlaps"
    ADJACENT = "adjacent"
    ALIGNED_HORIZONTAL = "aligned_horizontal"
    ALIGNED_VERTICAL = "aligned_vertical"


@dataclass
class BoundingBoxAnnotation:
    """Bounding box annotation with metadata"""
    id: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    label: str
    confidence: float = 1.0
    page_num: int = 0
    text: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    spatial_relations: List[Tuple[str, SpatialRelation]] = field(default_factory=list)


@dataclass
class BoundingBoxResult:
    """Bounding box annotation result"""
    annotations: List[BoundingBoxAnnotation]
    spatial_graph: Dict[str, List[Tuple[str, SpatialRelation]]]
    layout_structure: Dict[str, Any]
    processing_time: float = 0.0


class BoundingBoxConfig(BaseModel):
    """Bounding box annotator configuration"""
    
    # Detection settings
    min_box_area: int = Field(50, description="Minimum bounding box area")
    max_box_area: int = Field(500000, description="Maximum bounding box area")
    overlap_threshold: float = Field(0.5, description="Overlap threshold for merging")
    
    # Spatial analysis
    enable_spatial_relations: bool = Field(True, description="Analyze spatial relationships")
    alignment_tolerance: int = Field(10, description="Tolerance for alignment detection")
    adjacency_threshold: int = Field(20, description="Threshold for adjacency detection")
    
    # Layout detection
    detect_columns: bool = Field(True, description="Detect column layout")
    detect_tables: bool = Field(True, description="Detect table structures")
    detect_reading_order: bool = Field(True, description="Detect reading order")


class BoundingBoxAnnotator:
    """
    Bounding Box Annotator for spatial layout analysis
    """
    
    def __init__(self, config: Optional[BoundingBoxConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or BoundingBoxConfig()
        
        self.logger.info("Bounding Box Annotator initialized")
    
    def annotate_layout(self, ocr_result) -> BoundingBoxResult:
        """Create bounding box annotations from OCR result"""
        start_time = time.time()
        
        try:
            annotations = []
            
            # Process each page
            for page in ocr_result.pages:
                page_annotations = self._process_page(page)
                annotations.extend(page_annotations)
            
            # Analyze spatial relationships
            spatial_graph = {}
            if self.config.enable_spatial_relations:
                spatial_graph = self._analyze_spatial_relations(annotations)
            
            # Detect layout structure
            layout_structure = self._detect_layout_structure(annotations)
            
            processing_time = time.time() - start_time
            
            result = BoundingBoxResult(
                annotations=annotations,
                spatial_graph=spatial_graph,
                layout_structure=layout_structure,
                processing_time=processing_time
            )
            
            self.logger.debug(f"Bounding box annotation completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Bounding box annotation failed: {e}")
            raise ProcessingError(f"Bounding box annotation error: {e}")
    
    def _process_page(self, page) -> List[BoundingBoxAnnotation]:
        """Process a single page for bounding box annotations"""
        annotations = []
        
        # Process words
        for word in page.words:
            if self._is_valid_bbox(word.bbox):
                annotation = BoundingBoxAnnotation(
                    id=f"word_{page.page_num}_{word.word_num}",
                    bbox=(word.bbox.x, word.bbox.y, 
                          word.bbox.x + word.bbox.width, 
                          word.bbox.y + word.bbox.height),
                    label="word",
                    confidence=word.confidence / 100.0,
                    page_num=page.page_num,
                    text=word.text,
                    attributes={
                        'line_num': word.line_num,
                        'word_num': word.word_num
                    }
                )
                annotations.append(annotation)
        
        # Process lines
        for line in page.lines:
            if self._is_valid_bbox(line.bbox):
                annotation = BoundingBoxAnnotation(
                    id=f"line_{page.page_num}_{line.line_num}",
                    bbox=(line.bbox.x, line.bbox.y,
                          line.bbox.x + line.bbox.width,
                          line.bbox.y + line.bbox.height),
                    label="text_line",
                    confidence=line.confidence / 100.0,
                    page_num=page.page_num,
                    text=line.text,
                    attributes={
                        'line_num': line.line_num,
                        'word_count': len(line.words)
                    }
                )
                annotations.append(annotation)
        
        return annotations
    
    def _is_valid_bbox(self, bbox) -> bool:
        """Check if bounding box is valid"""
        area = bbox.width * bbox.height
        return (self.config.min_box_area <= area <= self.config.max_box_area and
                bbox.width > 0 and bbox.height > 0)
    
    def _analyze_spatial_relations(self, annotations: List[BoundingBoxAnnotation]) -> Dict[str, List[Tuple[str, SpatialRelation]]]:
        """Analyze spatial relationships between annotations"""
        spatial_graph = {}
        
        for i, ann1 in enumerate(annotations):
            relations = []
            
            for j, ann2 in enumerate(annotations):
                if i != j and ann1.page_num == ann2.page_num:
                    relation = self._get_spatial_relation(ann1.bbox, ann2.bbox)
                    if relation:
                        relations.append((ann2.id, relation))
            
            spatial_graph[ann1.id] = relations
        
        return spatial_graph
    
    def _get_spatial_relation(self, bbox1: Tuple[int, int, int, int], 
                            bbox2: Tuple[int, int, int, int]) -> Optional[SpatialRelation]:
        """Determine spatial relationship between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate overlap
        overlap_x = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        overlap_y = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        overlap_area = overlap_x * overlap_y
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Check for containment
        if overlap_area == area2:  # bbox2 is inside bbox1
            return SpatialRelation.CONTAINS
        elif overlap_area == area1:  # bbox1 is inside bbox2
            return SpatialRelation.INSIDE
        
        # Check for overlap
        overlap_ratio = overlap_area / min(area1, area2) if min(area1, area2) > 0 else 0
        if overlap_ratio > self.config.overlap_threshold:
            return SpatialRelation.OVERLAPS
        
        # Check spatial positions
        center1_x, center1_y = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
        center2_x, center2_y = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
        
        # Check alignment
        if abs(center1_y - center2_y) <= self.config.alignment_tolerance:
            if center1_x < center2_x:
                return SpatialRelation.LEFT_OF
            else:
                return SpatialRelation.RIGHT_OF
        
        if abs(center1_x - center2_x) <= self.config.alignment_tolerance:
            if center1_y < center2_y:
                return SpatialRelation.ABOVE
            else:
                return SpatialRelation.BELOW
        
        # Check adjacency
        h_distance = min(abs(x2_1 - x1_2), abs(x2_2 - x1_1))
        v_distance = min(abs(y2_1 - y1_2), abs(y2_2 - y1_1))
        
        if (h_distance <= self.config.adjacency_threshold or 
            v_distance <= self.config.adjacency_threshold):
            return SpatialRelation.ADJACENT
        
        return None
    
    def _detect_layout_structure(self, annotations: List[BoundingBoxAnnotation]) -> Dict[str, Any]:
        """Detect layout structure from annotations"""
        structure = {
            'columns': [],
            'rows': [],
            'tables': [],
            'reading_order': []
        }
        
        # Group annotations by page
        page_annotations = {}
        for ann in annotations:
            if ann.page_num not in page_annotations:
                page_annotations[ann.page_num] = []
            page_annotations[ann.page_num].append(ann)
        
        # Analyze each page
        for page_num, page_anns in page_annotations.items():
            if self.config.detect_columns:
                columns = self._detect_columns(page_anns)
                structure['columns'].extend(columns)
            
            if self.config.detect_tables:
                tables = self._detect_tables(page_anns)
                structure['tables'].extend(tables)
            
            if self.config.detect_reading_order:
                reading_order = self._detect_reading_order(page_anns)
                structure['reading_order'].extend(reading_order)
        
        return structure
    
    def _detect_columns(self, annotations: List[BoundingBoxAnnotation]) -> List[Dict[str, Any]]:
        """Detect column layout"""
        if not annotations:
            return []
        
        # Sort by x-coordinate
        sorted_anns = sorted(annotations, key=lambda a: a.bbox[0])
        
        # Group by x-position (simplified column detection)
        columns = []
        current_column = [sorted_anns[0]]
        
        for ann in sorted_anns[1:]:
            # Check if annotation is in the same column
            last_ann = current_column[-1]
            x_distance = abs(ann.bbox[0] - last_ann.bbox[0])
            
            if x_distance <= 50:  # Same column threshold
                current_column.append(ann)
            else:
                # Start new column
                if len(current_column) > 1:
                    columns.append({
                        'annotations': [a.id for a in current_column],
                        'x_range': (min(a.bbox[0] for a in current_column),
                                   max(a.bbox[2] for a in current_column))
                    })
                current_column = [ann]
        
        # Add last column
        if len(current_column) > 1:
            columns.append({
                'annotations': [a.id for a in current_column],
                'x_range': (min(a.bbox[0] for a in current_column),
                           max(a.bbox[2] for a in current_column))
            })
        
        return columns
    
    def _detect_tables(self, annotations: List[BoundingBoxAnnotation]) -> List[Dict[str, Any]]:
        """Detect table structures"""
        # Simplified table detection based on grid alignment
        tables = []
        
        # Group annotations by approximate y-coordinates (rows)
        y_groups = {}
        tolerance = 20
        
        for ann in annotations:
            y_center = (ann.bbox[1] + ann.bbox[3]) / 2
            
            # Find existing group or create new one
            found_group = False
            for group_y in y_groups:
                if abs(y_center - group_y) <= tolerance:
                    y_groups[group_y].append(ann)
                    found_group = True
                    break
            
            if not found_group:
                y_groups[y_center] = [ann]
        
        # Check for table-like structures (multiple aligned rows)
        if len(y_groups) >= 2:
            rows = []
            for y_pos in sorted(y_groups.keys()):
                row_anns = sorted(y_groups[y_pos], key=lambda a: a.bbox[0])
                if len(row_anns) >= 2:  # At least 2 columns
                    rows.append({
                        'y_position': y_pos,
                        'annotations': [a.id for a in row_anns]
                    })
            
            if len(rows) >= 2:  # At least 2 rows
                tables.append({
                    'rows': rows,
                    'estimated_cells': sum(len(row['annotations']) for row in rows)
                })
        
        return tables
    
    def _detect_reading_order(self, annotations: List[BoundingBoxAnnotation]) -> List[str]:
        """Detect reading order of annotations"""
        # Simple top-to-bottom, left-to-right ordering
        text_anns = [ann for ann in annotations if ann.label in ['word', 'text_line']]
        
        # Sort by y-coordinate first, then x-coordinate
        sorted_anns = sorted(text_anns, key=lambda a: (a.bbox[1], a.bbox[0]))
        
        return [ann.id for ann in sorted_anns]
    
    def get_annotations_by_label(self, result: BoundingBoxResult, label: str) -> List[BoundingBoxAnnotation]:
        """Get annotations by label"""
        return [ann for ann in result.annotations if ann.label == label]
    
    def get_spatial_neighbors(self, result: BoundingBoxResult, annotation_id: str, 
                            relation: SpatialRelation) -> List[str]:
        """Get spatial neighbors of annotation with specific relationship"""
        if annotation_id in result.spatial_graph:
            return [ann_id for ann_id, rel in result.spatial_graph[annotation_id] if rel == relation]
        return []


# Factory function
def create_bounding_box_annotator(**config_kwargs) -> BoundingBoxAnnotator:
    """Factory function to create bounding box annotator"""
    config = BoundingBoxConfig(**config_kwargs)
    return BoundingBoxAnnotator(config)