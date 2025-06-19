"""
Layout quality metrics for evaluating document structure and visual arrangement.
Assesses alignment, spacing, table structure, and visual consistency.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

import math
from pydantic import BaseModel, Field, validator

from ...core.config import BaseConfig
from ...core.exceptions import ValidationError
from ...core.logging import get_logger

logger = get_logger(__name__)


class LayoutMetricType(Enum):
    """Layout quality metric types"""
    ALIGNMENT_CONSISTENCY = "alignment_consistency"
    SPACING_UNIFORMITY = "spacing_uniformity"
    ELEMENT_OVERLAP = "element_overlap"
    MARGIN_COMPLIANCE = "margin_compliance"
    COLUMN_BALANCE = "column_balance"
    TABLE_STRUCTURE = "table_structure"
    VISUAL_HIERARCHY = "visual_hierarchy"
    READABILITY_FLOW = "readability_flow"


class AlignmentType(Enum):
    """Element alignment types"""
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    TOP = "top"
    BOTTOM = "bottom"
    BASELINE = "baseline"


@dataclass
class BoundingBox:
    """Simple bounding box for layout analysis"""
    x: float
    y: float
    width: float
    height: float
    
    @property
    def x2(self) -> float:
        return self.x + self.width
    
    @property
    def y2(self) -> float:
        return self.y + self.height
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this box intersects with another"""
        return not (
            self.x2 < other.x or
            other.x2 < self.x or
            self.y2 < other.y or
            other.y2 < self.y
        )


@dataclass
class LayoutElement:
    """Layout element for analysis"""
    bbox: BoundingBox
    element_type: str
    content: Optional[str] = None
    style: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayoutIssue:
    """Layout quality issue"""
    issue_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    elements: List[str] = field(default_factory=list)
    suggested_fix: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "type": self.issue_type,
            "severity": self.severity,
            "description": self.description,
            "elements": self.elements,
            "suggested_fix": self.suggested_fix
        }


@dataclass
class LayoutMetricResult:
    """Layout metric calculation result"""
    metric_type: LayoutMetricType
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[LayoutIssue] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metric": self.metric_type.value,
            "score": self.score,
            "details": self.details,
            "issues": [issue.to_dict() for issue in self.issues]
        }


@dataclass
class LayoutQualityReport:
    """Comprehensive layout quality assessment report"""
    overall_score: float
    metrics: List[LayoutMetricResult]
    layout_statistics: Dict[str, Any]
    issues: List[LayoutIssue]
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "overall_score": self.overall_score,
            "metrics": [m.to_dict() for m in self.metrics],
            "layout_statistics": self.layout_statistics,
            "issues": [issue.to_dict() for issue in self.issues],
            "recommendations": self.recommendations
        }


class LayoutMetricsConfig(BaseConfig):
    """Layout metrics calculation configuration"""
    alignment_tolerance: float = Field(default=2.0, description="Alignment tolerance in points")
    spacing_tolerance: float = Field(default=1.0, description="Spacing tolerance in points")
    overlap_threshold: float = Field(default=0.1, description="Element overlap threshold")
    margin_tolerance: float = Field(default=5.0, description="Margin compliance tolerance")
    column_balance_threshold: float = Field(default=0.2, description="Column balance threshold")
    
    @validator("alignment_tolerance", "spacing_tolerance", "margin_tolerance")
    def validate_positive(cls, v):
        """Validate positive values"""
        if v < 0:
            raise ValueError("Tolerance values must be positive")
        return v


class LayoutMetricsCalculator:
    """
    Layout quality metrics calculator.
    Evaluates document layout quality and visual consistency.
    """
    
    def __init__(self, config: Optional[LayoutMetricsConfig] = None):
        """Initialize layout metrics calculator"""
        self.config = config or LayoutMetricsConfig()
        logger.info("Initialized LayoutMetricsCalculator")
    
    def evaluate_layout_quality(
        self,
        elements: List[LayoutElement],
        page_info: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[LayoutMetricType]] = None
    ) -> LayoutQualityReport:
        """
        Evaluate layout quality for a set of elements.
        
        Args:
            elements: List of layout elements to evaluate
            page_info: Page dimensions and margins
            metrics: List of metrics to calculate (None for all)
        
        Returns:
            Comprehensive layout quality report
        """
        # Calculate metrics
        if metrics is None:
            metrics = list(LayoutMetricType)
        
        metric_results = []
        all_issues = []
        
        for metric_type in metrics:
            try:
                result = self._calculate_layout_metric(metric_type, elements, page_info)
                metric_results.append(result)
                all_issues.extend(result.issues)
            except Exception as e:
                logger.warning(f"Failed to calculate {metric_type.value}: {str(e)}")
        
        # Calculate overall score
        overall_score = self._calculate_overall_layout_score(metric_results)
        
        # Generate statistics
        statistics = self._generate_layout_statistics(elements, page_info)
        
        # Generate recommendations
        recommendations = self._generate_layout_recommendations(metric_results, all_issues)
        
        return LayoutQualityReport(
            overall_score=overall_score,
            metrics=metric_results,
            layout_statistics=statistics,
            issues=all_issues,
            recommendations=recommendations
        )
    
    def _calculate_layout_metric(
        self,
        metric_type: LayoutMetricType,
        elements: List[LayoutElement],
        page_info: Optional[Dict[str, Any]]
    ) -> LayoutMetricResult:
        """Calculate individual layout metric"""
        if metric_type == LayoutMetricType.ALIGNMENT_CONSISTENCY:
            return self._calculate_alignment_consistency(elements)
        
        elif metric_type == LayoutMetricType.SPACING_UNIFORMITY:
            return self._calculate_spacing_uniformity(elements)
        
        elif metric_type == LayoutMetricType.ELEMENT_OVERLAP:
            return self._calculate_element_overlap(elements)
        
        elif metric_type == LayoutMetricType.MARGIN_COMPLIANCE:
            return self._calculate_margin_compliance(elements, page_info)
        
        elif metric_type == LayoutMetricType.COLUMN_BALANCE:
            return self._calculate_column_balance(elements, page_info)
        
        elif metric_type == LayoutMetricType.TABLE_STRUCTURE:
            return self._calculate_table_structure(elements)
        
        elif metric_type == LayoutMetricType.VISUAL_HIERARCHY:
            return self._calculate_visual_hierarchy(elements)
        
        elif metric_type == LayoutMetricType.READABILITY_FLOW:
            return self._calculate_readability_flow(elements)
        
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
    
    def _calculate_alignment_consistency(self, elements: List[LayoutElement]) -> LayoutMetricResult:
        """Calculate alignment consistency score"""
        if len(elements) < 2:
            return LayoutMetricResult(LayoutMetricType.ALIGNMENT_CONSISTENCY, 1.0)
        
        issues = []
        alignment_groups = self._group_elements_by_alignment(elements)
        
        # Calculate alignment scores for each group
        alignment_scores = []
        for alignment_type, group in alignment_groups.items():
            if len(group) > 1:
                score = self._calculate_group_alignment_score(group, alignment_type)
                alignment_scores.append(score)
                
                # Check for alignment issues
                if score < 0.8:
                    issues.append(LayoutIssue(
                        issue_type="poor_alignment",
                        severity="medium",
                        description=f"Poor {alignment_type} alignment consistency",
                        elements=[f"element_{i}" for i in range(len(group))],
                        suggested_fix=f"Improve {alignment_type} alignment of elements"
                    ))
        
        # Overall alignment score
        overall_score = statistics.mean(alignment_scores) if alignment_scores else 1.0
        
        details = {
            "alignment_groups": {k: len(v) for k, v in alignment_groups.items()},
            "group_scores": dict(zip(alignment_groups.keys(), alignment_scores)),
            "total_elements": len(elements)
        }
        
        return LayoutMetricResult(
            LayoutMetricType.ALIGNMENT_CONSISTENCY,
            overall_score,
            details,
            issues
        )
    
    def _group_elements_by_alignment(self, elements: List[LayoutElement]) -> Dict[str, List[LayoutElement]]:
        """Group elements by their alignment patterns"""
        # Simple grouping by approximate X or Y coordinates
        tolerance = self.config.alignment_tolerance
        
        # Group by left alignment
        left_groups = []
        for element in elements:
            placed = False
            for group in left_groups:
                if abs(element.bbox.x - group[0].bbox.x) <= tolerance:
                    group.append(element)
                    placed = True
                    break
            if not placed:
                left_groups.append([element])
        
        # Group by right alignment
        right_groups = []
        for element in elements:
            placed = False
            for group in right_groups:
                if abs(element.bbox.x2 - group[0].bbox.x2) <= tolerance:
                    group.append(element)
                    placed = True
                    break
            if not placed:
                right_groups.append([element])
        
        # Group by top alignment
        top_groups = []
        for element in elements:
            placed = False
            for group in top_groups:
                if abs(element.bbox.y - group[0].bbox.y) <= tolerance:
                    group.append(element)
                    placed = True
                    break
            if not placed:
                top_groups.append([element])
        
        return {
            "left": max(left_groups, key=len) if left_groups else [],
            "right": max(right_groups, key=len) if right_groups else [],
            "top": max(top_groups, key=len) if top_groups else []
        }
    
    def _calculate_group_alignment_score(self, group: List[LayoutElement], alignment_type: str) -> float:
        """Calculate alignment score for a group of elements"""
        if len(group) < 2:
            return 1.0
        
        if alignment_type == "left":
            coords = [elem.bbox.x for elem in group]
        elif alignment_type == "right":
            coords = [elem.bbox.x2 for elem in group]
        elif alignment_type == "top":
            coords = [elem.bbox.y for elem in group]
        else:
            return 1.0
        
        # Calculate coefficient of variation
        if len(coords) > 1:
            std_dev = statistics.stdev(coords)
            mean_coord = statistics.mean(coords)
            cv = std_dev / mean_coord if mean_coord != 0 else 0
            
            # Convert to score (lower CV = better alignment)
            score = max(0, 1 - (cv * 10))  # Scale factor
            return min(1.0, score)
        
        return 1.0
    
    def _calculate_spacing_uniformity(self, elements: List[LayoutElement]) -> LayoutMetricResult:
        """Calculate spacing uniformity score"""
        if len(elements) < 2:
            return LayoutMetricResult(LayoutMetricType.SPACING_UNIFORMITY, 1.0)
        
        # Calculate vertical and horizontal spacing
        vertical_spacings = []
        horizontal_spacings = []
        issues = []
        
        # Sort elements by position
        elements_by_y = sorted(elements, key=lambda e: e.bbox.y)
        elements_by_x = sorted(elements, key=lambda e: e.bbox.x)
        
        # Calculate vertical spacings
        for i in range(len(elements_by_y) - 1):
            curr = elements_by_y[i]
            next_elem = elements_by_y[i + 1]
            
            # Check if elements might be in same row
            if abs(curr.bbox.y - next_elem.bbox.y) > curr.bbox.height:
                spacing = next_elem.bbox.y - curr.bbox.y2
                if spacing >= 0:  # Only positive spacings
                    vertical_spacings.append(spacing)
        
        # Calculate horizontal spacings
        for i in range(len(elements_by_x) - 1):
            curr = elements_by_x[i]
            next_elem = elements_by_x[i + 1]
            
            # Check if elements might be in same column
            if abs(curr.bbox.x - next_elem.bbox.x) > curr.bbox.width:
                spacing = next_elem.bbox.x - curr.bbox.x2
                if spacing >= 0:  # Only positive spacings
                    horizontal_spacings.append(spacing)
        
        # Calculate uniformity scores
        v_score = self._calculate_spacing_score(vertical_spacings, "vertical", issues)
        h_score = self._calculate_spacing_score(horizontal_spacings, "horizontal", issues)
        
        overall_score = (v_score + h_score) / 2
        
        details = {
            "vertical_spacings": vertical_spacings,
            "horizontal_spacings": horizontal_spacings,
            "vertical_score": v_score,
            "horizontal_score": h_score,
            "spacing_statistics": {
                "vertical_mean": statistics.mean(vertical_spacings) if vertical_spacings else 0,
                "horizontal_mean": statistics.mean(horizontal_spacings) if horizontal_spacings else 0
            }
        }
        
        return LayoutMetricResult(
            LayoutMetricType.SPACING_UNIFORMITY,
            overall_score,
            details,
            issues
        )
    
    def _calculate_spacing_score(self, spacings: List[float], direction: str, issues: List[LayoutIssue]) -> float:
        """Calculate uniformity score for spacing values"""
        if len(spacings) < 2:
            return 1.0
        
        # Calculate coefficient of variation
        mean_spacing = statistics.mean(spacings)
        std_spacing = statistics.stdev(spacings)
        
        cv = std_spacing / mean_spacing if mean_spacing > 0 else 0
        
        # Convert to score
        score = max(0, 1 - cv)
        
        # Check for inconsistent spacing
        if cv > 0.3:  # 30% variation threshold
            issues.append(LayoutIssue(
                issue_type="inconsistent_spacing",
                severity="medium",
                description=f"Inconsistent {direction} spacing detected",
                suggested_fix=f"Standardize {direction} spacing between elements"
            ))
        
        return score
    
    def _calculate_element_overlap(self, elements: List[LayoutElement]) -> LayoutMetricResult:
        """Calculate element overlap score"""
        if len(elements) < 2:
            return LayoutMetricResult(LayoutMetricType.ELEMENT_OVERLAP, 1.0)
        
        overlaps = []
        issues = []
        
        # Check all pairs of elements
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                elem1, elem2 = elements[i], elements[j]
                
                if elem1.bbox.intersects(elem2.bbox):
                    # Calculate overlap area
                    overlap_area = self._calculate_overlap_area(elem1.bbox, elem2.bbox)
                    min_area = min(elem1.bbox.area, elem2.bbox.area)
                    overlap_ratio = overlap_area / min_area if min_area > 0 else 0
                    
                    overlaps.append(overlap_ratio)
                    
                    if overlap_ratio > self.config.overlap_threshold:
                        issues.append(LayoutIssue(
                            issue_type="element_overlap",
                            severity="high" if overlap_ratio > 0.5 else "medium",
                            description=f"Elements overlap by {overlap_ratio:.1%}",
                            elements=[f"element_{i}", f"element_{j}"],
                            suggested_fix="Adjust element positions to eliminate overlap"
                        ))
        
        # Calculate score (1.0 = no overlaps, lower = more overlaps)
        if overlaps:
            max_overlap = max(overlaps)
            avg_overlap = statistics.mean(overlaps)
            score = max(0, 1 - (max_overlap + avg_overlap) / 2)
        else:
            score = 1.0
        
        details = {
            "overlap_count": len(overlaps),
            "max_overlap_ratio": max(overlaps) if overlaps else 0,
            "avg_overlap_ratio": statistics.mean(overlaps) if overlaps else 0,
            "total_element_pairs": len(elements) * (len(elements) - 1) // 2
        }
        
        return LayoutMetricResult(
            LayoutMetricType.ELEMENT_OVERLAP,
            score,
            details,
            issues
        )
    
    def _calculate_overlap_area(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate overlap area between two bounding boxes"""
        left = max(bbox1.x, bbox2.x)
        right = min(bbox1.x2, bbox2.x2)
        top = max(bbox1.y, bbox2.y)
        bottom = min(bbox1.y2, bbox2.y2)
        
        if left < right and top < bottom:
            return (right - left) * (bottom - top)
        return 0.0
    
    def _calculate_margin_compliance(self, elements: List[LayoutElement], page_info: Optional[Dict[str, Any]]) -> LayoutMetricResult:
        """Calculate margin compliance score"""
        if not page_info:
            return LayoutMetricResult(LayoutMetricType.MARGIN_COMPLIANCE, 1.0)
        
        issues = []
        violations = []
        
        # Get page dimensions and margins
        page_width = page_info.get("width", 612)  # Default letter size
        page_height = page_info.get("height", 792)
        margins = page_info.get("margins", {"top": 72, "bottom": 72, "left": 72, "right": 72})
        
        # Calculate content area
        content_left = margins.get("left", 72)
        content_right = page_width - margins.get("right", 72)
        content_top = margins.get("top", 72)
        content_bottom = page_height - margins.get("bottom", 72)
        
        tolerance = self.config.margin_tolerance
        
        for i, element in enumerate(elements):
            # Check left margin
            if element.bbox.x < content_left - tolerance:
                violations.append(("left", element.bbox.x - content_left))
                issues.append(LayoutIssue(
                    issue_type="margin_violation",
                    severity="medium",
                    description="Element extends beyond left margin",
                    elements=[f"element_{i}"],
                    suggested_fix="Move element within page margins"
                ))
            
            # Check right margin
            if element.bbox.x2 > content_right + tolerance:
                violations.append(("right", element.bbox.x2 - content_right))
                issues.append(LayoutIssue(
                    issue_type="margin_violation",
                    severity="medium",
                    description="Element extends beyond right margin",
                    elements=[f"element_{i}"],
                    suggested_fix="Resize or reposition element within margins"
                ))
        
        # Calculate score
        if violations:
            avg_violation = statistics.mean([abs(v[1]) for v in violations])
            max_violation = max([abs(v[1]) for v in violations])
            
            # Normalize by page size
            page_size = max(page_width, page_height)
            score = max(0, 1 - (avg_violation + max_violation) / (2 * page_size))
        else:
            score = 1.0
        
        details = {
            "violation_count": len(violations),
            "violations_by_side": {
                side: sum(1 for v in violations if v[0] == side)
                for side in ["left", "right", "top", "bottom"]
            },
            "content_area": {
                "left": content_left,
                "right": content_right,
                "top": content_top,
                "bottom": content_bottom
            }
        }
        
        return LayoutMetricResult(
            LayoutMetricType.MARGIN_COMPLIANCE,
            score,
            details,
            issues
        )
    
    def _calculate_column_balance(self, elements: List[LayoutElement], page_info: Optional[Dict[str, Any]]) -> LayoutMetricResult:
        """Calculate column balance score"""
        # Group elements by approximate column (based on X position)
        if not elements:
            return LayoutMetricResult(LayoutMetricType.COLUMN_BALANCE, 1.0)
        
        # Simple column detection based on X coordinates
        elements_sorted = sorted(elements, key=lambda e: e.bbox.x)
        
        # Group elements into columns based on X coordinate gaps
        columns = []
        current_column = [elements_sorted[0]]
        
        for i in range(1, len(elements_sorted)):
            prev_elem = elements_sorted[i-1]
            curr_elem = elements_sorted[i]
            
            # If there's a significant gap, start new column
            if curr_elem.bbox.x - prev_elem.bbox.x2 > 50:  # 50 point gap threshold
                columns.append(current_column)
                current_column = [curr_elem]
            else:
                current_column.append(curr_elem)
        
        if current_column:
            columns.append(current_column)
        
        if len(columns) < 2:
            return LayoutMetricResult(LayoutMetricType.COLUMN_BALANCE, 1.0)
        
        # Calculate column heights (sum of element heights)
        column_heights = []
        for column in columns:
            height = sum(elem.bbox.height for elem in column)
            column_heights.append(height)
        
        issues = []
        
        # Calculate balance metrics
        mean_height = statistics.mean(column_heights)
        height_differences = [abs(h - mean_height) for h in column_heights]
        max_difference = max(height_differences)
        
        # Calculate balance score
        if mean_height > 0:
            balance_ratio = max_difference / mean_height
            score = max(0, 1 - balance_ratio)
        else:
            score = 1.0
        
        # Check for significant imbalance
        if balance_ratio > self.config.column_balance_threshold:
            issues.append(LayoutIssue(
                issue_type="column_imbalance",
                severity="medium",
                description=f"Column height imbalance of {balance_ratio:.1%}",
                suggested_fix="Redistribute content across columns for better balance"
            ))
        
        details = {
            "column_count": len(columns),
            "column_heights": column_heights,
            "mean_height": mean_height,
            "max_difference": max_difference,
            "balance_ratio": balance_ratio if mean_height > 0 else 0
        }
        
        return LayoutMetricResult(
            LayoutMetricType.COLUMN_BALANCE,
            score,
            details,
            issues
        )
    
    def _calculate_table_structure(self, elements: List[LayoutElement]) -> LayoutMetricResult:
        """Calculate table structure quality score"""
        table_elements = [e for e in elements if e.element_type == "table"]
        
        if not table_elements:
            return LayoutMetricResult(LayoutMetricType.TABLE_STRUCTURE, 1.0)
        
        issues = []
        table_scores = []
        
        for i, table_elem in enumerate(table_elements):
            # Simple table structure analysis
            table_score = 0.9  # Assume good structure
            
            if "rows" in table_elem.metadata and "cols" in table_elem.metadata:
                rows = table_elem.metadata["rows"]
                cols = table_elem.metadata["cols"]
                
                # Check for reasonable table dimensions
                if rows < 2 or cols < 2:
                    table_score *= 0.8
                    issues.append(LayoutIssue(
                        issue_type="small_table",
                        severity="low",
                        description=f"Table {i} has minimal dimensions ({rows}x{cols})",
                        elements=[f"table_{i}"]
                    ))
                
                # Check aspect ratio
                aspect_ratio = table_elem.bbox.width / table_elem.bbox.height
                if aspect_ratio > 5 or aspect_ratio < 0.2:
                    table_score *= 0.9
                    issues.append(LayoutIssue(
                        issue_type="table_aspect_ratio",
                        severity="low",
                        description=f"Table {i} has unusual aspect ratio",
                        elements=[f"table_{i}"]
                    ))
            
            table_scores.append(table_score)
        
        overall_score = statistics.mean(table_scores) if table_scores else 1.0
        
        details = {
            "table_count": len(table_elements),
            "table_scores": table_scores,
            "average_score": overall_score
        }
        
        return LayoutMetricResult(
            LayoutMetricType.TABLE_STRUCTURE,
            overall_score,
            details,
            issues
        )
    
    def _calculate_visual_hierarchy(self, elements: List[LayoutElement]) -> LayoutMetricResult:
        """Calculate visual hierarchy score"""
        issues = []
        
        # Group elements by type
        type_groups = {}
        for element in elements:
            elem_type = element.element_type
            if elem_type not in type_groups:
                type_groups[elem_type] = []
            type_groups[elem_type].append(element)
        
        # Check hierarchy consistency
        hierarchy_score = 1.0
        
        # Headings should be larger and positioned appropriately
        if "heading" in type_groups and "text" in type_groups:
            headings = type_groups["heading"]
            texts = type_groups["text"]
            
            for heading in headings:
                # Check if heading is visually prominent
                heading_size = heading.bbox.height
                avg_text_size = statistics.mean([t.bbox.height for t in texts])
                
                if heading_size <= avg_text_size:
                    hierarchy_score *= 0.9
                    issues.append(LayoutIssue(
                        issue_type="weak_hierarchy",
                        severity="medium",
                        description="Heading not visually prominent enough",
                        suggested_fix="Increase heading size or styling"
                    ))
        
        details = {
            "element_types": list(type_groups.keys()),
            "type_counts": {k: len(v) for k, v in type_groups.items()},
            "hierarchy_issues": len([i for i in issues if i.issue_type == "weak_hierarchy"])
        }
        
        return LayoutMetricResult(
            LayoutMetricType.VISUAL_HIERARCHY,
            hierarchy_score,
            details,
            issues
        )
    
    def _calculate_readability_flow(self, elements: List[LayoutElement]) -> LayoutMetricResult:
        """Calculate reading flow score"""
        text_elements = [e for e in elements if e.element_type in ["text", "heading"]]
        
        if len(text_elements) < 2:
            return LayoutMetricResult(LayoutMetricType.READABILITY_FLOW, 1.0)
        
        issues = []
        
        # Sort elements by reading order (top-to-bottom, left-to-right)
        sorted_elements = sorted(text_elements, key=lambda e: (e.bbox.y, e.bbox.x))
        
        # Check for logical flow
        flow_score = 1.0
        
        for i in range(len(sorted_elements) - 1):
            curr = sorted_elements[i]
            next_elem = sorted_elements[i + 1]
            
            # Check for reasonable vertical progression
            if next_elem.bbox.y < curr.bbox.y - curr.bbox.height:
                flow_score *= 0.95
                issues.append(LayoutIssue(
                    issue_type="reading_flow",
                    severity="low",
                    description="Unusual reading flow detected",
                    suggested_fix="Review element positioning for logical reading order"
                ))
        
        details = {
            "text_element_count": len(text_elements),
            "reading_order_issues": len([i for i in issues if i.issue_type == "reading_flow"])
        }
        
        return LayoutMetricResult(
            LayoutMetricType.READABILITY_FLOW,
            flow_score,
            details,
            issues
        )
    
    def _calculate_overall_layout_score(self, metric_results: List[LayoutMetricResult]) -> float:
        """Calculate weighted overall layout score"""
        if not metric_results:
            return 0.0
        
        # Define weights for different metrics
        weights = {
            LayoutMetricType.ALIGNMENT_CONSISTENCY: 0.20,
            LayoutMetricType.SPACING_UNIFORMITY: 0.20,
            LayoutMetricType.ELEMENT_OVERLAP: 0.25,
            LayoutMetricType.MARGIN_COMPLIANCE: 0.15,
            LayoutMetricType.COLUMN_BALANCE: 0.10,
            LayoutMetricType.TABLE_STRUCTURE: 0.05,
            LayoutMetricType.VISUAL_HIERARCHY: 0.03,
            LayoutMetricType.READABILITY_FLOW: 0.02
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for result in metric_results:
            weight = weights.get(result.metric_type, 0.1)
            weighted_sum += result.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_layout_statistics(self, elements: List[LayoutElement], page_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed layout statistics"""
        page_width = page_info.get("width", 612) if page_info else 612
        page_height = page_info.get("height", 792) if page_info else 792
        
        return {
            "page_info": {
                "width": page_width,
                "height": page_height,
                "margins": page_info.get("margins", {}) if page_info else {}
            },
            "element_statistics": {
                "total_count": len(elements),
                "types": dict(Counter([e.element_type for e in elements])),
                "avg_area": statistics.mean([e.bbox.area for e in elements]) if elements else 0,
                "total_coverage": sum([e.bbox.area for e in elements]) / (page_width * page_height)
            },
            "spacing_statistics": {
                "avg_element_height": statistics.mean([e.bbox.height for e in elements]) if elements else 0,
                "avg_element_width": statistics.mean([e.bbox.width for e in elements]) if elements else 0
            }
        }
    
    def _generate_layout_recommendations(
        self,
        metric_results: List[LayoutMetricResult],
        issues: List[LayoutIssue]
    ) -> List[str]:
        """Generate layout improvement recommendations"""
        recommendations = []
        
        # Analyze issues by type and severity
        critical_issues = [i for i in issues if i.severity == "critical"]
        high_issues = [i for i in issues if i.severity == "high"]
        medium_issues = [i for i in issues if i.severity == "medium"]
        
        if critical_issues:
            recommendations.append("Address critical layout issues immediately to ensure document usability.")
        
        if high_issues:
            recommendations.append("Fix high-priority layout issues to improve document quality.")
        
        # Specific recommendations based on metric scores
        for result in metric_results:
            if result.score < 0.6:
                if result.metric_type == LayoutMetricType.ALIGNMENT_CONSISTENCY:
                    recommendations.append("Improve element alignment consistency for better visual structure.")
                elif result.metric_type == LayoutMetricType.SPACING_UNIFORMITY:
                    recommendations.append("Standardize spacing between elements for uniform appearance.")
                elif result.metric_type == LayoutMetricType.ELEMENT_OVERLAP:
                    recommendations.append("Eliminate element overlaps to prevent content conflicts.")
                elif result.metric_type == LayoutMetricType.MARGIN_COMPLIANCE:
                    recommendations.append("Ensure all elements stay within page margins.")
        
        if not recommendations:
            recommendations.append("Layout quality is good. Continue monitoring for consistency.")
        
        return recommendations


def create_layout_metrics_calculator(
    config: Optional[Union[Dict[str, Any], LayoutMetricsConfig]] = None
) -> LayoutMetricsCalculator:
    """Factory function to create layout metrics calculator"""
    if isinstance(config, dict):
        config = LayoutMetricsConfig(**config)
    return LayoutMetricsCalculator(config)


def evaluate_layout_sample() -> LayoutQualityReport:
    """Generate sample layout evaluation for testing"""
    calculator = create_layout_metrics_calculator()
    
    # Create sample elements
    elements = [
        LayoutElement(
            bbox=BoundingBox(50, 50, 200, 30),
            element_type="heading",
            content="Sample Heading"
        ),
        LayoutElement(
            bbox=BoundingBox(50, 90, 300, 60),
            element_type="text",
            content="Sample text content"
        ),
        LayoutElement(
            bbox=BoundingBox(50, 160, 250, 100),
            element_type="table",
            metadata={"rows": 5, "cols": 3}
        )
    ]
    
    page_info = {
        "width": 612,
        "height": 792,
        "margins": {"top": 72, "bottom": 72, "left": 72, "right": 72}
    }
    
    return calculator.evaluate_layout_quality(elements, page_info)