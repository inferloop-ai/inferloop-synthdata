"""
Document layout generation engine for creating complex page layouts.
Manages page structure, margins, columns, sections, and element positioning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

from pydantic import BaseModel, Field, validator

from ...core.config import BaseConfig
from ...core.exceptions import GenerationError
from ...core.logging import get_logger

logger = get_logger(__name__)


class PageSize(Enum):
    """Standard page sizes"""
    LETTER = (8.5, 11.0)  # inches
    LEGAL = (8.5, 14.0)
    A4 = (8.27, 11.69)
    A3 = (11.69, 16.54)
    TABLOID = (11.0, 17.0)


class Orientation(Enum):
    """Page orientation"""
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


class LayoutType(Enum):
    """Layout structure types"""
    SINGLE_COLUMN = "single_column"
    TWO_COLUMN = "two_column"
    THREE_COLUMN = "three_column"
    GRID = "grid"
    MAGAZINE = "magazine"
    FORM = "form"
    TABLE = "table"
    MIXED = "mixed"


class ElementType(Enum):
    """Layout element types"""
    TEXT = "text"
    HEADING = "heading"
    IMAGE = "image"
    TABLE = "table"
    FORM_FIELD = "form_field"
    SIGNATURE = "signature"
    BARCODE = "barcode"
    QR_CODE = "qr_code"
    CHART = "chart"
    SEPARATOR = "separator"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    WATERMARK = "watermark"


@dataclass
class Margin:
    """Page margins in inches"""
    top: float = 1.0
    bottom: float = 1.0
    left: float = 1.0
    right: float = 1.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "top": self.top,
            "bottom": self.bottom,
            "left": self.left,
            "right": self.right
        }


@dataclass
class BoundingBox:
    """Element bounding box coordinates"""
    x: float
    y: float
    width: float
    height: float
    
    @property
    def x2(self) -> float:
        """Right edge coordinate"""
        return self.x + self.width
    
    @property
    def y2(self) -> float:
        """Bottom edge coordinate"""
        return self.y + self.height
    
    @property
    def area(self) -> float:
        """Area of the bounding box"""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the box"""
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    def intersects(self, other: BoundingBox) -> bool:
        """Check if this box intersects with another"""
        return not (
            self.x2 < other.x or
            other.x2 < self.x or
            self.y2 < other.y or
            other.y2 < self.y
        )
    
    def contains(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside the box"""
        x, y = point
        return self.x <= x <= self.x2 and self.y <= y <= self.y2
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "x2": self.x2,
            "y2": self.y2
        }


@dataclass
class LayoutElement:
    """Individual layout element"""
    element_type: ElementType
    bbox: BoundingBox
    content: Optional[Any] = None
    style: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "type": self.element_type.value,
            "bbox": self.bbox.to_dict(),
            "content": self.content,
            "style": self.style,
            "metadata": self.metadata
        }


@dataclass
class Column:
    """Column definition"""
    x: float
    y: float
    width: float
    height: float
    elements: List[LayoutElement] = field(default_factory=list)
    
    @property
    def bbox(self) -> BoundingBox:
        """Get column bounding box"""
        return BoundingBox(self.x, self.y, self.width, self.height)
    
    @property
    def used_height(self) -> float:
        """Calculate used height in column"""
        if not self.elements:
            return 0
        return max(elem.bbox.y2 - self.y for elem in self.elements)
    
    @property
    def available_height(self) -> float:
        """Calculate available height in column"""
        return self.height - self.used_height
    
    def can_fit(self, element_height: float, spacing: float = 0.1) -> bool:
        """Check if element can fit in column"""
        return self.available_height >= element_height + spacing
    
    def add_element(self, element: LayoutElement, spacing: float = 0.1) -> None:
        """Add element to column"""
        # Position element
        if self.elements:
            last_element = self.elements[-1]
            element.bbox.y = last_element.bbox.y2 + spacing
        else:
            element.bbox.y = self.y
        
        element.bbox.x = self.x
        self.elements.append(element)


@dataclass
class Page:
    """Page layout definition"""
    width: float
    height: float
    margins: Margin
    columns: List[Column] = field(default_factory=list)
    elements: List[LayoutElement] = field(default_factory=list)
    page_number: Optional[int] = None
    
    @property
    def content_width(self) -> float:
        """Available content width"""
        return self.width - self.margins.left - self.margins.right
    
    @property
    def content_height(self) -> float:
        """Available content height"""
        return self.height - self.margins.top - self.margins.bottom
    
    @property
    def content_bbox(self) -> BoundingBox:
        """Content area bounding box"""
        return BoundingBox(
            self.margins.left,
            self.margins.top,
            self.content_width,
            self.content_height
        )
    
    def add_header(self, height: float = 0.5) -> LayoutElement:
        """Add header to page"""
        header = LayoutElement(
            ElementType.HEADER,
            BoundingBox(
                self.margins.left,
                self.margins.top / 2,
                self.content_width,
                height
            )
        )
        self.elements.append(header)
        return header
    
    def add_footer(self, height: float = 0.5) -> LayoutElement:
        """Add footer to page"""
        footer = LayoutElement(
            ElementType.FOOTER,
            BoundingBox(
                self.margins.left,
                self.height - self.margins.bottom / 2 - height,
                self.content_width,
                height
            )
        )
        self.elements.append(footer)
        return footer
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "width": self.width,
            "height": self.height,
            "margins": self.margins.to_dict(),
            "columns": [{"bbox": col.bbox.to_dict(), "elements": [e.to_dict() for e in col.elements]} for col in self.columns],
            "elements": [e.to_dict() for e in self.elements],
            "page_number": self.page_number
        }


class LayoutConfig(BaseConfig):
    """Layout generation configuration"""
    page_size: PageSize = Field(default=PageSize.LETTER, description="Page size")
    orientation: Orientation = Field(default=Orientation.PORTRAIT, description="Page orientation")
    margins: Dict[str, float] = Field(
        default_factory=lambda: {"top": 1.0, "bottom": 1.0, "left": 1.0, "right": 1.0},
        description="Page margins in inches"
    )
    layout_type: LayoutType = Field(default=LayoutType.SINGLE_COLUMN, description="Layout type")
    column_spacing: float = Field(default=0.25, description="Space between columns")
    line_spacing: float = Field(default=0.1, description="Space between lines")
    enable_headers: bool = Field(default=True, description="Enable page headers")
    enable_footers: bool = Field(default=True, description="Enable page footers")
    enable_page_numbers: bool = Field(default=True, description="Enable page numbering")
    
    @validator("margins")
    def validate_margins(cls, v):
        """Validate margin values"""
        required_keys = {"top", "bottom", "left", "right"}
        if not all(key in v for key in required_keys):
            raise ValueError(f"Margins must include: {required_keys}")
        if not all(val >= 0 for val in v.values()):
            raise ValueError("Margin values must be non-negative")
        return v


class LayoutEngine:
    """
    Document layout generation engine.
    Creates complex page layouts with columns, sections, and element positioning.
    """
    
    def __init__(self, config: Optional[LayoutConfig] = None):
        """Initialize layout engine"""
        self.config = config or LayoutConfig()
        logger.info(f"Initialized LayoutEngine with {self.config.layout_type.value} layout")
    
    def create_page(self, page_number: Optional[int] = None) -> Page:
        """Create a new page with configured layout"""
        # Get page dimensions
        width, height = self.config.page_size.value
        if self.config.orientation == Orientation.LANDSCAPE:
            width, height = height, width
        
        # Create page
        margins = Margin(**self.config.margins)
        page = Page(width, height, margins, page_number=page_number)
        
        # Add header/footer if enabled
        if self.config.enable_headers:
            header = page.add_header()
            if page_number and self.config.enable_page_numbers:
                header.content = f"Page {page_number}"
        
        if self.config.enable_footers:
            footer = page.add_footer()
            if page_number and self.config.enable_page_numbers and not self.config.enable_headers:
                footer.content = f"Page {page_number}"
        
        # Create columns based on layout type
        self._create_columns(page)
        
        return page
    
    def _create_columns(self, page: Page) -> None:
        """Create columns based on layout type"""
        content_bbox = page.content_bbox
        spacing = self.config.column_spacing
        
        if self.config.layout_type == LayoutType.SINGLE_COLUMN:
            page.columns.append(Column(
                content_bbox.x,
                content_bbox.y,
                content_bbox.width,
                content_bbox.height
            ))
        
        elif self.config.layout_type == LayoutType.TWO_COLUMN:
            col_width = (content_bbox.width - spacing) / 2
            for i in range(2):
                x = content_bbox.x + i * (col_width + spacing)
                page.columns.append(Column(x, content_bbox.y, col_width, content_bbox.height))
        
        elif self.config.layout_type == LayoutType.THREE_COLUMN:
            col_width = (content_bbox.width - 2 * spacing) / 3
            for i in range(3):
                x = content_bbox.x + i * (col_width + spacing)
                page.columns.append(Column(x, content_bbox.y, col_width, content_bbox.height))
        
        elif self.config.layout_type == LayoutType.MAGAZINE:
            # Magazine layout: 2/3 main column, 1/3 sidebar
            main_width = content_bbox.width * 0.65
            sidebar_width = content_bbox.width * 0.35 - spacing
            
            page.columns.append(Column(
                content_bbox.x,
                content_bbox.y,
                main_width,
                content_bbox.height
            ))
            page.columns.append(Column(
                content_bbox.x + main_width + spacing,
                content_bbox.y,
                sidebar_width,
                content_bbox.height
            ))
        
        elif self.config.layout_type == LayoutType.GRID:
            # 2x2 grid layout
            col_width = (content_bbox.width - spacing) / 2
            row_height = (content_bbox.height - spacing) / 2
            
            for row in range(2):
                for col in range(2):
                    x = content_bbox.x + col * (col_width + spacing)
                    y = content_bbox.y + row * (row_height + spacing)
                    page.columns.append(Column(x, y, col_width, row_height))
    
    def add_text_element(
        self,
        page: Page,
        text: str,
        element_type: ElementType = ElementType.TEXT,
        height: Optional[float] = None,
        column_index: Optional[int] = None,
        style: Optional[Dict[str, Any]] = None
    ) -> Optional[LayoutElement]:
        """Add text element to page"""
        # Estimate height if not provided
        if height is None:
            # Rough estimation: ~0.15 inches per line
            lines = max(1, len(text) / 80)  # ~80 chars per line
            height = lines * 0.15
        
        # Create element
        element = LayoutElement(
            element_type,
            BoundingBox(0, 0, 0, height),  # Width/position set by column
            content=text,
            style=style or {}
        )
        
        # Find suitable column
        if column_index is not None and 0 <= column_index < len(page.columns):
            column = page.columns[column_index]
            if column.can_fit(height, self.config.line_spacing):
                element.bbox.width = column.width
                column.add_element(element, self.config.line_spacing)
                return element
        else:
            # Try all columns
            for column in page.columns:
                if column.can_fit(height, self.config.line_spacing):
                    element.bbox.width = column.width
                    column.add_element(element, self.config.line_spacing)
                    return element
        
        # No space found
        logger.warning(f"No space found for text element of height {height}")
        return None
    
    def add_table_element(
        self,
        page: Page,
        rows: int,
        cols: int,
        row_height: float = 0.25,
        column_index: Optional[int] = None,
        style: Optional[Dict[str, Any]] = None
    ) -> Optional[LayoutElement]:
        """Add table element to page"""
        height = rows * row_height
        
        # Tables typically span full width
        content_bbox = page.content_bbox
        element = LayoutElement(
            ElementType.TABLE,
            BoundingBox(
                content_bbox.x,
                0,  # Y position set later
                content_bbox.width,
                height
            ),
            content={"rows": rows, "cols": cols},
            style=style or {},
            metadata={"row_height": row_height}
        )
        
        # Find position
        if page.columns:
            # Place after last element in first column
            column = page.columns[0]
            if column.elements:
                last_y = column.elements[-1].bbox.y2
                element.bbox.y = last_y + self.config.line_spacing
            else:
                element.bbox.y = column.y
            
            # Check if fits
            if element.bbox.y2 <= content_bbox.y2:
                page.elements.append(element)
                return element
        
        logger.warning(f"No space found for table element of height {height}")
        return None
    
    def add_form_field(
        self,
        page: Page,
        label: str,
        field_type: str = "text",
        height: float = 0.3,
        column_index: Optional[int] = None,
        style: Optional[Dict[str, Any]] = None
    ) -> Optional[LayoutElement]:
        """Add form field element to page"""
        element = LayoutElement(
            ElementType.FORM_FIELD,
            BoundingBox(0, 0, 0, height),
            content={"label": label, "type": field_type},
            style=style or {}
        )
        
        # Find suitable column
        if column_index is not None and 0 <= column_index < len(page.columns):
            column = page.columns[column_index]
            if column.can_fit(height, self.config.line_spacing):
                element.bbox.width = column.width
                column.add_element(element, self.config.line_spacing)
                return element
        else:
            # Try all columns
            for column in page.columns:
                if column.can_fit(height, self.config.line_spacing):
                    element.bbox.width = column.width
                    column.add_element(element, self.config.line_spacing)
                    return element
        
        logger.warning(f"No space found for form field '{label}'")
        return None
    
    def add_signature_block(
        self,
        page: Page,
        labels: List[str],
        height: float = 1.0,
        style: Optional[Dict[str, Any]] = None
    ) -> Optional[LayoutElement]:
        """Add signature block element"""
        content_bbox = page.content_bbox
        element = LayoutElement(
            ElementType.SIGNATURE,
            BoundingBox(
                content_bbox.x,
                0,  # Y position set later
                content_bbox.width,
                height
            ),
            content={"labels": labels},
            style=style or {}
        )
        
        # Place at bottom of page
        element.bbox.y = content_bbox.y2 - height - 0.5  # 0.5 inch from bottom
        
        # Check for overlap with existing elements
        for col in page.columns:
            for elem in col.elements:
                if elem.bbox.intersects(element.bbox):
                    logger.warning("Signature block would overlap with existing content")
                    return None
        
        page.elements.append(element)
        return element
    
    def create_document_layout(
        self,
        content_elements: List[Dict[str, Any]],
        max_pages: int = 10
    ) -> List[Page]:
        """Create multi-page document layout from content elements"""
        pages = []
        current_page = self.create_page(page_number=1)
        pages.append(current_page)
        
        for element_data in content_elements:
            element_type = ElementType(element_data.get("type", "text"))
            
            added = False
            if element_type in [ElementType.TEXT, ElementType.HEADING]:
                added = self.add_text_element(
                    current_page,
                    element_data.get("content", ""),
                    element_type,
                    element_data.get("height"),
                    element_data.get("column"),
                    element_data.get("style")
                ) is not None
            
            elif element_type == ElementType.TABLE:
                added = self.add_table_element(
                    current_page,
                    element_data.get("rows", 5),
                    element_data.get("cols", 3),
                    element_data.get("row_height", 0.25),
                    element_data.get("column"),
                    element_data.get("style")
                ) is not None
            
            elif element_type == ElementType.FORM_FIELD:
                added = self.add_form_field(
                    current_page,
                    element_data.get("label", "Field"),
                    element_data.get("field_type", "text"),
                    element_data.get("height", 0.3),
                    element_data.get("column"),
                    element_data.get("style")
                ) is not None
            
            elif element_type == ElementType.SIGNATURE:
                added = self.add_signature_block(
                    current_page,
                    element_data.get("labels", ["Signature"]),
                    element_data.get("height", 1.0),
                    element_data.get("style")
                ) is not None
            
            # If element couldn't fit, create new page
            if not added and len(pages) < max_pages:
                current_page = self.create_page(page_number=len(pages) + 1)
                pages.append(current_page)
                
                # Try adding to new page
                if element_type in [ElementType.TEXT, ElementType.HEADING]:
                    self.add_text_element(
                        current_page,
                        element_data.get("content", ""),
                        element_type,
                        element_data.get("height"),
                        element_data.get("column"),
                        element_data.get("style")
                    )
        
        return pages
    
    def generate_sample_layout(self, layout_type: Optional[LayoutType] = None) -> Page:
        """Generate a sample layout for testing"""
        # Use specified layout type or configured
        if layout_type:
            original_type = self.config.layout_type
            self.config.layout_type = layout_type
        
        page = self.create_page(page_number=1)
        
        # Add sample content based on layout type
        if self.config.layout_type == LayoutType.FORM:
            # Form layout
            self.add_text_element(page, "APPLICATION FORM", ElementType.HEADING, 0.3)
            self.add_form_field(page, "Full Name", "text")
            self.add_form_field(page, "Date of Birth", "date")
            self.add_form_field(page, "Email Address", "email")
            self.add_form_field(page, "Phone Number", "tel")
            self.add_text_element(page, "Please provide additional information below:", ElementType.TEXT, 0.2)
            self.add_form_field(page, "Comments", "textarea", 1.0)
            self.add_signature_block(page, ["Applicant Signature", "Date"])
        
        elif self.config.layout_type == LayoutType.TABLE:
            # Table-heavy layout
            self.add_text_element(page, "QUARTERLY REPORT", ElementType.HEADING, 0.3)
            self.add_text_element(page, "Financial Summary for Q4 2023", ElementType.TEXT, 0.2)
            self.add_table_element(page, 10, 5)
            self.add_text_element(page, "Regional Performance", ElementType.HEADING, 0.25)
            self.add_table_element(page, 8, 4)
        
        elif self.config.layout_type == LayoutType.TWO_COLUMN:
            # Two-column article
            self.add_text_element(page, "RESEARCH ARTICLE", ElementType.HEADING, 0.3, column_index=0)
            self.add_text_element(
                page,
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
                ElementType.TEXT,
                column_index=0
            )
            self.add_text_element(
                page,
                "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
                ElementType.TEXT,
                column_index=1
            )
        
        else:
            # Default single column
            self.add_text_element(page, "DOCUMENT TITLE", ElementType.HEADING, 0.3)
            self.add_text_element(
                page,
                "This is a sample document demonstrating the layout engine capabilities. It can handle various content types and layouts.",
                ElementType.TEXT
            )
            self.add_text_element(page, "Section 1: Introduction", ElementType.HEADING, 0.25)
            self.add_text_element(
                page,
                "The layout engine provides flexible document structure generation with support for multiple columns, tables, forms, and more.",
                ElementType.TEXT
            )
        
        # Restore original layout type if changed
        if layout_type:
            self.config.layout_type = original_type
        
        return page


def create_layout_engine(config: Optional[Union[Dict[str, Any], LayoutConfig]] = None) -> LayoutEngine:
    """Factory function to create layout engine"""
    if isinstance(config, dict):
        config = LayoutConfig(**config)
    return LayoutEngine(config)


def generate_sample_layouts() -> Dict[str, Page]:
    """Generate sample layouts for all layout types"""
    engine = create_layout_engine()
    samples = {}
    
    for layout_type in LayoutType:
        try:
            page = engine.generate_sample_layout(layout_type)
            samples[layout_type.value] = page
            logger.info(f"Generated sample layout for {layout_type.value}")
        except Exception as e:
            logger.error(f"Failed to generate sample for {layout_type.value}: {str(e)}")
    
    return samples