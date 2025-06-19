"""
Table layout generator for creating complex table structures.
Generates tables with various layouts, cell types, and formatting options.
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
from ...generation.content.domain_data_generator import DomainDataGenerator, DomainType

logger = get_logger(__name__)


class CellType(Enum):
    """Table cell content types"""
    TEXT = "text"
    NUMBER = "number"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DATE = "date"
    TIME = "time"
    BOOLEAN = "boolean"
    EMPTY = "empty"
    HEADER = "header"
    TOTAL = "total"
    SUBTOTAL = "subtotal"


class CellAlignment(Enum):
    """Cell alignment options"""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "justify"


class BorderStyle(Enum):
    """Table border styles"""
    NONE = "none"
    SINGLE = "single"
    DOUBLE = "double"
    THICK = "thick"
    DASHED = "dashed"
    DOTTED = "dotted"


class TableStyle(Enum):
    """Predefined table styles"""
    PLAIN = "plain"
    GRID = "grid"
    STRIPED = "striped"
    BORDERED = "bordered"
    BORDERLESS = "borderless"
    COMPACT = "compact"
    FINANCIAL = "financial"
    REPORT = "report"
    INVOICE = "invoice"
    SPREADSHEET = "spreadsheet"


@dataclass
class CellStyle:
    """Cell styling attributes"""
    alignment: CellAlignment = CellAlignment.LEFT
    bold: bool = False
    italic: bool = False
    underline: bool = False
    background_color: Optional[str] = None
    text_color: Optional[str] = None
    font_size: Optional[int] = None
    border: Optional[BorderStyle] = None
    padding: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "alignment": self.alignment.value,
            "bold": self.bold,
            "italic": self.italic,
            "underline": self.underline,
            "background_color": self.background_color,
            "text_color": self.text_color,
            "font_size": self.font_size,
            "border": self.border.value if self.border else None,
            "padding": self.padding
        }


@dataclass
class TableCell:
    """Individual table cell"""
    content: Any
    cell_type: CellType
    style: CellStyle = field(default_factory=CellStyle)
    colspan: int = 1
    rowspan: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "type": self.cell_type.value,
            "style": self.style.to_dict(),
            "colspan": self.colspan,
            "rowspan": self.rowspan,
            "metadata": self.metadata
        }


@dataclass
class TableRow:
    """Table row containing cells"""
    cells: List[TableCell]
    is_header: bool = False
    height: Optional[float] = None
    style: Optional[CellStyle] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "cells": [cell.to_dict() for cell in self.cells],
            "is_header": self.is_header,
            "height": self.height,
            "style": self.style.to_dict() if self.style else None,
            "metadata": self.metadata
        }


@dataclass
class TableLayout:
    """Complete table layout"""
    rows: List[TableRow]
    columns: int
    title: Optional[str] = None
    caption: Optional[str] = None
    style: TableStyle = TableStyle.PLAIN
    column_widths: Optional[List[float]] = None
    total_width: float = 6.5  # inches
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def row_count(self) -> int:
        """Total number of rows"""
        return len(self.rows)
    
    @property
    def has_header(self) -> bool:
        """Check if table has header row"""
        return any(row.is_header for row in self.rows)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "caption": self.caption,
            "style": self.style.value,
            "columns": self.columns,
            "rows": [row.to_dict() for row in self.rows],
            "column_widths": self.column_widths,
            "total_width": self.total_width,
            "metadata": self.metadata
        }


class TableConfig(BaseConfig):
    """Table generation configuration"""
    min_rows: int = Field(default=3, ge=1, description="Minimum number of rows")
    max_rows: int = Field(default=20, ge=1, description="Maximum number of rows")
    min_cols: int = Field(default=2, ge=1, description="Minimum number of columns")
    max_cols: int = Field(default=8, ge=1, description="Maximum number of columns")
    include_headers: bool = Field(default=True, description="Include header row")
    include_totals: bool = Field(default=True, description="Include total rows for numeric data")
    style: TableStyle = Field(default=TableStyle.GRID, description="Table style")
    cell_padding: float = Field(default=0.1, description="Cell padding in inches")
    stripe_rows: bool = Field(default=False, description="Alternate row coloring")
    
    @validator("max_rows")
    def validate_max_rows(cls, v, values):
        """Ensure max_rows >= min_rows"""
        if "min_rows" in values and v < values["min_rows"]:
            raise ValueError("max_rows must be >= min_rows")
        return v
    
    @validator("max_cols")
    def validate_max_cols(cls, v, values):
        """Ensure max_cols >= min_cols"""
        if "min_cols" in values and v < values["min_cols"]:
            raise ValueError("max_cols must be >= min_cols")
        return v


class TableGenerator:
    """
    Table layout generator for creating complex table structures.
    Generates various types of tables with appropriate data and formatting.
    """
    
    # Common table templates
    TABLE_TEMPLATES = {
        "financial_report": {
            "title": "Financial Summary",
            "headers": ["Item", "Q1", "Q2", "Q3", "Q4", "Total"],
            "row_templates": [
                ("Revenue", CellType.CURRENCY),
                ("Expenses", CellType.CURRENCY),
                ("Profit", CellType.CURRENCY),
                ("Margin", CellType.PERCENTAGE)
            ]
        },
        "invoice": {
            "title": "Invoice Details",
            "headers": ["Item", "Description", "Quantity", "Unit Price", "Total"],
            "row_templates": [
                ("ITEM", CellType.TEXT),
                ("Description text", CellType.TEXT),
                ("1", CellType.NUMBER),
                ("100.00", CellType.CURRENCY),
                ("100.00", CellType.CURRENCY)
            ]
        },
        "inventory": {
            "title": "Inventory Report",
            "headers": ["SKU", "Product Name", "Stock", "Unit Cost", "Total Value", "Status"],
            "row_templates": [
                ("SKU-123", CellType.TEXT),
                ("Product Name", CellType.TEXT),
                ("100", CellType.NUMBER),
                ("10.00", CellType.CURRENCY),
                ("1000.00", CellType.CURRENCY),
                ("In Stock", CellType.TEXT)
            ]
        },
        "employee": {
            "title": "Employee Directory",
            "headers": ["ID", "Name", "Department", "Position", "Email", "Phone"],
            "row_templates": [
                ("EMP001", CellType.TEXT),
                ("John Doe", CellType.TEXT),
                ("Sales", CellType.TEXT),
                ("Manager", CellType.TEXT),
                ("john@company.com", CellType.TEXT),
                ("555-1234", CellType.TEXT)
            ]
        },
        "schedule": {
            "title": "Weekly Schedule",
            "headers": ["Time", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "row_templates": [
                ("9:00 AM", CellType.TIME),
                ("Meeting", CellType.TEXT),
                ("Task", CellType.TEXT),
                ("Task", CellType.TEXT),
                ("Meeting", CellType.TEXT),
                ("Review", CellType.TEXT)
            ]
        }
    }
    
    def __init__(self, config: Optional[TableConfig] = None):
        """Initialize table generator"""
        self.config = config or TableConfig()
        self.domain_generator = DomainDataGenerator()
        logger.info(f"Initialized TableGenerator with {self.config.style.value} style")
    
    def generate_table(
        self,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        table_type: Optional[str] = None
    ) -> TableLayout:
        """Generate a complete table layout"""
        # Determine dimensions
        if rows is None:
            rows = random.randint(self.config.min_rows, self.config.max_rows)
        if cols is None:
            cols = random.randint(self.config.min_cols, self.config.max_cols)
        
        # Generate based on type or create random
        if table_type and table_type in self.TABLE_TEMPLATES:
            table = self._generate_from_template(table_type, rows)
        else:
            table = self._generate_random_table(rows, cols)
        
        # Apply styling
        self._apply_table_style(table)
        
        return table
    
    def _generate_from_template(self, template_name: str, rows: int) -> TableLayout:
        """Generate table from template"""
        template = self.TABLE_TEMPLATES[template_name]
        
        # Create header row
        header_cells = []
        for header in template["headers"]:
            cell = TableCell(
                content=header,
                cell_type=CellType.HEADER,
                style=CellStyle(
                    alignment=CellAlignment.CENTER,
                    bold=True,
                    background_color="#f0f0f0"
                )
            )
            header_cells.append(cell)
        
        header_row = TableRow(cells=header_cells, is_header=True)
        table_rows = [header_row]
        
        # Generate data rows
        for i in range(rows - 1):  # -1 for header
            row_cells = []
            row_template = template["row_templates"]
            
            for j, (sample_content, cell_type) in enumerate(row_template):
                content = self._generate_cell_content(cell_type, sample_content, i, j)
                
                # Determine alignment based on type
                alignment = CellAlignment.LEFT
                if cell_type in [CellType.NUMBER, CellType.CURRENCY, CellType.PERCENTAGE]:
                    alignment = CellAlignment.RIGHT
                elif cell_type == CellType.DATE or cell_type == CellType.TIME:
                    alignment = CellAlignment.CENTER
                
                cell = TableCell(
                    content=content,
                    cell_type=cell_type,
                    style=CellStyle(alignment=alignment)
                )
                row_cells.append(cell)
            
            table_rows.append(TableRow(cells=row_cells))
        
        # Add totals row if applicable
        if self.config.include_totals and self._has_numeric_columns(table_rows):
            total_row = self._generate_total_row(table_rows, template["headers"])
            table_rows.append(total_row)
        
        # Calculate column widths
        column_widths = self._calculate_column_widths(len(template["headers"]))
        
        return TableLayout(
            rows=table_rows,
            columns=len(template["headers"]),
            title=template["title"],
            style=self.config.style,
            column_widths=column_widths
        )
    
    def _generate_random_table(self, rows: int, cols: int) -> TableLayout:
        """Generate random table layout"""
        table_rows = []
        
        # Generate headers if enabled
        if self.config.include_headers:
            header_cells = []
            for j in range(cols):
                header = f"Column {chr(65 + j)}"  # A, B, C, etc.
                cell = TableCell(
                    content=header,
                    cell_type=CellType.HEADER,
                    style=CellStyle(
                        alignment=CellAlignment.CENTER,
                        bold=True,
                        background_color="#f0f0f0"
                    )
                )
                header_cells.append(cell)
            
            table_rows.append(TableRow(cells=header_cells, is_header=True))
            rows -= 1  # Adjust for header row
        
        # Determine column types
        column_types = []
        for j in range(cols):
            # First column is often text/ID
            if j == 0:
                column_types.append(random.choice([CellType.TEXT, CellType.NUMBER]))
            else:
                column_types.append(random.choice([
                    CellType.TEXT, CellType.NUMBER, CellType.CURRENCY,
                    CellType.PERCENTAGE, CellType.DATE
                ]))
        
        # Generate data rows
        for i in range(rows):
            row_cells = []
            for j in range(cols):
                cell_type = column_types[j]
                content = self._generate_cell_content(cell_type, None, i, j)
                
                # Determine alignment
                alignment = CellAlignment.LEFT
                if cell_type in [CellType.NUMBER, CellType.CURRENCY, CellType.PERCENTAGE]:
                    alignment = CellAlignment.RIGHT
                elif cell_type == CellType.DATE:
                    alignment = CellAlignment.CENTER
                
                cell = TableCell(
                    content=content,
                    cell_type=cell_type,
                    style=CellStyle(alignment=alignment)
                )
                row_cells.append(cell)
            
            table_rows.append(TableRow(cells=row_cells))
        
        # Add totals if applicable
        if self.config.include_totals and self._has_numeric_columns(table_rows):
            total_row = self._generate_total_row(table_rows, None)
            table_rows.append(total_row)
        
        # Calculate column widths
        column_widths = self._calculate_column_widths(cols)
        
        return TableLayout(
            rows=table_rows,
            columns=cols,
            title="Data Table",
            style=self.config.style,
            column_widths=column_widths
        )
    
    def _generate_cell_content(
        self,
        cell_type: CellType,
        sample: Optional[str],
        row_idx: int,
        col_idx: int
    ) -> Any:
        """Generate appropriate content for cell type"""
        if cell_type == CellType.TEXT:
            if sample and not sample.startswith("ITEM"):
                # Use sample as base
                return f"{sample} {row_idx + 1}"
            else:
                # Generate random text
                text_options = [
                    f"Item {row_idx + 1}",
                    f"Product {chr(65 + row_idx % 26)}",
                    f"Entry #{row_idx + 1:03d}",
                    f"Data {row_idx + 1}"
                ]
                return random.choice(text_options)
        
        elif cell_type == CellType.NUMBER:
            return random.randint(1, 1000)
        
        elif cell_type == CellType.CURRENCY:
            return f"${random.uniform(10, 10000):.2f}"
        
        elif cell_type == CellType.PERCENTAGE:
            return f"{random.uniform(0, 100):.1f}%"
        
        elif cell_type == CellType.DATE:
            # Generate dates in 2023-2024
            year = random.choice([2023, 2024])
            month = random.randint(1, 12)
            day = random.randint(1, 28)  # Safe for all months
            return f"{month:02d}/{day:02d}/{year}"
        
        elif cell_type == CellType.TIME:
            hour = random.randint(8, 17)  # Business hours
            minute = random.choice([0, 15, 30, 45])
            return f"{hour}:{minute:02d}"
        
        elif cell_type == CellType.BOOLEAN:
            return random.choice(["Yes", "No", "True", "False"])
        
        else:
            return ""
    
    def _has_numeric_columns(self, rows: List[TableRow]) -> bool:
        """Check if table has numeric columns"""
        if len(rows) < 2:  # Need at least header + 1 data row
            return False
        
        # Check first data row
        data_row = rows[1] if rows[0].is_header else rows[0]
        
        for cell in data_row.cells:
            if cell.cell_type in [CellType.NUMBER, CellType.CURRENCY]:
                return True
        
        return False
    
    def _generate_total_row(
        self,
        rows: List[TableRow],
        headers: Optional[List[str]] = None
    ) -> TableRow:
        """Generate total row for numeric columns"""
        # Find first data row to determine structure
        first_data_idx = 1 if rows[0].is_header else 0
        if first_data_idx >= len(rows):
            return TableRow(cells=[])
        
        first_data_row = rows[first_data_idx]
        total_cells = []
        
        for col_idx, cell in enumerate(first_data_row.cells):
            if col_idx == 0:
                # First column shows "Total"
                total_cell = TableCell(
                    content="Total",
                    cell_type=CellType.TOTAL,
                    style=CellStyle(bold=True, alignment=CellAlignment.LEFT)
                )
            elif cell.cell_type in [CellType.NUMBER, CellType.CURRENCY]:
                # Calculate sum for numeric columns
                total = 0
                for row in rows[first_data_idx:]:
                    if not row.is_header and col_idx < len(row.cells):
                        cell_content = row.cells[col_idx].content
                        if isinstance(cell_content, (int, float)):
                            total += cell_content
                        elif isinstance(cell_content, str) and cell_content.startswith("$"):
                            # Parse currency
                            try:
                                total += float(cell_content.replace("$", "").replace(",", ""))
                            except:
                                pass
                
                # Format based on cell type
                if cell.cell_type == CellType.CURRENCY:
                    content = f"${total:.2f}"
                else:
                    content = str(int(total))
                
                total_cell = TableCell(
                    content=content,
                    cell_type=CellType.TOTAL,
                    style=CellStyle(
                        bold=True,
                        alignment=CellAlignment.RIGHT,
                        border=BorderStyle.SINGLE
                    )
                )
            else:
                # Empty cell for non-numeric columns
                total_cell = TableCell(
                    content="",
                    cell_type=CellType.EMPTY,
                    style=CellStyle()
                )
            
            total_cells.append(total_cell)
        
        return TableRow(cells=total_cells, style=CellStyle(background_color="#e0e0e0"))
    
    def _calculate_column_widths(self, num_cols: int) -> List[float]:
        """Calculate column widths based on content"""
        total_width = self.config.style != TableStyle.SPREADSHEET and 6.5 or 8.0
        
        if num_cols == 1:
            return [total_width]
        
        # Simple equal distribution with slight variation
        base_width = total_width / num_cols
        widths = []
        
        for i in range(num_cols):
            # First column often needs more space
            if i == 0:
                width = base_width * 1.2
            else:
                # Add small random variation
                width = base_width * random.uniform(0.9, 1.1)
            widths.append(width)
        
        # Normalize to ensure total equals table width
        total = sum(widths)
        widths = [w * total_width / total for w in widths]
        
        return widths
    
    def _apply_table_style(self, table: TableLayout) -> None:
        """Apply style-specific formatting to table"""
        if table.style == TableStyle.STRIPED or self.config.stripe_rows:
            # Apply alternating row colors
            for i, row in enumerate(table.rows):
                if not row.is_header and i % 2 == 0:
                    if not row.style:
                        row.style = CellStyle()
                    row.style.background_color = "#f9f9f9"
        
        elif table.style == TableStyle.BORDERED:
            # Add borders to all cells
            for row in table.rows:
                for cell in row.cells:
                    cell.style.border = BorderStyle.SINGLE
        
        elif table.style == TableStyle.FINANCIAL:
            # Financial styling
            for row in table.rows:
                for i, cell in enumerate(row.cells):
                    if cell.cell_type in [CellType.CURRENCY, CellType.PERCENTAGE]:
                        cell.style.alignment = CellAlignment.RIGHT
                    if cell.cell_type == CellType.TOTAL:
                        cell.style.border = BorderStyle.DOUBLE
        
        elif table.style == TableStyle.SPREADSHEET:
            # Spreadsheet-like styling
            for row in table.rows:
                for cell in row.cells:
                    cell.style.border = BorderStyle.SINGLE
                    cell.style.padding = 0.05
    
    def generate_complex_table(
        self,
        sections: List[Dict[str, Any]],
        merge_cells: bool = True
    ) -> TableLayout:
        """Generate complex table with multiple sections and merged cells"""
        all_rows = []
        max_cols = max(section.get("cols", 4) for section in sections)
        
        for section_idx, section in enumerate(sections):
            # Add section header if provided
            if "title" in section:
                header_cell = TableCell(
                    content=section["title"],
                    cell_type=CellType.HEADER,
                    colspan=max_cols,
                    style=CellStyle(
                        alignment=CellAlignment.CENTER,
                        bold=True,
                        background_color="#d0d0d0"
                    )
                )
                all_rows.append(TableRow(cells=[header_cell]))
            
            # Generate section content
            section_rows = section.get("rows", 5)
            section_cols = section.get("cols", max_cols)
            
            for i in range(section_rows):
                row_cells = []
                for j in range(section_cols):
                    # Random cell content
                    cell_type = random.choice([
                        CellType.TEXT, CellType.NUMBER,
                        CellType.CURRENCY, CellType.PERCENTAGE
                    ])
                    content = self._generate_cell_content(cell_type, None, i, j)
                    
                    cell = TableCell(content=content, cell_type=cell_type)
                    
                    # Occasionally merge cells
                    if merge_cells and random.random() < 0.1 and j < section_cols - 1:
                        cell.colspan = 2
                        j += 1  # Skip next cell
                    
                    row_cells.append(cell)
                
                # Pad row if needed
                while len(row_cells) < max_cols:
                    row_cells.append(TableCell(content="", cell_type=CellType.EMPTY))
                
                all_rows.append(TableRow(cells=row_cells))
            
            # Add separator between sections
            if section_idx < len(sections) - 1:
                separator_row = TableRow(
                    cells=[TableCell(
                        content="",
                        cell_type=CellType.EMPTY,
                        colspan=max_cols,
                        style=CellStyle(background_color="#ffffff", padding=0.05)
                    )]
                )
                all_rows.append(separator_row)
        
        return TableLayout(
            rows=all_rows,
            columns=max_cols,
            title="Complex Table Report",
            style=TableStyle.REPORT
        )


def create_table_generator(config: Optional[Union[Dict[str, Any], TableConfig]] = None) -> TableGenerator:
    """Factory function to create table generator"""
    if isinstance(config, dict):
        config = TableConfig(**config)
    return TableGenerator(config)


def generate_sample_tables() -> Dict[str, TableLayout]:
    """Generate sample tables of various types"""
    generator = create_table_generator()
    samples = {}
    
    # Generate template-based tables
    for template_name in TableGenerator.TABLE_TEMPLATES:
        try:
            table = generator.generate_table(table_type=template_name)
            samples[template_name] = table
            logger.info(f"Generated sample table: {template_name}")
        except Exception as e:
            logger.error(f"Failed to generate table {template_name}: {str(e)}")
    
    # Generate style-based tables
    for style in TableStyle:
        try:
            config = TableConfig(style=style)
            style_generator = TableGenerator(config)
            table = style_generator.generate_table(rows=8, cols=5)
            table.title = f"{style.value.title()} Style Table"
            samples[f"style_{style.value}"] = table
            logger.info(f"Generated {style.value} style table")
        except Exception as e:
            logger.error(f"Failed to generate {style.value} style table: {str(e)}")
    
    # Generate complex table
    try:
        complex_table = generator.generate_complex_table([
            {"title": "Sales Data", "rows": 5, "cols": 4},
            {"title": "Inventory Status", "rows": 4, "cols": 5},
            {"title": "Financial Summary", "rows": 3, "cols": 3}
        ])
        samples["complex_multi_section"] = complex_table
        logger.info("Generated complex multi-section table")
    except Exception as e:
        logger.error(f"Failed to generate complex table: {str(e)}")
    
    return samples