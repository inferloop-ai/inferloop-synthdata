"""
Document layout generation module.
Provides engines for creating complex page layouts, forms, and tables.
"""

from typing import Dict, List, Optional, Union, Any

from .layout_engine import (
    LayoutEngine,
    LayoutConfig,
    Page,
    PageSize,
    Orientation,
    LayoutType,
    ElementType,
    LayoutElement,
    BoundingBox,
    Column,
    Margin,
    create_layout_engine,
    generate_sample_layouts
)

from .form_generator import (
    FormGenerator,
    FormConfig,
    FormType,
    FieldType,
    FormField,
    FormSection,
    FormLayout,
    create_form_generator,
    generate_sample_forms
)

from .table_generator import (
    TableGenerator,
    TableConfig,
    TableLayout,
    TableRow,
    TableCell,
    CellType,
    CellAlignment,
    CellStyle,
    BorderStyle,
    TableStyle,
    create_table_generator,
    generate_sample_tables
)

from ...core.logging import get_logger

logger = get_logger(__name__)

# Module version
__version__ = "1.0.0"

# Public API
__all__ = [
    # Layout Engine
    "LayoutEngine",
    "LayoutConfig",
    "Page",
    "PageSize",
    "Orientation",
    "LayoutType",
    "ElementType",
    "LayoutElement",
    "BoundingBox",
    "Column",
    "Margin",
    "create_layout_engine",
    "generate_sample_layouts",
    
    # Form Generator
    "FormGenerator",
    "FormConfig",
    "FormType",
    "FieldType",
    "FormField",
    "FormSection",
    "FormLayout",
    "create_form_generator",
    "generate_sample_forms",
    
    # Table Generator
    "TableGenerator",
    "TableConfig",
    "TableLayout",
    "TableRow",
    "TableCell",
    "CellType",
    "CellAlignment",
    "CellStyle",
    "BorderStyle",
    "TableStyle",
    "create_table_generator",
    "generate_sample_tables",
    
    # Factory functions
    "create_layout_pipeline",
    "create_document_layout",
    "create_form_layout",
    "create_table_layout",
    "generate_all_samples"
]


def create_layout_pipeline(
    layout_config: Optional[Union[Dict[str, Any], LayoutConfig]] = None,
    form_config: Optional[Union[Dict[str, Any], FormConfig]] = None,
    table_config: Optional[Union[Dict[str, Any], TableConfig]] = None
) -> Dict[str, Any]:
    """
    Create a complete layout generation pipeline.
    
    Args:
        layout_config: Configuration for layout engine
        form_config: Configuration for form generator
        table_config: Configuration for table generator
    
    Returns:
        Dictionary with initialized components
    """
    return {
        "layout_engine": create_layout_engine(layout_config),
        "form_generator": create_form_generator(form_config),
        "table_generator": create_table_generator(table_config)
    }


def create_document_layout(
    layout_type: LayoutType = LayoutType.SINGLE_COLUMN,
    content_elements: Optional[List[Dict[str, Any]]] = None,
    page_size: PageSize = PageSize.LETTER,
    orientation: Orientation = Orientation.PORTRAIT,
    **kwargs
) -> List[Page]:
    """
    Create a document layout with specified parameters.
    
    Args:
        layout_type: Type of layout to create
        content_elements: List of content elements to add
        page_size: Page size
        orientation: Page orientation
        **kwargs: Additional configuration options
    
    Returns:
        List of pages with layout
    """
    config = LayoutConfig(
        layout_type=layout_type,
        page_size=page_size,
        orientation=orientation,
        **kwargs
    )
    
    engine = LayoutEngine(config)
    
    if content_elements:
        return engine.create_document_layout(content_elements)
    else:
        # Create single sample page
        return [engine.generate_sample_layout()]


def create_form_layout(
    form_type: FormType = FormType.APPLICATION,
    include_signatures: bool = True,
    **kwargs
) -> FormLayout:
    """
    Create a form layout of specified type.
    
    Args:
        form_type: Type of form to generate
        include_signatures: Whether to include signature fields
        **kwargs: Additional configuration options
    
    Returns:
        Complete form layout
    """
    config = FormConfig(
        form_type=form_type,
        include_signatures=include_signatures,
        **kwargs
    )
    
    generator = FormGenerator(config)
    return generator.generate_form()


def create_table_layout(
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    table_type: Optional[str] = None,
    style: TableStyle = TableStyle.GRID,
    include_headers: bool = True,
    include_totals: bool = True,
    **kwargs
) -> TableLayout:
    """
    Create a table layout with specified parameters.
    
    Args:
        rows: Number of rows (None for random)
        cols: Number of columns (None for random)
        table_type: Predefined table type (e.g., "financial_report")
        style: Table style
        include_headers: Whether to include header row
        include_totals: Whether to include totals row
        **kwargs: Additional configuration options
    
    Returns:
        Complete table layout
    """
    config = TableConfig(
        style=style,
        include_headers=include_headers,
        include_totals=include_totals,
        **kwargs
    )
    
    generator = TableGenerator(config)
    return generator.generate_table(rows, cols, table_type)


def generate_all_samples() -> Dict[str, Any]:
    """
    Generate sample layouts for all types.
    Useful for testing and demonstration.
    
    Returns:
        Dictionary containing sample layouts, forms, and tables
    """
    logger.info("Generating all sample layouts...")
    
    samples = {
        "layouts": {},
        "forms": {},
        "tables": {}
    }
    
    try:
        # Generate layout samples
        samples["layouts"] = generate_sample_layouts()
        logger.info(f"Generated {len(samples['layouts'])} layout samples")
    except Exception as e:
        logger.error(f"Failed to generate layout samples: {str(e)}")
    
    try:
        # Generate form samples
        samples["forms"] = generate_sample_forms()
        logger.info(f"Generated {len(samples['forms'])} form samples")
    except Exception as e:
        logger.error(f"Failed to generate form samples: {str(e)}")
    
    try:
        # Generate table samples
        samples["tables"] = generate_sample_tables()
        logger.info(f"Generated {len(samples['tables'])} table samples")
    except Exception as e:
        logger.error(f"Failed to generate table samples: {str(e)}")
    
    # Add combined examples
    try:
        # Document with mixed content
        layout_engine = create_layout_engine()
        mixed_content = [
            {"type": "heading", "content": "ANNUAL REPORT 2024"},
            {"type": "text", "content": "Executive Summary"},
            {"type": "table", "rows": 5, "cols": 4},
            {"type": "text", "content": "Financial Overview"},
            {"type": "form_field", "label": "Reviewed By", "field_type": "text"},
            {"type": "signature", "labels": ["CEO Signature", "CFO Signature"]}
        ]
        samples["layouts"]["mixed_document"] = layout_engine.create_document_layout(mixed_content)
        
        # Form with embedded table
        form_generator = create_form_generator()
        form = form_generator.generate_form(FormType.ORDER)
        samples["forms"]["order_form_with_table"] = form
        
        logger.info("Generated combined examples")
    except Exception as e:
        logger.error(f"Failed to generate combined examples: {str(e)}")
    
    return samples


# Initialize module
logger.info(f"Initialized layout generation module v{__version__}")
logger.info("Available components: LayoutEngine, FormGenerator, TableGenerator")