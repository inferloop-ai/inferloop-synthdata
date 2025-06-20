"""
Unit tests for PDF document generation.

Tests the PDFGenerator class for creating synthetic PDF documents
with various layouts, content types, and formatting options.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import io

from structured_docs_synth.generation.engines.pdf_generator import (
    PDFGenerator,
    PDFConfig,
    PDFLayout,
    PDFStyle,
    PDFElement
)
from structured_docs_synth.core.exceptions import GenerationError


class TestPDFConfig:
    """Test PDF configuration."""
    
    def test_default_config(self):
        """Test default PDF configuration."""
        config = PDFConfig()
        
        assert config.page_size == 'A4'
        assert config.orientation == 'portrait'
        assert config.margins == {'top': 72, 'bottom': 72, 'left': 72, 'right': 72}
        assert config.font_family == 'Helvetica'
        assert config.font_size == 12
        assert config.line_spacing == 1.2
        assert config.include_headers is True
        assert config.include_footers is True
        assert config.enable_compression is True
    
    def test_custom_config(self):
        """Test custom PDF configuration."""
        config = PDFConfig(
            page_size='Letter',
            orientation='landscape',
            font_family='Times-Roman',
            font_size=14,
            include_headers=False
        )
        
        assert config.page_size == 'Letter'
        assert config.orientation == 'landscape'
        assert config.font_family == 'Times-Roman'
        assert config.font_size == 14
        assert config.include_headers is False


class TestPDFStyle:
    """Test PDF styling."""
    
    def test_default_style(self):
        """Test default PDF style."""
        style = PDFStyle()
        
        assert style.text_color == (0, 0, 0)
        assert style.background_color == (255, 255, 255)
        assert style.border_color == (0, 0, 0)
        assert style.border_width == 1
        assert style.padding == 5
        assert style.alignment == 'left'
    
    def test_custom_style(self):
        """Test custom PDF style."""
        style = PDFStyle(
            text_color=(255, 0, 0),
            alignment='center',
            border_width=2,
            font_weight='bold'
        )
        
        assert style.text_color == (255, 0, 0)
        assert style.alignment == 'center'
        assert style.border_width == 2
        assert style.font_weight == 'bold'


class TestPDFElement:
    """Test PDF elements."""
    
    def test_text_element(self):
        """Test text element creation."""
        element = PDFElement(
            type='text',
            content='Hello World',
            style=PDFStyle(font_size=16)
        )
        
        assert element.type == 'text'
        assert element.content == 'Hello World'
        assert element.style.font_size == 16
    
    def test_table_element(self):
        """Test table element creation."""
        element = PDFElement(
            type='table',
            content={
                'headers': ['Name', 'Age'],
                'rows': [['John', '30'], ['Jane', '25']]
            },
            style=PDFStyle(border_width=1)
        )
        
        assert element.type == 'table'
        assert element.content['headers'] == ['Name', 'Age']
        assert len(element.content['rows']) == 2
    
    def test_image_element(self):
        """Test image element creation."""
        element = PDFElement(
            type='image',
            content={'path': '/tmp/image.png', 'width': 200, 'height': 150},
            style=PDFStyle(alignment='center')
        )
        
        assert element.type == 'image'
        assert element.content['path'] == '/tmp/image.png'
        assert element.content['width'] == 200


class TestPDFGenerator:
    """Test PDF generator."""
    
    @pytest.fixture
    def pdf_config(self):
        """Provide PDF configuration."""
        return PDFConfig(
            page_size='A4',
            font_family='Helvetica',
            font_size=12
        )
    
    @pytest.fixture
    def pdf_generator(self, pdf_config):
        """Provide PDF generator instance."""
        return PDFGenerator(pdf_config)
    
    @pytest.fixture
    def mock_canvas(self):
        """Provide mock PDF canvas."""
        with patch('reportlab.pdfgen.canvas.Canvas') as mock:
            canvas = MagicMock()
            mock.return_value = canvas
            yield canvas
    
    def test_generator_initialization(self, pdf_generator, pdf_config):
        """Test generator initialization."""
        assert pdf_generator.config == pdf_config
        assert pdf_generator.current_page == 1
        assert pdf_generator.elements == []
    
    def test_add_text_element(self, pdf_generator):
        """Test adding text element."""
        pdf_generator.add_text(
            content="Test Document",
            style=PDFStyle(font_size=16, font_weight='bold')
        )
        
        assert len(pdf_generator.elements) == 1
        assert pdf_generator.elements[0].type == 'text'
        assert pdf_generator.elements[0].content == "Test Document"
        assert pdf_generator.elements[0].style.font_size == 16
    
    def test_add_paragraph(self, pdf_generator):
        """Test adding paragraph."""
        text = "This is a long paragraph that should be wrapped properly."
        pdf_generator.add_paragraph(text)
        
        assert len(pdf_generator.elements) == 1
        assert pdf_generator.elements[0].type == 'paragraph'
        assert pdf_generator.elements[0].content == text
    
    def test_add_table(self, pdf_generator):
        """Test adding table."""
        headers = ['Product', 'Price', 'Quantity']
        rows = [
            ['Item A', '$10', '5'],
            ['Item B', '$20', '3']
        ]
        
        pdf_generator.add_table(headers=headers, rows=rows)
        
        assert len(pdf_generator.elements) == 1
        assert pdf_generator.elements[0].type == 'table'
        assert pdf_generator.elements[0].content['headers'] == headers
        assert pdf_generator.elements[0].content['rows'] == rows
    
    def test_add_image(self, pdf_generator, temp_dir):
        """Test adding image."""
        # Create dummy image file
        image_path = temp_dir / "test_image.png"
        image_path.write_bytes(b"fake image data")
        
        pdf_generator.add_image(
            image_path=str(image_path),
            width=200,
            height=150,
            alignment='center'
        )
        
        assert len(pdf_generator.elements) == 1
        assert pdf_generator.elements[0].type == 'image'
        assert pdf_generator.elements[0].content['path'] == str(image_path)
        assert pdf_generator.elements[0].content['width'] == 200
    
    def test_add_line_break(self, pdf_generator):
        """Test adding line break."""
        pdf_generator.add_line_break()
        
        assert len(pdf_generator.elements) == 1
        assert pdf_generator.elements[0].type == 'line_break'
    
    def test_add_page_break(self, pdf_generator):
        """Test adding page break."""
        pdf_generator.add_page_break()
        
        assert len(pdf_generator.elements) == 1
        assert pdf_generator.elements[0].type == 'page_break'
    
    @patch('reportlab.pdfgen.canvas.Canvas')
    @patch('reportlab.platypus.SimpleDocTemplate')
    def test_generate_pdf(self, mock_doc_template, mock_canvas, pdf_generator, temp_dir):
        """Test PDF generation."""
        # Add content
        pdf_generator.add_text("Test Document", PDFStyle(font_size=20))
        pdf_generator.add_paragraph("This is a test paragraph.")
        pdf_generator.add_table(['Col1', 'Col2'], [['A', 'B']])
        
        # Mock document template
        mock_doc = MagicMock()
        mock_doc_template.return_value = mock_doc
        
        # Generate PDF
        output_path = temp_dir / "test.pdf"
        result = pdf_generator.generate(str(output_path))
        
        assert result['success'] is True
        assert result['output_path'] == str(output_path)
        assert result['page_count'] >= 1
        assert 'file_size' in result
        
        # Verify document build was called
        mock_doc.build.assert_called_once()
    
    def test_generate_with_metadata(self, pdf_generator, temp_dir):
        """Test PDF generation with metadata."""
        metadata = {
            'title': 'Test Document',
            'author': 'Test Author',
            'subject': 'Unit Testing',
            'keywords': ['test', 'pdf', 'generation']
        }
        
        pdf_generator.add_text("Test Content")
        
        with patch('reportlab.platypus.SimpleDocTemplate') as mock_doc:
            mock_doc_instance = MagicMock()
            mock_doc.return_value = mock_doc_instance
            
            output_path = temp_dir / "test_meta.pdf"
            result = pdf_generator.generate(
                str(output_path),
                metadata=metadata
            )
            
            assert result['success'] is True
            assert result['metadata'] == metadata
    
    def test_add_header_footer(self, pdf_generator):
        """Test adding headers and footers."""
        pdf_generator.set_header("Document Header", style=PDFStyle(alignment='center'))
        pdf_generator.set_footer("Page {page_number}", style=PDFStyle(font_size=10))
        
        assert pdf_generator.header_content == "Document Header"
        assert pdf_generator.footer_content == "Page {page_number}"
    
    def test_apply_watermark(self, pdf_generator, temp_dir):
        """Test applying watermark."""
        pdf_generator.add_text("Document content")
        pdf_generator.apply_watermark(
            text="CONFIDENTIAL",
            opacity=0.3,
            angle=45
        )
        
        assert pdf_generator.watermark_text == "CONFIDENTIAL"
        assert pdf_generator.watermark_opacity == 0.3
        assert pdf_generator.watermark_angle == 45
    
    def test_generate_form_fields(self, pdf_generator):
        """Test generating form fields."""
        pdf_generator.add_form_field(
            field_type='text',
            name='full_name',
            label='Full Name:',
            required=True
        )
        
        pdf_generator.add_form_field(
            field_type='checkbox',
            name='agree',
            label='I agree to terms'
        )
        
        assert len(pdf_generator.form_fields) == 2
        assert pdf_generator.form_fields[0]['name'] == 'full_name'
        assert pdf_generator.form_fields[1]['field_type'] == 'checkbox'
    
    def test_error_handling(self, pdf_generator):
        """Test error handling during generation."""
        # Add invalid content
        pdf_generator.elements.append(
            PDFElement(type='invalid', content='test')
        )
        
        with patch('reportlab.platypus.SimpleDocTemplate') as mock_doc:
            mock_doc.side_effect = Exception("PDF generation failed")
            
            result = pdf_generator.generate("/tmp/test.pdf")
            
            assert result['success'] is False
            assert 'PDF generation failed' in result['error']
    
    def test_page_numbering(self, pdf_generator):
        """Test automatic page numbering."""
        # Add content that spans multiple pages
        for i in range(100):
            pdf_generator.add_paragraph(f"Paragraph {i}" * 50)
        
        pdf_generator.enable_page_numbering(
            format="Page {current} of {total}",
            position='bottom-center'
        )
        
        assert pdf_generator.page_numbering_enabled is True
        assert pdf_generator.page_number_format == "Page {current} of {total}"
    
    def test_custom_fonts(self, pdf_generator):
        """Test using custom fonts."""
        with patch('reportlab.pdfbase.pdfmetrics.registerFont') as mock_register:
            pdf_generator.register_font(
                font_name='CustomFont',
                font_path='/path/to/font.ttf'
            )
            
            mock_register.assert_called_once()
            assert 'CustomFont' in pdf_generator.registered_fonts
    
    def test_generate_invoice_template(self, pdf_generator, temp_dir):
        """Test generating invoice template."""
        # Create invoice layout
        pdf_generator.add_text("INVOICE", PDFStyle(font_size=24, alignment='center'))
        pdf_generator.add_line_break()
        
        # Company info
        pdf_generator.add_text("ABC Company Ltd.", PDFStyle(font_size=14))
        pdf_generator.add_text("123 Business St.", PDFStyle(font_size=12))
        pdf_generator.add_line_break()
        
        # Invoice details
        pdf_generator.add_table(
            headers=['Description', 'Quantity', 'Price', 'Total'],
            rows=[
                ['Service A', '10', '$50', '$500'],
                ['Service B', '5', '$100', '$500']
            ],
            style=PDFStyle(border_width=1)
        )
        
        # Total
        pdf_generator.add_text("Total: $1000", PDFStyle(alignment='right', font_weight='bold'))
        
        with patch('reportlab.platypus.SimpleDocTemplate'):
            output_path = temp_dir / "invoice.pdf"
            result = pdf_generator.generate(str(output_path))
            
            assert result['success'] is True
            assert len(pdf_generator.elements) > 5