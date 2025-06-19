#!/usr/bin/env python3
"""
PDF renderer for generating high-quality PDF documents.

Provides capabilities for creating PDF documents from templates,
HTML content, or structured data with proper formatting, fonts,
and layout preservation.
"""

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4, legal
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.platypus.flowables import PageBreak
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from ...core import get_logger
from ...utils.file_utils import write_file

logger = get_logger(__name__)


class PDFRenderer:
    """High-quality PDF document renderer"""
    
    def __init__(self, page_size: str = 'letter', margins: Dict[str, float] = None):
        self.page_size = self._get_page_size(page_size)
        self.margins = margins or {
            'top': 1.0 * inch,
            'bottom': 1.0 * inch,
            'left': 1.0 * inch,
            'right': 1.0 * inch
        }
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self._register_fonts()
    
    def _get_page_size(self, page_size: str):
        """Convert page size string to reportlab size"""
        sizes = {
            'letter': letter,
            'a4': A4,
            'legal': legal
        }
        return sizes.get(page_size.lower(), letter)
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomSubheading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=16,
            spaceAfter=8,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceBefore=6,
            spaceAfter=6,
            alignment=0,  # Left
            leftIndent=0.25 * inch
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomFooter',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=1  # Center
        ))
    
    def _register_fonts(self):
        """Register custom fonts if available"""
        try:
            # Try to register common system fonts
            font_paths = [
                '/System/Library/Fonts/Arial.ttf',  # macOS
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',  # Linux
                'C:/Windows/Fonts/arial.ttf'  # Windows
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont('Arial', font_path))
                    break
                    
        except Exception as e:
            logger.debug(f"Could not register custom fonts: {e}")
    
    async def render_document(self, content: Dict[str, Any], 
                            output_path: Union[str, Path],
                            template: Optional[str] = None) -> bool:
        """
        Render document content to PDF.
        
        Args:
            content: Document content dictionary
            output_path: Output PDF file path
            template: Template name to use
        
        Returns:
            True if successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=self.page_size,
                rightMargin=self.margins['right'],
                leftMargin=self.margins['left'],
                topMargin=self.margins['top'],
                bottomMargin=self.margins['bottom']
            )
            
            # Build document content
            story = self._build_document_story(content, template)
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Successfully rendered PDF: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to render PDF {output_path}: {e}")
            raise
    
    def _build_document_story(self, content: Dict[str, Any], 
                            template: Optional[str]) -> List:
        """Build document story from content"""
        story = []
        
        # Add title if present
        if 'title' in content:
            title = Paragraph(content['title'], self.styles['CustomTitle'])
            story.append(title)
            story.append(Spacer(1, 20))
        
        # Add metadata section
        if 'metadata' in content:
            story.extend(self._add_metadata_section(content['metadata']))
        
        # Add main content
        if 'content' in content:
            story.extend(self._process_content(content['content']))
        
        # Add sections if present
        if 'sections' in content:
            for section in content['sections']:
                story.extend(self._add_section(section))
        
        # Add tables if present
        if 'tables' in content:
            for table_data in content['tables']:
                story.extend(self._add_table(table_data))
        
        # Add annotations if present
        if 'annotations' in content:
            story.extend(self._add_annotations_section(content['annotations']))
        
        # Add footer
        story.append(self._create_footer(content))
        
        return story
    
    def _add_metadata_section(self, metadata: Dict[str, Any]) -> List:
        """Add metadata section to document"""
        elements = []
        
        # Create metadata table
        data = []
        for key, value in metadata.items():
            if key not in ['tags', 'annotations']:
                display_key = key.replace('_', ' ').title()
                data.append([display_key, str(value)])
        
        if data:
            elements.append(Paragraph("Document Information", self.styles['CustomHeading']))
            
            table = Table(data, colWidths=[2*inch, 4*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 20))
        
        return elements
    
    def _process_content(self, content: Union[str, List, Dict]) -> List:
        """Process various content types"""
        elements = []
        
        if isinstance(content, str):
            # Simple text content
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    elements.append(Paragraph(para.strip(), self.styles['CustomBody']))
                    elements.append(Spacer(1, 6))
        
        elif isinstance(content, list):
            # List of content items
            for item in content:
                elements.extend(self._process_content(item))
        
        elif isinstance(content, dict):
            # Structured content
            if 'type' in content:
                if content['type'] == 'paragraph':
                    elements.append(Paragraph(content['text'], self.styles['CustomBody']))
                elif content['type'] == 'heading':
                    level = content.get('level', 1)
                    style = 'CustomHeading' if level <= 2 else 'CustomSubheading'
                    elements.append(Paragraph(content['text'], self.styles[style]))
                elif content['type'] == 'list':
                    elements.extend(self._add_list(content['items']))
        
        return elements
    
    def _add_section(self, section: Dict[str, Any]) -> List:
        """Add a document section"""
        elements = []
        
        # Section title
        if 'title' in section:
            elements.append(Paragraph(section['title'], self.styles['CustomHeading']))
            elements.append(Spacer(1, 12))
        
        # Section content
        if 'content' in section:
            elements.extend(self._process_content(section['content']))
        
        elements.append(Spacer(1, 20))
        return elements
    
    def _add_table(self, table_data: Dict[str, Any]) -> List:
        """Add a table to the document"""
        elements = []
        
        # Table title
        if 'title' in table_data:
            elements.append(Paragraph(table_data['title'], self.styles['CustomSubheading']))
            elements.append(Spacer(1, 8))
        
        # Table data
        if 'data' in table_data:
            data = table_data['data']
            headers = table_data.get('headers', [])
            
            # Prepare table data
            table_rows = []
            if headers:
                table_rows.append(headers)
            table_rows.extend(data)
            
            # Create table
            table = Table(table_rows)
            
            # Style table
            style_commands = [
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]
            
            # Header row styling
            if headers:
                style_commands.extend([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
                ])
            
            table.setStyle(TableStyle(style_commands))
            elements.append(table)
            elements.append(Spacer(1, 20))
        
        return elements
    
    def _add_list(self, items: List[str]) -> List:
        """Add a bulleted list"""
        elements = []
        
        for item in items:
            bullet_text = f"" {item}"
            elements.append(Paragraph(bullet_text, self.styles['CustomBody']))
        
        elements.append(Spacer(1, 12))
        return elements
    
    def _add_annotations_section(self, annotations: List[Dict[str, Any]]) -> List:
        """Add annotations section"""
        elements = []
        
        if annotations:
            elements.append(PageBreak())
            elements.append(Paragraph("Annotations", self.styles['CustomHeading']))
            elements.append(Spacer(1, 12))
            
            # Create annotations table
            headers = ['ID', 'Type', 'Bounds', 'Confidence']
            data = [headers]
            
            for ann in annotations:
                row = [
                    ann.get('id', 'N/A')[:8],  # Truncate ID
                    ann.get('category_name', 'unknown'),
                    f"{ann.get('bbox', [0,0,0,0])}",
                    f"{ann.get('confidence', 0):.2f}"
                ]
                data.append(row)
            
            table = Table(data, colWidths=[1.2*inch, 1.5*inch, 2*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(table)
        
        return elements
    
    def _create_footer(self, content: Dict[str, Any]) -> Spacer:
        """Create document footer"""
        return Spacer(1, 20)
    
    async def render_from_html(self, html_content: str, 
                             output_path: Union[str, Path]) -> bool:
        """
        Render HTML content to PDF.
        
        Args:
            html_content: HTML content string
            output_path: Output PDF file path
        
        Returns:
            True if successful
        """
        try:
            # Convert HTML to document structure
            content = self._parse_html_content(html_content)
            return await self.render_document(content, output_path)
            
        except Exception as e:
            logger.error(f"Failed to render HTML to PDF: {e}")
            raise
    
    def _parse_html_content(self, html_content: str) -> Dict[str, Any]:
        """Parse HTML content into document structure"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        content = {
            'title': '',
            'content': [],
            'sections': []
        }
        
        # Extract title
        title_tag = soup.find('title') or soup.find('h1')
        if title_tag:
            content['title'] = title_tag.get_text(strip=True)
        
        # Extract content
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div']):
            text = element.get_text(strip=True)
            if text:
                if element.name.startswith('h'):
                    level = int(element.name[1])
                    content['content'].append({
                        'type': 'heading',
                        'level': level,
                        'text': text
                    })
                else:
                    content['content'].append({
                        'type': 'paragraph',
                        'text': text
                    })
        
        return content
    
    async def render_financial_report(self, data: Dict[str, Any], 
                                    output_path: Union[str, Path]) -> bool:
        """
        Render specialized financial report.
        
        Args:
            data: Financial data
            output_path: Output PDF file path
        
        Returns:
            True if successful
        """
        try:
            # Create financial report structure
            content = {
                'title': data.get('title', 'Financial Report'),
                'metadata': {
                    'report_date': data.get('report_date', datetime.now().strftime('%Y-%m-%d')),
                    'company': data.get('company', 'Sample Corporation'),
                    'period': data.get('period', 'Q1 2024'),
                    'currency': data.get('currency', 'USD')
                },
                'sections': [
                    {
                        'title': 'Executive Summary',
                        'content': data.get('executive_summary', 'Financial performance summary...')
                    },
                    {
                        'title': 'Revenue Analysis',
                        'content': data.get('revenue_analysis', 'Revenue analysis details...')
                    }
                ],
                'tables': [
                    {
                        'title': 'Financial Summary',
                        'headers': ['Metric', 'Current Period', 'Previous Period', 'Change'],
                        'data': data.get('financial_data', [
                            ['Revenue', '$1,000,000', '$900,000', '+11.1%'],
                            ['Expenses', '$800,000', '$750,000', '+6.7%'],
                            ['Net Income', '$200,000', '$150,000', '+33.3%']
                        ])
                    }
                ]
            }
            
            return await self.render_document(content, output_path)
            
        except Exception as e:
            logger.error(f"Failed to render financial report: {e}")
            raise
    
    async def render_to_bytes(self, content: Dict[str, Any]) -> bytes:
        """
        Render document to bytes instead of file.
        
        Args:
            content: Document content
        
        Returns:
            PDF content as bytes
        """
        try:
            buffer = io.BytesIO()
            
            doc = SimpleDocTemplate(
                buffer,
                pagesize=self.page_size,
                rightMargin=self.margins['right'],
                leftMargin=self.margins['left'],
                topMargin=self.margins['top'],
                bottomMargin=self.margins['bottom']
            )
            
            story = self._build_document_story(content, None)
            doc.build(story)
            
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to render PDF to bytes: {e}")
            raise


def create_pdf_renderer(page_size: str = 'letter', 
                       margins: Optional[Dict[str, float]] = None) -> PDFRenderer:
    """Factory function to create PDF renderer"""
    return PDFRenderer(page_size=page_size, margins=margins)


__all__ = ['PDFRenderer', 'create_pdf_renderer']