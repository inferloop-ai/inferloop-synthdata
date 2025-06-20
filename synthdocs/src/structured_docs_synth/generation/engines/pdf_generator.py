"""
PDF generator for structured document generation
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

from ...core import (
    get_logger,
    get_config,
    DocumentGenerationError,
    ValidationError,
    get_document_type_config
)
from .template_engine import get_template_engine


class PDFGenerator:
    """PDF document generator using ReportLab"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.config = get_config()
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(self.config.generation.output_dir) / "pdf"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize template engine
        self.template_engine = get_template_engine()
        
        # Set up styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        self.logger.info(f"PDF generator initialized with output directory: {self.output_dir}")
    
    def _setup_custom_styles(self):
        """Set up custom paragraph styles"""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=18,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=20,
            alignment=TA_CENTER
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor('#333333'),
            alignment=TA_JUSTIFY
        ))
        
        # Label style
        self.styles.add(ParagraphStyle(
            name='Label',
            parent=self.styles['BodyText'],
            fontSize=10,
            textColor=colors.HexColor('#666666'),
            spaceAfter=2
        ))
        
        # Value style
        self.styles.add(ParagraphStyle(
            name='Value',
            parent=self.styles['BodyText'],
            fontSize=11,
            textColor=colors.HexColor('#000000'),
            spaceAfter=8
        ))
    
    def generate_pdf(
        self,
        document_type: str,
        data: Dict[str, Any],
        output_filename: Optional[str] = None,
        page_size: str = 'letter',
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Generate PDF document from template and data"""
        
        try:
            # Render content from template
            content = self.template_engine.render_template(document_type, data)
            
            # Generate filename if not provided
            if not output_filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"{document_type}_{timestamp}.pdf"
            
            output_path = self.output_dir / output_filename
            
            # Create PDF document
            page_size_obj = letter if page_size == 'letter' else A4
            
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=page_size_obj,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=1*inch,
                bottomMargin=0.75*inch
            )
            
            # Build document story
            story = self._build_document_story(document_type, content, data, metadata)
            
            # Build PDF
            doc.build(story, onFirstPage=self._add_page_info, onLaterPages=self._add_page_info)
            
            self.logger.info(f"Generated PDF: {output_path}")
            return output_path
            
        except Exception as e:
            raise DocumentGenerationError(
                f"Failed to generate PDF for {document_type}: {str(e)}",
                details={
                    'document_type': document_type,
                    'output_filename': output_filename,
                    'error': str(e)
                }
            )
    
    def _build_document_story(
        self,
        document_type: str,
        content: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List:
        """Build the document story (content) for PDF"""
        
        story = []
        
        # Add document header if metadata provided
        if metadata:
            header_data = []
            if 'title' in metadata:
                header_data.append(['Document:', metadata['title']])
            if 'date' in metadata:
                header_data.append(['Date:', metadata['date']])
            if 'reference' in metadata:
                header_data.append(['Reference:', metadata['reference']])
            
            if header_data:
                header_table = Table(header_data, colWidths=[1.5*inch, 4*inch])
                header_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#666666')),
                    ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                    ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                ]))
                story.append(header_table)
                story.append(Spacer(1, 0.5*inch))
        
        # Parse content into sections
        sections = self._parse_content_sections(content)
        
        # Build each section
        for section in sections:
            # Section title
            if section['title']:
                story.append(Paragraph(section['title'], self.styles['CustomHeading']))
            
            # Section content
            for line in section['content']:
                if line.strip():
                    # Check if it's a form field (contains ':')
                    if ':' in line and not line.strip().endswith(':'):
                        parts = line.split(':', 1)
                        label = parts[0].strip()
                        value = parts[1].strip()
                        
                        # Create a two-column layout for form fields
                        field_data = [[f"{label}:", value]]
                        field_table = Table(field_data, colWidths=[2*inch, 4*inch])
                        field_table.setStyle(TableStyle([
                            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, -1), 10),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ('LEFTPADDING', (0, 0), (-1, -1), 0),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                            ('TOPPADDING', (0, 0), (-1, -1), 2),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                        ]))
                        story.append(field_table)
                    else:
                        # Regular paragraph
                        story.append(Paragraph(line, self.styles['CustomBody']))
                else:
                    # Empty line - add small spacer
                    story.append(Spacer(1, 6))
            
            # Add space between sections
            story.append(Spacer(1, 0.2*inch))
        
        # Add signature section for certain document types
        if document_type in ['legal_contract', 'loan_application']:
            story.append(Spacer(1, 0.5*inch))
            story.extend(self._create_signature_section(document_type, data))
        
        return story
    
    def _parse_content_sections(self, content: str) -> List[Dict[str, Any]]:
        """Parse content into sections based on uppercase headers"""
        
        sections = []
        current_section = {'title': '', 'content': []}
        
        lines = content.split('\n')
        
        for line in lines:
            # Check if line is a section header (all uppercase)
            if line.strip() and line.strip().isupper() and len(line.strip()) > 3:
                # Save previous section if it has content
                if current_section['content']:
                    sections.append(current_section)
                # Start new section
                current_section = {'title': line.strip(), 'content': []}
            else:
                # Add line to current section
                current_section['content'].append(line)
        
        # Add last section
        if current_section['content'] or current_section['title']:
            sections.append(current_section)
        
        return sections
    
    def _create_signature_section(self, document_type: str, data: Dict[str, Any]) -> List:
        """Create signature section for documents that require it"""
        
        signature_elements = []
        
        # Add signature title
        signature_elements.append(Paragraph('SIGNATURES', self.styles['CustomHeading']))
        signature_elements.append(Spacer(1, 0.3*inch))
        
        # Determine number of signature lines needed
        if document_type == 'legal_contract' and 'parties' in data:
            parties = data['parties']
        else:
            parties = ['Party 1', 'Party 2']
        
        # Create signature table
        signature_data = []
        for i, party in enumerate(parties[:2]):  # Max 2 parties for now
            signature_data.append([
                f"{party}:",
                "_" * 40,
                "Date:",
                "_" * 20
            ])
            if i < len(parties) - 1:
                signature_data.append(['', '', '', ''])  # Empty row for spacing
        
        signature_table = Table(
            signature_data,
            colWidths=[1.5*inch, 2.5*inch, 0.5*inch, 1.5*inch]
        )
        
        signature_table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'BOTTOM'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 20),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        
        signature_elements.append(signature_table)
        
        return signature_elements
    
    def _add_page_info(self, canvas, doc):
        """Add page header/footer information"""
        
        # Save the state
        canvas.saveState()
        
        # Add page number
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.HexColor('#666666'))
        canvas.drawRightString(
            doc.pagesize[0] - 0.75*inch,
            0.5*inch,
            text
        )
        
        # Add generation timestamp (small, bottom left)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        canvas.setFont('Helvetica', 7)
        canvas.drawString(
            0.75*inch,
            0.5*inch,
            f"Generated: {timestamp}"
        )
        
        # Restore the state
        canvas.restoreState()
    
    def generate_batch(
        self,
        document_type: str,
        data_list: List[Dict[str, Any]],
        output_prefix: str = "",
        **kwargs
    ) -> List[Path]:
        """Generate multiple PDF documents in batch"""
        
        generated_files = []
        
        for i, data in enumerate(data_list):
            try:
                filename = f"{output_prefix}{document_type}_{i+1:04d}.pdf"
                output_path = self.generate_pdf(
                    document_type,
                    data,
                    output_filename=filename,
                    **kwargs
                )
                generated_files.append(output_path)
                
            except Exception as e:
                self.logger.error(f"Failed to generate PDF {i+1}/{len(data_list)}: {str(e)}")
                if self.config.generation.batch_continue_on_error:
                    continue
                else:
                    raise
        
        self.logger.info(f"Generated {len(generated_files)} PDF documents")
        return generated_files