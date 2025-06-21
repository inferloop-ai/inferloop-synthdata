#!/usr/bin/env python3
"""
HTML Exporter for document export functionality.

Provides HTML export capabilities with templates, styling,
and interactive features.
"""

import html
import base64
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json
import re
from urllib.parse import quote

from ...core.logging import get_logger
from ...core.exceptions import ValidationError, ProcessingError


logger = get_logger(__name__)


# Default HTML template
DEFAULT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        {styles}
    </style>
    {custom_head}
</head>
<body>
    <div class="container">
        {content}
    </div>
    {scripts}
</body>
</html>"""

# Default CSS styles
DEFAULT_STYLES = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f5f5f5;
}

.container {
    background-color: white;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

h1, h2, h3, h4, h5, h6 {
    color: #2c3e50;
    margin-top: 24px;
    margin-bottom: 16px;
}

h1 { font-size: 2.5em; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; }
h2 { font-size: 2em; }
h3 { font-size: 1.5em; }

p {
    margin-bottom: 16px;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
}

table th,
table td {
    border: 1px solid #ddd;
    padding: 12px;
    text-align: left;
}

table th {
    background-color: #f8f9fa;
    font-weight: bold;
}

table tr:nth-child(even) {
    background-color: #f8f9fa;
}

.metadata {
    background-color: #e3f2fd;
    padding: 15px;
    border-radius: 4px;
    margin-bottom: 20px;
}

.metadata h3 {
    margin-top: 0;
}

.section {
    margin-bottom: 30px;
}

.image-container {
    text-align: center;
    margin: 20px 0;
}

.image-container img {
    max-width: 100%;
    height: auto;
    border: 1px solid #ddd;
    border-radius: 4px;
}

.image-caption {
    font-style: italic;
    color: #666;
    margin-top: 10px;
}

code {
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Courier New', Courier, monospace;
}

pre {
    background-color: #f4f4f4;
    padding: 15px;
    border-radius: 4px;
    overflow-x: auto;
}

.highlight {
    background-color: #fff3cd;
    padding: 2px 4px;
}

.footer {
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #e0e0e0;
    text-align: center;
    color: #666;
    font-size: 0.9em;
}
"""


class HTMLExporter:
    """
    HTML document exporter with rich formatting capabilities.
    
    Features:
    - Template-based export
    - Custom styling
    - Table formatting
    - Image embedding
    - Interactive elements
    - Multi-page support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize HTML exporter"""
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Template settings
        self.template = self.config.get('template', DEFAULT_TEMPLATE)
        self.styles = self.config.get('styles', DEFAULT_STYLES)
        self.custom_head = self.config.get('custom_head', '')
        self.scripts = self.config.get('scripts', '')
        
        # Export settings
        self.encoding = self.config.get('encoding', 'utf-8')
        self.embed_images = self.config.get('embed_images', True)
        self.include_metadata = self.config.get('include_metadata', True)
        self.syntax_highlighting = self.config.get('syntax_highlighting', True)
        self.table_of_contents = self.config.get('table_of_contents', True)
        
        # Styling options
        self.theme = self.config.get('theme', 'default')
        self.custom_css = self.config.get('custom_css', '')
        
        self.logger.info("HTML exporter initialized")
    
    def export_document(
        self,
        data: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
        standalone: bool = True
    ) -> str:
        """
        Export document as HTML.
        
        Args:
            data: Document data
            output_path: Optional output file path
            title: Document title
            standalone: Create standalone HTML document
            
        Returns:
            HTML string
        """
        try:
            # Extract title if not provided
            if not title:
                title = data.get('metadata', {}).get('title', 'Document')
            
            # Generate content
            content_html = self._generate_content(data)
            
            if standalone:
                # Create full HTML document
                html_content = self.template.format(
                    title=html.escape(title),
                    styles=self.styles + '\n' + self.custom_css,
                    custom_head=self.custom_head,
                    content=content_html,
                    scripts=self.scripts
                )
            else:
                # Return content only
                html_content = content_html
            
            # Save to file if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding=self.encoding) as f:
                    f.write(html_content)
                
                self.logger.info(f"HTML exported to {output_path}")
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"HTML export failed: {e}")
            raise ProcessingError(f"Failed to export HTML: {e}")
    
    def export_batch(
        self,
        documents: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        index_file: str = "index.html",
        separate_files: bool = True
    ) -> List[Path]:
        """
        Export multiple documents with index page.
        
        Args:
            documents: List of documents
            output_dir: Output directory
            index_file: Index page filename
            separate_files: Create separate file for each document
            
        Returns:
            List of exported file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        if separate_files:
            # Export each document separately
            doc_links = []
            
            for i, doc in enumerate(documents):
                filename = f"document_{i}.html"
                file_path = output_dir / filename
                
                title = doc.get('metadata', {}).get('title', f'Document {i}')
                self.export_document(doc, file_path, title)
                
                exported_files.append(file_path)
                doc_links.append({
                    'title': title,
                    'filename': filename,
                    'metadata': doc.get('metadata', {})
                })
            
            # Create index page
            index_content = self._create_index_page(doc_links)
            index_path = output_dir / index_file
            
            with open(index_path, 'w', encoding=self.encoding) as f:
                f.write(index_content)
            
            exported_files.append(index_path)
        else:
            # Export all documents in single file
            combined_path = output_dir / index_file
            combined_content = self._create_combined_document(documents)
            
            with open(combined_path, 'w', encoding=self.encoding) as f:
                f.write(combined_content)
            
            exported_files.append(combined_path)
        
        self.logger.info(f"Exported {len(documents)} HTML documents to {output_dir}")
        
        return exported_files
    
    def _generate_content(self, data: Dict[str, Any]) -> str:
        """Generate HTML content from document data"""
        sections = []
        
        # Add metadata section
        if self.include_metadata and 'metadata' in data:
            sections.append(self._format_metadata(data['metadata']))
        
        # Add table of contents
        if self.table_of_contents and 'sections' in data:
            sections.append(self._generate_toc(data['sections']))
        
        # Add main content
        if 'content' in data:
            sections.append(self._format_content(data['content']))
        
        # Add sections
        if 'sections' in data:
            sections.append(self._format_sections(data['sections']))
        
        # Add tables
        if 'tables' in data:
            sections.append(self._format_tables(data['tables']))
        
        # Add images
        if 'images' in data:
            sections.append(self._format_images(data['images']))
        
        # Add footer
        sections.append(self._generate_footer())
        
        return '\n'.join(filter(None, sections))
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata section"""
        html_parts = ['<div class="metadata">']
        html_parts.append('<h3>Document Information</h3>')
        html_parts.append('<dl>')
        
        for key, value in metadata.items():
            html_parts.append(f'<dt>{html.escape(str(key).title())}:</dt>')
            html_parts.append(f'<dd>{html.escape(str(value))}</dd>')
        
        html_parts.append('</dl>')
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def _generate_toc(self, sections: List[Dict[str, Any]]) -> str:
        """Generate table of contents"""
        html_parts = ['<div class="toc">']
        html_parts.append('<h3>Table of Contents</h3>')
        html_parts.append('<ol>')
        
        for i, section in enumerate(sections):
            section_id = section.get('id', f'section-{i}')
            title = section.get('title', f'Section {i+1}')
            
            html_parts.append(f'<li><a href="#{section_id}">{html.escape(title)}</a>')
            
            # Add subsections
            if 'subsections' in section:
                html_parts.append('<ol>')
                for j, subsection in enumerate(section['subsections']):
                    subsection_id = subsection.get('id', f'{section_id}-{j}')
                    subtitle = subsection.get('title', f'Subsection {j+1}')
                    html_parts.append(f'<li><a href="#{subsection_id}">{html.escape(subtitle)}</a></li>')
                html_parts.append('</ol>')
            
            html_parts.append('</li>')
        
        html_parts.append('</ol>')
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def _format_content(self, content: Union[str, Dict[str, Any]]) -> str:
        """Format main content"""
        if isinstance(content, str):
            return f'<div class="content">{self._process_text(content)}</div>'
        
        elif isinstance(content, dict):
            html_parts = ['<div class="content">']
            
            if 'title' in content:
                html_parts.append(f'<h1>{html.escape(content["title"])}</h1>')
            
            if 'text' in content:
                html_parts.append(self._process_text(content['text']))
            
            if 'html' in content:
                # Raw HTML content
                html_parts.append(content['html'])
            
            if 'structured' in content:
                html_parts.append(self._format_structured_content(content['structured']))
            
            html_parts.append('</div>')
            
            return '\n'.join(html_parts)
        
        return ''
    
    def _format_sections(self, sections: List[Dict[str, Any]], level: int = 2) -> str:
        """Format document sections"""
        html_parts = []
        
        for i, section in enumerate(sections):
            section_id = section.get('id', f'section-{i}')
            
            html_parts.append(f'<div class="section" id="{section_id}">')
            
            if 'title' in section:
                html_parts.append(f'<h{level}>{html.escape(section["title"])}</h{level}>')
            
            if 'content' in section:
                html_parts.append(self._format_content(section['content']))
            
            if 'subsections' in section:
                html_parts.append(self._format_sections(section['subsections'], level + 1))
            
            html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def _format_tables(self, tables: List[Dict[str, Any]]) -> str:
        """Format tables"""
        html_parts = []
        
        for table in tables:
            html_parts.append('<div class="table-container">')
            
            if 'caption' in table:
                html_parts.append(f'<h4>{html.escape(table["caption"])}</h4>')
            
            html_parts.append('<table>')
            
            # Headers
            if 'headers' in table:
                html_parts.append('<thead><tr>')
                for header in table['headers']:
                    html_parts.append(f'<th>{html.escape(str(header))}</th>')
                html_parts.append('</tr></thead>')
            
            # Rows
            if 'rows' in table:
                html_parts.append('<tbody>')
                for row in table['rows']:
                    html_parts.append('<tr>')
                    for cell in row:
                        html_parts.append(f'<td>{html.escape(str(cell))}</td>')
                    html_parts.append('</tr>')
                html_parts.append('</tbody>')
            
            html_parts.append('</table>')
            html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def _format_images(self, images: List[Dict[str, Any]]) -> str:
        """Format images"""
        html_parts = []
        
        for image in images:
            html_parts.append('<div class="image-container">')
            
            if 'data' in image and self.embed_images:
                # Embed image as data URL
                mime_type = image.get('mime_type', 'image/png')
                
                if isinstance(image['data'], bytes):
                    img_data = base64.b64encode(image['data']).decode('ascii')
                else:
                    img_data = image['data']
                
                img_src = f'data:{mime_type};base64,{img_data}'
            else:
                # External image path
                img_src = image.get('path', '')
            
            alt_text = html.escape(image.get('alt_text', 'Image'))
            html_parts.append(f'<img src="{img_src}" alt="{alt_text}">')
            
            if 'caption' in image:
                html_parts.append(f'<div class="image-caption">{html.escape(image["caption"])}</div>')
            
            html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def _format_structured_content(self, content: Any) -> str:
        """Format structured content recursively"""
        if isinstance(content, dict):
            html_parts = ['<dl>']
            for key, value in content.items():
                html_parts.append(f'<dt>{html.escape(str(key))}:</dt>')
                html_parts.append(f'<dd>{self._format_structured_content(value)}</dd>')
            html_parts.append('</dl>')
            return '\n'.join(html_parts)
        
        elif isinstance(content, list):
            html_parts = ['<ul>']
            for item in content:
                html_parts.append(f'<li>{self._format_structured_content(item)}</li>')
            html_parts.append('</ul>')
            return '\n'.join(html_parts)
        
        else:
            return html.escape(str(content))
    
    def _process_text(self, text: str) -> str:
        """Process text with formatting"""
        # Escape HTML
        processed = html.escape(text)
        
        # Convert line breaks
        processed = processed.replace('\n\n', '</p><p>')
        processed = processed.replace('\n', '<br>')
        processed = f'<p>{processed}</p>'
        
        # Apply syntax highlighting if enabled
        if self.syntax_highlighting:
            # Simple code block detection
            processed = re.sub(
                r'```(\w+)?\n(.*?)\n```',
                lambda m: f'<pre><code class="language-{m.group(1) or "text"}">{html.escape(m.group(2))}</code></pre>',
                processed,
                flags=re.DOTALL
            )
            
            # Inline code
            processed = re.sub(
                r'`([^`]+)`',
                r'<code>\1</code>',
                processed
            )
        
        # Apply highlighting
        # Simple **bold** syntax
        processed = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', processed)
        
        # Simple *italic* syntax
        processed = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', processed)
        
        return processed
    
    def _generate_footer(self) -> str:
        """Generate footer"""
        return f'''
        <div class="footer">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        '''
    
    def _create_index_page(self, doc_links: List[Dict[str, Any]]) -> str:
        """Create index page for multiple documents"""
        content = ['<h1>Document Index</h1>']
        content.append('<div class="document-list">')
        content.append('<ul>')
        
        for link in doc_links:
            title = html.escape(link['title'])
            filename = link['filename']
            
            content.append(f'<li><a href="{filename}">{title}</a>')
            
            # Add metadata preview
            if link.get('metadata'):
                meta_items = []
                for key, value in link['metadata'].items():
                    if key != 'title':
                        meta_items.append(f'{key}: {value}')
                
                if meta_items:
                    content.append(f' <small>({", ".join(meta_items[:3])})</small>')
            
            content.append('</li>')
        
        content.append('</ul>')
        content.append('</div>')
        
        return self.template.format(
            title='Document Index',
            styles=self.styles + '\n' + self.custom_css,
            custom_head=self.custom_head,
            content='\n'.join(content),
            scripts=self.scripts
        )
    
    def _create_combined_document(self, documents: List[Dict[str, Any]]) -> str:
        """Create single HTML with all documents"""
        content_parts = ['<h1>Document Collection</h1>']
        
        for i, doc in enumerate(documents):
            content_parts.append(f'<div class="document-section" id="doc-{i}">')
            content_parts.append(f'<h2>Document {i+1}</h2>')
            content_parts.append(self._generate_content(doc))
            content_parts.append('</div>')
            content_parts.append('<hr>')
        
        return self.template.format(
            title='Document Collection',
            styles=self.styles + '\n' + self.custom_css,
            custom_head=self.custom_head,
            content='\n'.join(content_parts),
            scripts=self.scripts
        )


# Factory function
def create_html_exporter(config: Optional[Dict[str, Any]] = None) -> HTMLExporter:
    """Create and return an HTML exporter instance"""
    return HTMLExporter(config)