#!/usr/bin/env python3
"""
LaTeX generator for creating high-quality structured documents.

Provides capabilities for generating LaTeX documents with proper
formatting, mathematical expressions, tables, and academic styling.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ...core import get_logger
from ...utils.file_utils import write_file, read_file

logger = get_logger(__name__)


class LaTeXGenerator:
    """High-quality LaTeX document generator"""
    
    def __init__(self, template_dir: Optional[str] = None):
        self.template_dir = Path(template_dir) if template_dir else None
        self.packages = [
            'amsmath', 'amsfonts', 'amssymb', 'graphicx', 'geometry',
            'fancyhdr', 'titlesec', 'enumitem', 'booktabs', 'array',
            'longtable', 'multirow', 'hyperref', 'xcolor', 'listings'
        ]
        
    async def generate_document(self, content: Dict[str, Any], 
                              output_path: Union[str, Path],
                              compile_pdf: bool = True,
                              template: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate LaTeX document from content.
        
        Args:
            content: Document content dictionary
            output_path: Output file path
            compile_pdf: Whether to compile to PDF
            template: Template name to use
        
        Returns:
            Dictionary with generation results
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate LaTeX source
            latex_source = self._build_latex_document(content, template)
            
            # Write LaTeX file
            tex_file = output_path.with_suffix('.tex')
            write_file(tex_file, latex_source)
            
            result = {
                'latex_file': str(tex_file),
                'latex_source': latex_source,
                'success': True
            }
            
            # Compile to PDF if requested
            if compile_pdf:
                pdf_result = await self._compile_latex(tex_file)
                result.update(pdf_result)
            
            logger.info(f"Generated LaTeX document: {tex_file}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate LaTeX document: {e}")
            raise
    
    def _build_latex_document(self, content: Dict[str, Any], 
                            template: Optional[str]) -> str:
        """Build complete LaTeX document"""
        doc_class = content.get('document_class', 'article')
        doc_options = content.get('document_options', ['11pt', 'a4paper'])
        
        # Start document
        latex = f"\\documentclass[{','.join(doc_options)}]{{{doc_class}}}\n\n"
        
        # Add packages
        latex += self._add_packages(content.get('packages', []))
        
        # Add custom commands and settings
        latex += self._add_preamble(content)
        
        # Begin document
        latex += "\\begin{document}\n\n"
        
        # Add title page
        if 'title' in content or 'author' in content:
            latex += self._add_title_page(content)
        
        # Add table of contents
        if content.get('include_toc', False):
            latex += "\\tableofcontents\n\\newpage\n\n"
        
        # Add main content
        latex += self._add_content(content)
        
        # Add bibliography if present
        if 'bibliography' in content:
            latex += self._add_bibliography(content['bibliography'])
        
        # End document
        latex += "\\end{document}\n"
        
        return latex
    
    def _add_packages(self, additional_packages: List[str]) -> str:
        """Add LaTeX packages"""
        all_packages = self.packages + additional_packages
        latex = ""
        
        for package in all_packages:
            if isinstance(package, str):
                latex += f"\\usepackage{{{package}}}\n"
            elif isinstance(package, dict):
                pkg_name = package['name']
                options = package.get('options', [])
                if options:
                    latex += f"\\usepackage[{','.join(options)}]{{{pkg_name}}}\n"
                else:
                    latex += f"\\usepackage{{{pkg_name}}}\n"
        
        latex += "\n"
        return latex
    
    def _add_preamble(self, content: Dict[str, Any]) -> str:
        """Add document preamble with settings"""
        latex = ""
        
        # Page geometry
        margins = content.get('margins', {
            'top': '2.5cm',
            'bottom': '2.5cm', 
            'left': '2.5cm',
            'right': '2.5cm'
        })
        
        margin_str = ', '.join([f"{k}={v}" for k, v in margins.items()])
        latex += f"\\geometry{{{margin_str}}}\n"
        
        # Headers and footers
        if content.get('fancy_headers', True):
            latex += "\\pagestyle{fancy}\n"
            latex += "\\fancyhf{}\n"  # Clear all headers/footers
            latex += "\\fancyhead[L]{\\leftmark}\n"
            latex += "\\fancyhead[R]{\\thepage}\n"
            latex += "\\renewcommand{\\headrulewidth}{0.4pt}\n"
        
        # Custom commands
        custom_commands = content.get('custom_commands', [])
        for cmd in custom_commands:
            latex += f"{cmd}\n"
        
        latex += "\n"
        return latex
    
    def _add_title_page(self, content: Dict[str, Any]) -> str:
        """Add title page"""
        latex = ""
        
        if 'title' in content:
            latex += f"\\title{{{self._escape_latex(content['title'])}}}\n"
        
        if 'author' in content:
            if isinstance(content['author'], list):
                authors = ' \\and '.join([self._escape_latex(a) for a in content['author']])
            else:
                authors = self._escape_latex(content['author'])
            latex += f"\\author{{{authors}}}\n"
        
        if 'date' in content:
            latex += f"\\date{{{self._escape_latex(content['date'])}}}\n"
        else:
            latex += "\\date{\\today}\n"
        
        latex += "\\maketitle\n"
        
        # Add abstract if present
        if 'abstract' in content:
            latex += "\\begin{abstract}\n"
            latex += f"{self._escape_latex(content['abstract'])}\n"
            latex += "\\end{abstract}\n"
        
        latex += "\\newpage\n\n"
        return latex
    
    def _add_content(self, content: Dict[str, Any]) -> str:
        """Add main document content"""
        latex = ""
        
        # Process main content
        if 'content' in content:
            latex += self._process_content(content['content'])
        
        # Process sections
        if 'sections' in content:
            for section in content['sections']:
                latex += self._add_section(section)
        
        return latex
    
    def _process_content(self, content: Union[str, List, Dict]) -> str:
        """Process various content types"""
        latex = ""
        
        if isinstance(content, str):
            # Simple text content
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    latex += f"{self._escape_latex(para.strip())}\n\n"
        
        elif isinstance(content, list):
            # List of content items
            for item in content:
                latex += self._process_content(item)
        
        elif isinstance(content, dict):
            # Structured content
            content_type = content.get('type', 'paragraph')
            
            if content_type == 'paragraph':
                latex += f"{self._escape_latex(content['text'])}\n\n"
            
            elif content_type == 'heading':
                level = content.get('level', 1)
                text = self._escape_latex(content['text'])
                
                if level == 1:
                    latex += f"\\section{{{text}}}\n"
                elif level == 2:
                    latex += f"\\subsection{{{text}}}\n"
                elif level == 3:
                    latex += f"\\subsubsection{{{text}}}\n"
                else:
                    latex += f"\\paragraph{{{text}}}\n"
            
            elif content_type == 'list':
                latex += self._add_list(content)
            
            elif content_type == 'table':
                latex += self._add_table(content)
            
            elif content_type == 'equation':
                latex += self._add_equation(content)
            
            elif content_type == 'figure':
                latex += self._add_figure(content)
            
            elif content_type == 'code':
                latex += self._add_code_block(content)
        
        return latex
    
    def _add_section(self, section: Dict[str, Any]) -> str:
        """Add document section"""
        latex = ""
        
        # Section title
        if 'title' in section:
            level = section.get('level', 1)
            title = self._escape_latex(section['title'])
            
            if level == 1:
                latex += f"\\section{{{title}}}\n"
            elif level == 2:
                latex += f"\\subsection{{{title}}}\n"
            elif level == 3:
                latex += f"\\subsubsection{{{title}}}\n"
            else:
                latex += f"\\paragraph{{{title}}}\n"
        
        # Section content
        if 'content' in section:
            latex += self._process_content(section['content'])
        
        latex += "\n"
        return latex
    
    def _add_list(self, list_data: Dict[str, Any]) -> str:
        """Add list (itemize or enumerate)"""
        list_type = list_data.get('list_type', 'itemize')
        items = list_data.get('items', [])
        
        latex = f"\\begin{{{list_type}}}\n"
        
        for item in items:
            if isinstance(item, str):
                latex += f"\\item {self._escape_latex(item)}\n"
            elif isinstance(item, dict):
                item_text = self._escape_latex(item.get('text', ''))
                latex += f"\\item {item_text}\n"
                
                # Nested list
                if 'subitems' in item:
                    nested_list = {
                        'list_type': list_type,
                        'items': item['subitems']
                    }
                    latex += self._add_list(nested_list)
        
        latex += f"\\end{{{list_type}}}\n\n"
        return latex
    
    def _add_table(self, table_data: Dict[str, Any]) -> str:
        """Add table"""
        latex = ""
        
        # Table caption
        caption = table_data.get('caption', '')
        label = table_data.get('label', '')
        
        latex += "\\begin{table}[h]\n"
        latex += "\\centering\n"
        
        if caption:
            latex += f"\\caption{{{self._escape_latex(caption)}}}\n"
        
        if label:
            latex += f"\\label{{{label}}}\n"
        
        # Table data
        headers = table_data.get('headers', [])
        data = table_data.get('data', [])
        
        if headers or data:
            # Determine column specification
            num_cols = len(headers) if headers else len(data[0]) if data else 0
            col_spec = 'l' * num_cols  # Left-aligned columns
            
            latex += f"\\begin{{tabular}}{{{col_spec}}}\n"
            latex += "\\toprule\n"
            
            # Headers
            if headers:
                header_row = ' & '.join([self._escape_latex(h) for h in headers])
                latex += f"{header_row} \\\\\n"
                latex += "\\midrule\n"
            
            # Data rows
            for row in data:
                row_data = ' & '.join([self._escape_latex(str(cell)) for cell in row])
                latex += f"{row_data} \\\\\n"
            
            latex += "\\bottomrule\n"
            latex += "\\end{tabular}\n"
        
        latex += "\\end{table}\n\n"
        return latex
    
    def _add_equation(self, equation_data: Dict[str, Any]) -> str:
        """Add mathematical equation"""
        equation = equation_data.get('equation', '')
        label = equation_data.get('label', '')
        numbered = equation_data.get('numbered', True)
        
        if numbered:
            latex = "\\begin{equation}\n"
            if label:
                latex += f"\\label{{{label}}}\n"
            latex += f"{equation}\n"
            latex += "\\end{equation}\n\n"
        else:
            latex = f"\\[{equation}\\]\n\n"
        
        return latex
    
    def _add_figure(self, figure_data: Dict[str, Any]) -> str:
        """Add figure"""
        latex = ""
        
        path = figure_data.get('path', '')
        caption = figure_data.get('caption', '')
        label = figure_data.get('label', '')
        width = figure_data.get('width', '0.8\\textwidth')
        
        latex += "\\begin{figure}[h]\n"
        latex += "\\centering\n"
        
        if path:
            latex += f"\\includegraphics[width={width}]{{{path}}}\n"
        
        if caption:
            latex += f"\\caption{{{self._escape_latex(caption)}}}\n"
        
        if label:
            latex += f"\\label{{{label}}}\n"
        
        latex += "\\end{figure}\n\n"
        return latex
    
    def _add_code_block(self, code_data: Dict[str, Any]) -> str:
        """Add code block"""
        code = code_data.get('code', '')
        language = code_data.get('language', 'text')
        caption = code_data.get('caption', '')
        
        latex = "\\begin{lstlisting}"
        
        # Add language and other options
        options = []
        if language:
            options.append(f"language={language}")
        if caption:
            options.append(f"caption={{{self._escape_latex(caption)}}}")
        
        if options:
            latex += f"[{','.join(options)}]"
        
        latex += "\n"
        latex += f"{code}\n"
        latex += "\\end{lstlisting}\n\n"
        
        return latex
    
    def _add_bibliography(self, bib_data: Dict[str, Any]) -> str:
        """Add bibliography"""
        latex = ""
        
        bib_style = bib_data.get('style', 'plain')
        bib_file = bib_data.get('file', '')
        
        if bib_file:
            latex += f"\\bibliographystyle{{{bib_style}}}\n"
            latex += f"\\bibliography{{{bib_file}}}\n"
        
        return latex
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters"""
        if not isinstance(text, str):
            text = str(text)
        
        # LaTeX special characters that need escaping
        replacements = {
            '\\': r'\textbackslash{}',
            '{': r'\{',
            '}': r'\}',
            '$': r'\$',
            '&': r'\&',
            '%': r'\%',
            '#': r'\#',
            '^': r'\textasciicircum{}',
            '_': r'\_',
            '~': r'\textasciitilde{}'
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text
    
    async def _compile_latex(self, tex_file: Path) -> Dict[str, Any]:
        """Compile LaTeX to PDF"""
        result = {
            'pdf_compiled': False,
            'pdf_file': None,
            'compilation_log': ''
        }
        
        try:
            # Check if pdflatex is available
            if not self._check_latex_installation():
                result['compilation_log'] = 'LaTeX not installed or not in PATH'
                return result
            
            # Compile with pdflatex
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy tex file to temp directory
                temp_tex = Path(temp_dir) / tex_file.name
                temp_tex.write_text(tex_file.read_text())
                
                # Run pdflatex
                cmd = [
                    'pdflatex',
                    '-interaction=nonstopmode',
                    '-output-directory', temp_dir,
                    str(temp_tex)
                ]
                
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=temp_dir
                )
                
                result['compilation_log'] = process.stdout + process.stderr
                
                # Check if PDF was created
                pdf_file = Path(temp_dir) / tex_file.with_suffix('.pdf').name
                
                if pdf_file.exists():
                    # Copy PDF to output location
                    output_pdf = tex_file.with_suffix('.pdf')
                    output_pdf.write_bytes(pdf_file.read_bytes())
                    
                    result['pdf_compiled'] = True
                    result['pdf_file'] = str(output_pdf)
                    
                    logger.info(f"Successfully compiled PDF: {output_pdf}")
                else:
                    logger.warning(f"PDF compilation failed for {tex_file}")
        
        except Exception as e:
            result['compilation_log'] += f"\nCompilation error: {str(e)}"
            logger.error(f"LaTeX compilation error: {e}")
        
        return result
    
    def _check_latex_installation(self) -> bool:
        """Check if LaTeX is installed"""
        try:
            subprocess.run(['pdflatex', '--version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    async def generate_academic_paper(self, paper_data: Dict[str, Any], 
                                    output_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Generate academic paper with proper formatting.
        
        Args:
            paper_data: Paper content and metadata
            output_path: Output file path
        
        Returns:
            Generation results
        """
        # Academic paper template
        content = {
            'document_class': 'article',
            'document_options': ['11pt', 'a4paper'],
            'packages': [
                {'name': 'natbib', 'options': ['numbers']},
                'amsmath',
                'amsfonts',
                'amssymb',
                'graphicx',
                'url'
            ],
            'title': paper_data.get('title', 'Research Paper'),
            'author': paper_data.get('authors', ['Author Name']),
            'abstract': paper_data.get('abstract', ''),
            'include_toc': False,
            'sections': paper_data.get('sections', []),
            'bibliography': paper_data.get('bibliography', {})
        }
        
        return await self.generate_document(content, output_path)
    
    async def generate_mathematical_document(self, math_data: Dict[str, Any],
                                           output_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Generate document with heavy mathematical content.
        
        Args:
            math_data: Mathematical content
            output_path: Output file path
        
        Returns:
            Generation results
        """
        content = {
            'document_class': 'article',
            'packages': [
                'amsmath',
                'amsfonts', 
                'amssymb',
                'amsthm',
                'mathtools',
                'physics'
            ],
            'custom_commands': [
                '\\newtheorem{theorem}{Theorem}',
                '\\newtheorem{lemma}{Lemma}',
                '\\newtheorem{definition}{Definition}'
            ],
            'title': math_data.get('title', 'Mathematical Document'),
            'author': math_data.get('author', 'Mathematician'),
            'content': math_data.get('content', [])
        }
        
        return await self.generate_document(content, output_path)


def create_latex_generator(template_dir: Optional[str] = None) -> LaTeXGenerator:
    """Factory function to create LaTeX generator"""
    return LaTeXGenerator(template_dir=template_dir)


__all__ = ['LaTeXGenerator', 'create_latex_generator']