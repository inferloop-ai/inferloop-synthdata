#!/usr/bin/env python3
"""
XML Exporter for document export functionality.

Provides XML export capabilities with schema validation and
customizable formatting options.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime, date
import base64

from ...core.logging import get_logger
from ...core.exceptions import ValidationError, ProcessingError


logger = get_logger(__name__)


class XMLExporter:
    """
    XML document exporter with advanced formatting capabilities.
    
    Features:
    - Multiple XML formats support
    - Schema validation
    - Custom namespaces
    - Pretty printing
    - CDATA support
    - Attribute handling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize XML exporter"""
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Default settings
        self.default_encoding = self.config.get('encoding', 'utf-8')
        self.pretty_print = self.config.get('pretty_print', True)
        self.indent_size = self.config.get('indent_size', 2)
        self.include_declaration = self.config.get('include_declaration', True)
        
        # Namespace management
        self.namespaces = self.config.get('namespaces', {})
        self.default_namespace = self.config.get('default_namespace')
        
        # Schema settings
        self.validate_schema = self.config.get('validate_schema', False)
        self.schema_path = self.config.get('schema_path')
        
        self.logger.info("XML exporter initialized")
    
    def export_document(
        self,
        data: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        root_element: str = 'document',
        attributes: Optional[Dict[str, str]] = None,
        namespaces: Optional[Dict[str, str]] = None,
        schema_location: Optional[str] = None
    ) -> Union[str, bytes]:
        """
        Export data as XML document.
        
        Args:
            data: Document data to export
            output_path: Optional output file path
            root_element: Root element name
            attributes: Root element attributes
            namespaces: XML namespaces
            schema_location: XSD schema location
            
        Returns:
            XML string or bytes if output_path not provided
        """
        try:
            # Merge namespaces
            all_namespaces = self.namespaces.copy()
            if namespaces:
                all_namespaces.update(namespaces)
            
            # Register namespaces
            for prefix, uri in all_namespaces.items():
                ET.register_namespace(prefix, uri)
            
            # Create root element
            if self.default_namespace:
                root = ET.Element(f"{{{self.default_namespace}}}{root_element}")
            else:
                root = ET.Element(root_element)
            
            # Add attributes
            if attributes:
                for key, value in attributes.items():
                    root.set(key, str(value))
            
            # Add schema location if provided
            if schema_location:
                root.set('{http://www.w3.org/2001/XMLSchema-instance}schemaLocation', schema_location)
            
            # Add metadata
            self._add_metadata(root, data.get('metadata', {}))
            
            # Process document content
            if 'content' in data:
                self._process_content(root, data['content'])
            
            # Process sections
            if 'sections' in data:
                self._process_sections(root, data['sections'])
            
            # Process tables
            if 'tables' in data:
                self._process_tables(root, data['tables'])
            
            # Process images
            if 'images' in data:
                self._process_images(root, data['images'])
            
            # Create XML tree
            tree = ET.ElementTree(root)
            
            # Validate against schema if required
            if self.validate_schema and self.schema_path:
                self._validate_xml(tree)
            
            # Format output
            xml_string = self._format_xml(tree)
            
            # Save to file if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding=self.default_encoding) as f:
                    f.write(xml_string)
                
                self.logger.info(f"XML document exported to {output_path}")
                return xml_string
            else:
                return xml_string
                
        except Exception as e:
            self.logger.error(f"XML export failed: {e}")
            raise ProcessingError(f"Failed to export XML: {e}")
    
    def export_batch(
        self,
        documents: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        filename_template: str = "document_{index}.xml",
        collection_file: Optional[str] = None
    ) -> List[Path]:
        """
        Export multiple documents as XML files.
        
        Args:
            documents: List of documents to export
            output_dir: Output directory
            filename_template: Filename template with {index} placeholder
            collection_file: Optional collection XML file
            
        Returns:
            List of exported file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        # Export individual documents
        for i, doc in enumerate(documents):
            filename = filename_template.format(index=i, **doc.get('metadata', {}))
            file_path = output_dir / filename
            
            self.export_document(doc, file_path)
            exported_files.append(file_path)
        
        # Create collection file if requested
        if collection_file:
            self._create_collection_file(
                documents,
                output_dir / collection_file,
                exported_files
            )
        
        self.logger.info(f"Exported {len(documents)} XML documents to {output_dir}")
        
        return exported_files
    
    def _add_metadata(self, parent: ET.Element, metadata: Dict[str, Any]):
        """Add metadata to XML element"""
        if not metadata:
            return
        
        meta_elem = ET.SubElement(parent, 'metadata')
        
        for key, value in metadata.items():
            elem = ET.SubElement(meta_elem, self._sanitize_tag_name(key))
            
            if isinstance(value, dict):
                self._add_dict_to_element(elem, value)
            elif isinstance(value, list):
                self._add_list_to_element(elem, value)
            else:
                elem.text = self._convert_value_to_string(value)
    
    def _process_content(self, parent: ET.Element, content: Union[str, Dict[str, Any]]):
        """Process document content"""
        content_elem = ET.SubElement(parent, 'content')
        
        if isinstance(content, str):
            # Simple text content
            content_elem.text = content
        elif isinstance(content, dict):
            # Structured content
            if 'type' in content:
                content_elem.set('type', content['type'])
            
            if 'text' in content:
                text_elem = ET.SubElement(content_elem, 'text')
                text_elem.text = content['text']
            
            if 'html' in content:
                # Add HTML as CDATA
                html_elem = ET.SubElement(content_elem, 'html')
                html_elem.text = content['html']  # Will be wrapped in CDATA later
            
            if 'structured' in content:
                self._add_dict_to_element(content_elem, content['structured'])
    
    def _process_sections(self, parent: ET.Element, sections: List[Dict[str, Any]]):
        """Process document sections"""
        sections_elem = ET.SubElement(parent, 'sections')
        
        for i, section in enumerate(sections):
            section_elem = ET.SubElement(sections_elem, 'section')
            section_elem.set('index', str(i))
            
            if 'id' in section:
                section_elem.set('id', section['id'])
            
            if 'title' in section:
                title_elem = ET.SubElement(section_elem, 'title')
                title_elem.text = section['title']
            
            if 'content' in section:
                self._process_content(section_elem, section['content'])
            
            if 'subsections' in section:
                self._process_sections(section_elem, section['subsections'])
    
    def _process_tables(self, parent: ET.Element, tables: List[Dict[str, Any]]):
        """Process tables"""
        tables_elem = ET.SubElement(parent, 'tables')
        
        for i, table in enumerate(tables):
            table_elem = ET.SubElement(tables_elem, 'table')
            table_elem.set('index', str(i))
            
            if 'id' in table:
                table_elem.set('id', table['id'])
            
            if 'caption' in table:
                caption_elem = ET.SubElement(table_elem, 'caption')
                caption_elem.text = table['caption']
            
            # Process headers
            if 'headers' in table:
                headers_elem = ET.SubElement(table_elem, 'headers')
                for header in table['headers']:
                    header_elem = ET.SubElement(headers_elem, 'header')
                    header_elem.text = str(header)
            
            # Process rows
            if 'rows' in table:
                rows_elem = ET.SubElement(table_elem, 'rows')
                for row in table['rows']:
                    row_elem = ET.SubElement(rows_elem, 'row')
                    for cell in row:
                        cell_elem = ET.SubElement(row_elem, 'cell')
                        cell_elem.text = str(cell)
    
    def _process_images(self, parent: ET.Element, images: List[Dict[str, Any]]):
        """Process images"""
        images_elem = ET.SubElement(parent, 'images')
        
        for i, image in enumerate(images):
            image_elem = ET.SubElement(images_elem, 'image')
            image_elem.set('index', str(i))
            
            if 'id' in image:
                image_elem.set('id', image['id'])
            
            if 'path' in image:
                image_elem.set('src', image['path'])
            
            if 'caption' in image:
                caption_elem = ET.SubElement(image_elem, 'caption')
                caption_elem.text = image['caption']
            
            if 'data' in image and self.config.get('embed_images', False):
                # Embed image data as base64
                data_elem = ET.SubElement(image_elem, 'data')
                data_elem.set('encoding', 'base64')
                
                if isinstance(image['data'], bytes):
                    data_elem.text = base64.b64encode(image['data']).decode('ascii')
                else:
                    data_elem.text = image['data']
    
    def _add_dict_to_element(self, parent: ET.Element, data: Dict[str, Any]):
        """Recursively add dictionary to XML element"""
        for key, value in data.items():
            key = self._sanitize_tag_name(key)
            
            if isinstance(value, dict):
                elem = ET.SubElement(parent, key)
                self._add_dict_to_element(elem, value)
            elif isinstance(value, list):
                self._add_list_to_element(parent, value, key)
            else:
                elem = ET.SubElement(parent, key)
                elem.text = self._convert_value_to_string(value)
    
    def _add_list_to_element(self, parent: ET.Element, items: List[Any], tag_name: str = 'item'):
        """Add list to XML element"""
        for item in items:
            if isinstance(item, dict):
                elem = ET.SubElement(parent, tag_name)
                self._add_dict_to_element(elem, item)
            else:
                elem = ET.SubElement(parent, tag_name)
                elem.text = self._convert_value_to_string(item)
    
    def _sanitize_tag_name(self, name: str) -> str:
        """Sanitize string to be valid XML tag name"""
        # Replace invalid characters
        name = re.sub(r'[^a-zA-Z0-9_\-.]', '_', name)
        
        # Ensure starts with letter or underscore
        if name and not name[0].isalpha() and name[0] != '_':
            name = '_' + name
        
        return name or 'element'
    
    def _convert_value_to_string(self, value: Any) -> str:
        """Convert value to string for XML"""
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        elif isinstance(value, bool):
            return 'true' if value else 'false'
        elif value is None:
            return ''
        else:
            return str(value)
    
    def _format_xml(self, tree: ET.ElementTree) -> str:
        """Format XML with pretty printing"""
        # Convert to string
        xml_string = ET.tostring(tree.getroot(), encoding='unicode')
        
        if self.pretty_print:
            # Parse with minidom for pretty printing
            dom = minidom.parseString(xml_string)
            
            # Format with indentation
            pretty_xml = dom.toprettyxml(indent=' ' * self.indent_size)
            
            # Remove extra blank lines
            lines = [line for line in pretty_xml.split('\n') if line.strip()]
            
            # Add declaration if needed
            if self.include_declaration:
                if not lines[0].startswith('<?xml'):
                    lines.insert(0, f'<?xml version="1.0" encoding="{self.default_encoding}"?>')
            else:
                # Remove declaration if present
                if lines and lines[0].startswith('<?xml'):
                    lines.pop(0)
            
            return '\n'.join(lines)
        else:
            # Add declaration if needed
            if self.include_declaration and not xml_string.startswith('<?xml'):
                xml_string = f'<?xml version="1.0" encoding="{self.default_encoding}"?>\n' + xml_string
            
            return xml_string
    
    def _validate_xml(self, tree: ET.ElementTree):
        """Validate XML against schema"""
        if not self.schema_path:
            return
        
        try:
            from lxml import etree
            
            # Load schema
            with open(self.schema_path, 'r') as f:
                schema_doc = etree.parse(f)
                schema = etree.XMLSchema(schema_doc)
            
            # Convert ElementTree to lxml
            xml_string = ET.tostring(tree.getroot())
            doc = etree.fromstring(xml_string)
            
            # Validate
            if not schema.validate(doc):
                errors = schema.error_log
                raise ValidationError(f"XML validation failed: {errors}")
                
        except ImportError:
            self.logger.warning("lxml not available, skipping schema validation")
        except Exception as e:
            raise ValidationError(f"XML validation error: {e}")
    
    def _create_collection_file(
        self,
        documents: List[Dict[str, Any]],
        output_path: Path,
        file_paths: List[Path]
    ):
        """Create collection XML file"""
        root = ET.Element('documentCollection')
        root.set('count', str(len(documents)))
        root.set('created', datetime.now().isoformat())
        
        for i, (doc, file_path) in enumerate(zip(documents, file_paths)):
            doc_ref = ET.SubElement(root, 'documentRef')
            doc_ref.set('index', str(i))
            doc_ref.set('file', file_path.name)
            
            if 'metadata' in doc:
                if 'id' in doc['metadata']:
                    doc_ref.set('id', doc['metadata']['id'])
                if 'title' in doc['metadata']:
                    title_elem = ET.SubElement(doc_ref, 'title')
                    title_elem.text = doc['metadata']['title']
        
        tree = ET.ElementTree(root)
        xml_string = self._format_xml(tree)
        
        with open(output_path, 'w', encoding=self.default_encoding) as f:
            f.write(xml_string)


# Factory function
def create_xml_exporter(config: Optional[Dict[str, Any]] = None) -> XMLExporter:
    """Create and return an XML exporter instance"""
    return XMLExporter(config)