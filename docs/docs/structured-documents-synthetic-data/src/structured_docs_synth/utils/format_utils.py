#!/usr/bin/env python3
"""
Format detection, conversion, and validation utilities.

Provides functions for detecting file formats, converting between formats,
and parsing/formatting various data formats like JSON, XML, CSV, YAML.
"""

import csv
import json
import mimetypes
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from PIL import Image

from ..core import get_logger

logger = get_logger(__name__)


def detect_format(file_path: Union[str, Path]) -> Dict[str, str]:
    """
    Detect file format based on extension and content.
    
    Args:
        file_path: Path to file
    
    Returns:
        Dictionary with format information
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get MIME type and encoding
        mime_type, encoding = mimetypes.guess_type(str(path))
        
        # Get file extension
        extension = path.suffix.lower()
        
        # Analyze content for ambiguous cases
        content_info = _analyze_content(path)
        
        return {
            'extension': extension,
            'mime_type': mime_type or 'unknown',
            'encoding': encoding,
            'detected_format': content_info.get('format', 'unknown'),
            'confidence': content_info.get('confidence', 0.0),
            'size': path.stat().st_size,
            'is_text': content_info.get('is_text', False),
            'is_binary': content_info.get('is_binary', True)
        }
        
    except Exception as e:
        logger.error(f"Failed to detect format for {file_path}: {e}")
        raise


def _analyze_content(path: Path) -> Dict[str, Any]:
    """Analyze file content to determine format"""
    try:
        # Try to read as text first
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read(1024)  # Read first 1KB
            is_text = True
        except UnicodeDecodeError:
            is_text = False
            content = None
        
        if not is_text:
            # Check if it's an image
            try:
                with Image.open(path) as img:
                    return {
                        'format': f'image/{img.format.lower()}',
                        'confidence': 1.0,
                        'is_text': False,
                        'is_binary': True,
                        'image_format': img.format,
                        'image_size': img.size,
                        'image_mode': img.mode
                    }
            except Exception:
                pass
            
            return {
                'format': 'binary',
                'confidence': 0.8,
                'is_text': False,
                'is_binary': True
            }
        
        # Analyze text content
        content = content.strip()
        
        # Check for JSON
        if content.startswith(('{', '[')):
            try:
                json.loads(content)
                return {
                    'format': 'json',
                    'confidence': 0.9,
                    'is_text': True,
                    'is_binary': False
                }
            except json.JSONDecodeError:
                pass
        
        # Check for XML
        if content.startswith('<?xml') or content.startswith('<'):
            try:
                ET.fromstring(content)
                return {
                    'format': 'xml',
                    'confidence': 0.9,
                    'is_text': True,
                    'is_binary': False
                }
            except ET.ParseError:
                pass
        
        # Check for YAML
        if any(line.strip() and ':' in line for line in content.split('\n')[:5]):
            try:
                yaml.safe_load(content)
                return {
                    'format': 'yaml',
                    'confidence': 0.7,
                    'is_text': True,
                    'is_binary': False
                }
            except yaml.YAMLError:
                pass
        
        # Check for CSV
        if ',' in content or ';' in content:
            try:
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(content)
                if sniffer.has_header(content):
                    return {
                        'format': 'csv',
                        'confidence': 0.8,
                        'is_text': True,
                        'is_binary': False,
                        'csv_dialect': dialect
                    }
            except Exception:
                pass
        
        # Default to plain text
        return {
            'format': 'text/plain',
            'confidence': 0.5,
            'is_text': True,
            'is_binary': False
        }
        
    except Exception as e:
        logger.warning(f"Content analysis failed: {e}")
        return {
            'format': 'unknown',
            'confidence': 0.0,
            'is_text': False,
            'is_binary': True
        }


def validate_format(data: Union[str, bytes], format_type: str) -> Dict[str, Any]:
    """
    Validate data against specified format.
    
    Args:
        data: Data to validate
        format_type: Expected format ('json', 'xml', 'csv', 'yaml')
    
    Returns:
        Dictionary with validation results
    """
    try:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        
        result = {
            'valid': False,
            'format': format_type,
            'errors': [],
            'warnings': []
        }
        
        if format_type == 'json':
            try:
                parsed = json.loads(data)
                result['valid'] = True
                result['parsed_data'] = parsed
            except json.JSONDecodeError as e:
                result['errors'].append(f"JSON parsing error: {e}")
        
        elif format_type == 'xml':
            try:
                root = ET.fromstring(data)
                result['valid'] = True
                result['root_tag'] = root.tag
                result['elements_count'] = len(list(root.iter()))
            except ET.ParseError as e:
                result['errors'].append(f"XML parsing error: {e}")
        
        elif format_type == 'yaml':
            try:
                parsed = yaml.safe_load(data)
                result['valid'] = True
                result['parsed_data'] = parsed
            except yaml.YAMLError as e:
                result['errors'].append(f"YAML parsing error: {e}")
        
        elif format_type == 'csv':
            try:
                # Try to parse as CSV
                csv_reader = csv.reader(StringIO(data))
                rows = list(csv_reader)
                result['valid'] = True
                result['rows_count'] = len(rows)
                result['columns_count'] = len(rows[0]) if rows else 0
            except Exception as e:
                result['errors'].append(f"CSV parsing error: {e}")
        
        else:
            result['errors'].append(f"Unsupported format: {format_type}")
        
        return result
        
    except Exception as e:
        logger.error(f"Format validation failed: {e}")
        return {
            'valid': False,
            'format': format_type,
            'errors': [str(e)],
            'warnings': []
        }


def parse_json(data: Union[str, bytes]) -> Any:
    """
    Parse JSON data with error handling.
    
    Args:
        data: JSON string or bytes
    
    Returns:
        Parsed JSON object
    """
    try:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return json.loads(data)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        raise


def format_json(data: Any, indent: int = 2, ensure_ascii: bool = False) -> str:
    """
    Format data as JSON string.
    
    Args:
        data: Data to format
        indent: Indentation spaces
        ensure_ascii: Whether to escape non-ASCII characters
    
    Returns:
        Formatted JSON string
    """
    try:
        return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)
    except TypeError as e:
        logger.error(f"JSON formatting failed: {e}")
        raise


def parse_xml(data: Union[str, bytes]) -> ET.Element:
    """
    Parse XML data with error handling.
    
    Args:
        data: XML string or bytes
    
    Returns:
        XML root element
    """
    try:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return ET.fromstring(data)
    except ET.ParseError as e:
        logger.error(f"XML parsing failed: {e}")
        raise


def format_xml(element: ET.Element, encoding: str = 'unicode') -> str:
    """
    Format XML element as string.
    
    Args:
        element: XML element
        encoding: Output encoding
    
    Returns:
        Formatted XML string
    """
    try:
        # Pretty print XML
        _indent_xml(element)
        return ET.tostring(element, encoding=encoding, method='xml')
    except Exception as e:
        logger.error(f"XML formatting failed: {e}")
        raise


def _indent_xml(elem, level=0):
    """Add indentation to XML for pretty printing"""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            _indent_xml(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def parse_csv(data: Union[str, bytes], delimiter: str = ',', 
             has_header: bool = True) -> Dict[str, Any]:
    """
    Parse CSV data with error handling.
    
    Args:
        data: CSV string or bytes
        delimiter: Field delimiter
        has_header: Whether first row is header
    
    Returns:
        Dictionary with parsed CSV data
    """
    try:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        
        csv_reader = csv.reader(StringIO(data), delimiter=delimiter)
        rows = list(csv_reader)
        
        if not rows:
            return {'headers': [], 'data': []}
        
        if has_header:
            headers = rows[0]
            data_rows = rows[1:]
        else:
            headers = [f'column_{i}' for i in range(len(rows[0]))]
            data_rows = rows
        
        return {
            'headers': headers,
            'data': data_rows,
            'rows_count': len(data_rows),
            'columns_count': len(headers)
        }
        
    except Exception as e:
        logger.error(f"CSV parsing failed: {e}")
        raise


def format_csv(data: List[List[str]], headers: Optional[List[str]] = None,
              delimiter: str = ',') -> str:
    """
    Format data as CSV string.
    
    Args:
        data: List of rows (each row is list of values)
        headers: Optional column headers
        delimiter: Field delimiter
    
    Returns:
        Formatted CSV string
    """
    try:
        output = StringIO()
        writer = csv.writer(output, delimiter=delimiter)
        
        if headers:
            writer.writerow(headers)
        
        writer.writerows(data)
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"CSV formatting failed: {e}")
        raise


def parse_yaml(data: Union[str, bytes]) -> Any:
    """
    Parse YAML data with error handling.
    
    Args:
        data: YAML string or bytes
    
    Returns:
        Parsed YAML object
    """
    try:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return yaml.safe_load(data)
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing failed: {e}")
        raise


def format_yaml(data: Any, default_flow_style: bool = False) -> str:
    """
    Format data as YAML string.
    
    Args:
        data: Data to format
        default_flow_style: Whether to use flow style
    
    Returns:
        Formatted YAML string
    """
    try:
        return yaml.dump(data, default_flow_style=default_flow_style)
    except yaml.YAMLError as e:
        logger.error(f"YAML formatting failed: {e}")
        raise


def convert_format(data: Union[str, bytes], from_format: str, 
                  to_format: str) -> str:
    """
    Convert data from one format to another.
    
    Args:
        data: Input data
        from_format: Source format ('json', 'xml', 'csv', 'yaml')
        to_format: Target format ('json', 'xml', 'csv', 'yaml')
    
    Returns:
        Converted data string
    """
    try:
        # Parse source format
        if from_format == 'json':
            parsed = parse_json(data)
        elif from_format == 'yaml':
            parsed = parse_yaml(data)
        elif from_format == 'csv':
            csv_data = parse_csv(data)
            # Convert CSV to list of dictionaries
            parsed = [
                dict(zip(csv_data['headers'], row))
                for row in csv_data['data']
            ]
        else:
            raise ValueError(f"Unsupported source format: {from_format}")
        
        # Format to target format
        if to_format == 'json':
            return format_json(parsed)
        elif to_format == 'yaml':
            return format_yaml(parsed)
        elif to_format == 'csv':
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                headers = list(parsed[0].keys())
                rows = [[str(row.get(h, '')) for h in headers] for row in parsed]
                return format_csv(rows, headers)
            else:
                raise ValueError("Data cannot be converted to CSV format")
        else:
            raise ValueError(f"Unsupported target format: {to_format}")
        
    except Exception as e:
        logger.error(f"Format conversion failed from {from_format} to {to_format}: {e}")
        raise


def normalize_format_name(format_name: str) -> str:
    """
    Normalize format name to standard form.
    
    Args:
        format_name: Format name to normalize
    
    Returns:
        Normalized format name
    """
    format_mapping = {
        'application/json': 'json',
        'text/json': 'json',
        'application/xml': 'xml',
        'text/xml': 'xml',
        'text/csv': 'csv',
        'application/x-yaml': 'yaml',
        'text/yaml': 'yaml',
        'text/x-yaml': 'yaml',
        'application/yaml': 'yaml'
    }
    
    normalized = format_name.lower().strip()
    return format_mapping.get(normalized, normalized)


def get_format_extensions() -> Dict[str, List[str]]:
    """
    Get mapping of formats to their common file extensions.
    
    Returns:
        Dictionary mapping format names to extension lists
    """
    return {
        'json': ['.json', '.jsonl'],
        'xml': ['.xml', '.xsd', '.xsl'],
        'csv': ['.csv', '.tsv'],
        'yaml': ['.yaml', '.yml'],
        'pdf': ['.pdf'],
        'docx': ['.docx', '.doc'],
        'image': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'],
        'text': ['.txt', '.log', '.md'],
        'archive': ['.zip', '.tar', '.gz', '.bz2', '.xz']
    }


def is_supported_format(format_name: str) -> bool:
    """
    Check if format is supported by the system.
    
    Args:
        format_name: Format name to check
    
    Returns:
        True if format is supported
    """
    supported_formats = {
        'json', 'xml', 'csv', 'yaml', 'pdf', 'docx', 
        'png', 'jpg', 'jpeg', 'gif', 'text', 'html'
    }
    
    normalized = normalize_format_name(format_name)
    return normalized in supported_formats


__all__ = [
    'detect_format',
    'validate_format',
    'parse_json',
    'format_json',
    'parse_xml',
    'format_xml',
    'parse_csv',
    'format_csv',
    'parse_yaml',
    'format_yaml',
    'convert_format',
    'normalize_format_name',
    'get_format_extensions',
    'is_supported_format'
]