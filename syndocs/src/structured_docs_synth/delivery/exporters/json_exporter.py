#!/usr/bin/env python3
"""
JSON Exporter for document export functionality.

Provides JSON export capabilities with schema validation,
compression, and streaming support.
"""

import json
import gzip
import bz2
import lzma
from typing import Dict, List, Optional, Any, Union, Iterator, TextIO
from pathlib import Path
from datetime import datetime, date
from decimal import Decimal
import jsonschema
from jsonschema import validate, ValidationError as JsonSchemaValidationError

from ...core.logging import get_logger
from ...core.exceptions import ValidationError, ProcessingError
from ...utils.json_utils import (
    JSONEncoder,
    save_json,
    write_json_lines,
    json_transform,
    json_filter,
    json_sort
)


logger = get_logger(__name__)


class JSONExporter:
    """
    JSON document exporter with advanced formatting capabilities.
    
    Features:
    - Multiple JSON formats (standard, JSON Lines, streaming)
    - Schema validation
    - Compression support
    - Custom transformations
    - Filtering and sorting
    - Batch export
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize JSON exporter"""
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # JSON settings
        self.indent = self.config.get('indent', 2)
        self.sort_keys = self.config.get('sort_keys', False)
        self.ensure_ascii = self.config.get('ensure_ascii', False)
        self.encoding = self.config.get('encoding', 'utf-8')
        
        # Export settings
        self.compression = self.config.get('compression', None)  # None, 'gzip', 'bz2', 'xz'
        self.compression_level = self.config.get('compression_level', 9)
        self.validate_schema = self.config.get('validate_schema', False)
        self.schema = self.config.get('schema')
        
        # Transformation settings
        self.transformations = self.config.get('transformations', [])
        self.field_mapping = self.config.get('field_mapping', {})
        self.excluded_fields = set(self.config.get('excluded_fields', []))
        self.filters = self.config.get('filters', [])
        
        # Custom encoder
        self.encoder_class = self.config.get('encoder_class', JSONEncoder)
        
        self.logger.info("JSON exporter initialized")
    
    def export_document(
        self,
        data: Any,
        output_path: Optional[Union[str, Path]] = None,
        minify: bool = False,
        streaming: bool = False
    ) -> Union[str, bytes, None]:
        """
        Export data as JSON.
        
        Args:
            data: Document data to export
            output_path: Optional output file path
            minify: Minify JSON output
            streaming: Use streaming for large data
            
        Returns:
            JSON string/bytes if no output_path, None otherwise
        """
        try:
            # Apply transformations
            processed_data = self._apply_transformations(data)
            
            # Apply filters
            if self.filters:
                processed_data = self._apply_filters(processed_data)
            
            # Validate against schema
            if self.validate_schema and self.schema:
                self._validate_data(processed_data)
            
            # Prepare JSON options
            json_options = {
                'cls': self.encoder_class,
                'ensure_ascii': self.ensure_ascii,
                'sort_keys': self.sort_keys
            }
            
            if not minify:
                json_options['indent'] = self.indent
            
            # Export based on mode
            if output_path:
                output_path = Path(output_path)
                
                if streaming and isinstance(processed_data, (list, Iterator)):
                    self._export_streaming(processed_data, output_path, json_options)
                else:
                    self._export_to_file(processed_data, output_path, json_options)
                
                self.logger.info(f"JSON exported to {output_path}")
                return None
            else:
                # Return JSON string
                json_str = json.dumps(processed_data, **json_options)
                
                if self.compression:
                    return self._compress_data(json_str.encode(self.encoding))
                else:
                    return json_str
                    
        except Exception as e:
            self.logger.error(f"JSON export failed: {e}")
            raise ProcessingError(f"Failed to export JSON: {e}")
    
    def export_json_lines(
        self,
        documents: Union[List[Any], Iterator[Any]],
        output_path: Union[str, Path],
        append: bool = False
    ) -> int:
        """
        Export documents as JSON Lines format.
        
        Args:
            documents: Documents to export
            output_path: Output file path
            append: Append to existing file
            
        Returns:
            Number of lines written
        """
        try:
            output_path = Path(output_path)
            
            # Apply compression if needed
            if self.compression:
                output_path = Path(str(output_path) + self._get_compression_extension())
            
            # Process documents
            processed_docs = (
                self._apply_transformations(doc) for doc in documents
            )
            
            if self.filters:
                processed_docs = (
                    doc for doc in processed_docs
                    if self._passes_filters(doc)
                )
            
            # Write JSON Lines
            if self.compression:
                return self._write_compressed_json_lines(
                    processed_docs,
                    output_path,
                    append
                )
            else:
                mode = 'a' if append else 'w'
                count = 0
                
                with open(output_path, mode, encoding=self.encoding) as f:
                    for doc in processed_docs:
                        json_line = json.dumps(
                            doc,
                            cls=self.encoder_class,
                            ensure_ascii=self.ensure_ascii,
                            sort_keys=self.sort_keys
                        )
                        f.write(json_line + '\n')
                        count += 1
                
                self.logger.info(f"Exported {count} documents to {output_path}")
                return count
                
        except Exception as e:
            self.logger.error(f"JSON Lines export failed: {e}")
            raise ProcessingError(f"Failed to export JSON Lines: {e}")
    
    def export_batch(
        self,
        documents: List[Any],
        output_dir: Union[str, Path],
        filename_template: str = "document_{index}.json",
        single_file: bool = False,
        collection_name: str = "collection.json"
    ) -> Union[Path, List[Path]]:
        """
        Export multiple documents.
        
        Args:
            documents: List of documents
            output_dir: Output directory
            filename_template: Template for individual files
            single_file: Export all to single file
            collection_name: Name for single file
            
        Returns:
            Path or list of paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if single_file:
            # Export all documents to single file
            collection_path = output_dir / collection_name
            
            collection_data = {
                'metadata': {
                    'count': len(documents),
                    'created': datetime.now().isoformat(),
                    'exporter': 'JSONExporter'
                },
                'documents': documents
            }
            
            self.export_document(collection_data, collection_path)
            return collection_path
        else:
            # Export to separate files
            exported_files = []
            
            for i, doc in enumerate(documents):
                filename = filename_template.format(
                    index=i,
                    **doc.get('metadata', {})
                )
                
                if self.compression:
                    filename += self._get_compression_extension()
                
                file_path = output_dir / filename
                self.export_document(doc, file_path)
                exported_files.append(file_path)
            
            self.logger.info(f"Exported {len(documents)} JSON files to {output_dir}")
            return exported_files
    
    def export_schema(
        self,
        data: Any,
        output_path: Optional[Union[str, Path]] = None,
        title: str = "Document Schema",
        description: str = "Auto-generated JSON schema"
    ) -> Dict[str, Any]:
        """
        Generate and export JSON schema from data.
        
        Args:
            data: Sample data to generate schema from
            output_path: Optional output path
            title: Schema title
            description: Schema description
            
        Returns:
            Generated schema
        """
        try:
            schema = self._generate_schema(data, title, description)
            
            if output_path:
                save_json(schema, output_path, indent=2, sort_keys=True)
                self.logger.info(f"Schema exported to {output_path}")
            
            return schema
            
        except Exception as e:
            self.logger.error(f"Schema export failed: {e}")
            raise ProcessingError(f"Failed to export schema: {e}")
    
    def _apply_transformations(self, data: Any) -> Any:
        """Apply configured transformations to data"""
        result = data
        
        # Apply field mapping
        if self.field_mapping and isinstance(result, dict):
            result = json_transform(result, self.field_mapping)
        
        # Remove excluded fields
        if self.excluded_fields and isinstance(result, dict):
            result = {
                k: v for k, v in result.items()
                if k not in self.excluded_fields
            }
        
        # Apply custom transformations
        for transform in self.transformations:
            if callable(transform):
                result = transform(result)
            elif isinstance(transform, dict):
                # Configuration-based transformation
                transform_type = transform.get('type')
                
                if transform_type == 'rename':
                    mapping = transform.get('mapping', {})
                    result = json_transform(result, mapping)
                elif transform_type == 'filter':
                    condition = transform.get('condition')
                    if condition:
                        result = json_filter(result, condition)
                elif transform_type == 'sort':
                    key = transform.get('key')
                    reverse = transform.get('reverse', False)
                    result = json_sort(result, key, reverse)
        
        return result
    
    def _apply_filters(self, data: Any) -> Any:
        """Apply filters to data"""
        if isinstance(data, list):
            return [item for item in data if self._passes_filters(item)]
        else:
            return data if self._passes_filters(data) else None
    
    def _passes_filters(self, item: Any) -> bool:
        """Check if item passes all filters"""
        for filter_func in self.filters:
            if callable(filter_func):
                if not filter_func(item):
                    return False
            elif isinstance(filter_func, dict):
                # Configuration-based filter
                field = filter_func.get('field')
                operator = filter_func.get('operator', 'eq')
                value = filter_func.get('value')
                
                if field and isinstance(item, dict):
                    item_value = item.get(field)
                    
                    if operator == 'eq':
                        if item_value != value:
                            return False
                    elif operator == 'ne':
                        if item_value == value:
                            return False
                    elif operator == 'gt':
                        if not (item_value > value):
                            return False
                    elif operator == 'lt':
                        if not (item_value < value):
                            return False
                    elif operator == 'in':
                        if item_value not in value:
                            return False
                    elif operator == 'contains':
                        if value not in str(item_value):
                            return False
        
        return True
    
    def _validate_data(self, data: Any):
        """Validate data against schema"""
        try:
            validate(instance=data, schema=self.schema)
        except JsonSchemaValidationError as e:
            raise ValidationError(f"Schema validation failed: {e.message}")
    
    def _export_to_file(
        self,
        data: Any,
        output_path: Path,
        json_options: Dict[str, Any]
    ):
        """Export data to file with optional compression"""
        # Add compression extension if needed
        if self.compression:
            output_path = Path(str(output_path) + self._get_compression_extension())
        
        # Convert to JSON
        json_str = json.dumps(data, **json_options)
        json_bytes = json_str.encode(self.encoding)
        
        # Write to file
        if self.compression:
            compressed_data = self._compress_data(json_bytes)
            with open(output_path, 'wb') as f:
                f.write(compressed_data)
        else:
            with open(output_path, 'w', encoding=self.encoding) as f:
                f.write(json_str)
    
    def _export_streaming(
        self,
        data: Union[List[Any], Iterator[Any]],
        output_path: Path,
        json_options: Dict[str, Any]
    ):
        """Export data using streaming for memory efficiency"""
        if self.compression:
            output_path = Path(str(output_path) + self._get_compression_extension())
            
            with self._open_compressed_file(output_path, 'wb') as f:
                f.write(b'[\n')
                
                first = True
                for item in data:
                    if not first:
                        f.write(b',\n')
                    first = False
                    
                    json_str = json.dumps(item, **json_options)
                    f.write(json_str.encode(self.encoding))
                
                f.write(b'\n]')
        else:
            with open(output_path, 'w', encoding=self.encoding) as f:
                f.write('[\n')
                
                first = True
                for item in data:
                    if not first:
                        f.write(',\n')
                    first = False
                    
                    json.dump(item, f, **json_options)
                
                f.write('\n]')
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using configured compression"""
        if self.compression == 'gzip':
            return gzip.compress(data, compresslevel=self.compression_level)
        elif self.compression == 'bz2':
            return bz2.compress(data, compresslevel=self.compression_level)
        elif self.compression == 'xz' or self.compression == 'lzma':
            return lzma.compress(data, preset=self.compression_level)
        else:
            return data
    
    def _get_compression_extension(self) -> str:
        """Get file extension for compression type"""
        extensions = {
            'gzip': '.gz',
            'bz2': '.bz2',
            'xz': '.xz',
            'lzma': '.xz'
        }
        return extensions.get(self.compression, '')
    
    def _open_compressed_file(self, path: Path, mode: str):
        """Open compressed file for writing"""
        if self.compression == 'gzip':
            return gzip.open(path, mode, compresslevel=self.compression_level)
        elif self.compression == 'bz2':
            return bz2.open(path, mode, compresslevel=self.compression_level)
        elif self.compression == 'xz' or self.compression == 'lzma':
            return lzma.open(path, mode, preset=self.compression_level)
        else:
            return open(path, mode)
    
    def _write_compressed_json_lines(
        self,
        documents: Iterator[Any],
        output_path: Path,
        append: bool
    ) -> int:
        """Write JSON Lines with compression"""
        mode = 'ab' if append else 'wb'
        count = 0
        
        with self._open_compressed_file(output_path, mode) as f:
            for doc in documents:
                json_line = json.dumps(
                    doc,
                    cls=self.encoder_class,
                    ensure_ascii=self.ensure_ascii,
                    sort_keys=self.sort_keys
                )
                f.write((json_line + '\n').encode(self.encoding))
                count += 1
        
        return count
    
    def _generate_schema(
        self,
        data: Any,
        title: str,
        description: str
    ) -> Dict[str, Any]:
        """Generate JSON schema from sample data"""
        def infer_type(value: Any) -> Dict[str, Any]:
            """Infer JSON schema type from value"""
            if value is None:
                return {"type": "null"}
            elif isinstance(value, bool):
                return {"type": "boolean"}
            elif isinstance(value, int):
                return {"type": "integer"}
            elif isinstance(value, float):
                return {"type": "number"}
            elif isinstance(value, str):
                return {"type": "string"}
            elif isinstance(value, list):
                if value:
                    # Infer array item type from first element
                    item_schema = infer_type(value[0])
                    return {
                        "type": "array",
                        "items": item_schema
                    }
                else:
                    return {"type": "array"}
            elif isinstance(value, dict):
                properties = {}
                required = []
                
                for key, val in value.items():
                    properties[key] = infer_type(val)
                    required.append(key)
                
                return {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            else:
                return {"type": "string"}  # Default to string
        
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": title,
            "description": description
        }
        
        schema.update(infer_type(data))
        
        return schema


# Factory function
def create_json_exporter(config: Optional[Dict[str, Any]] = None) -> JSONExporter:
    """Create and return a JSON exporter instance"""
    return JSONExporter(config)