#!/usr/bin/env python3
"""
CSV Exporter for document export functionality.

Provides CSV export capabilities with flexible formatting options
and support for complex data structures.
"""

import csv
import io
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from datetime import datetime, date
import json

from ...core.logging import get_logger
from ...core.exceptions import ValidationError, ProcessingError


logger = get_logger(__name__)


class CSVExporter:
    """
    CSV document exporter with advanced formatting capabilities.
    
    Features:
    - Multiple CSV formats
    - Custom delimiters
    - Header customization
    - Data flattening
    - Type conversion
    - Batch export
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize CSV exporter"""
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # CSV settings
        self.delimiter = self.config.get('delimiter', ',')
        self.quotechar = self.config.get('quotechar', '"')
        self.quoting = self.config.get('quoting', csv.QUOTE_MINIMAL)
        self.lineterminator = self.config.get('lineterminator', '\n')
        self.encoding = self.config.get('encoding', 'utf-8')
        
        # Export settings
        self.include_headers = self.config.get('include_headers', True)
        self.flatten_nested = self.config.get('flatten_nested', True)
        self.null_value = self.config.get('null_value', '')
        self.date_format = self.config.get('date_format', '%Y-%m-%d')
        self.datetime_format = self.config.get('datetime_format', '%Y-%m-%d %H:%M:%S')
        
        # Field configuration
        self.field_mapping = self.config.get('field_mapping', {})
        self.excluded_fields = set(self.config.get('excluded_fields', []))
        self.field_order = self.config.get('field_order', [])
        
        self.logger.info("CSV exporter initialized")
    
    def export_document(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        output_path: Optional[Union[str, Path]] = None,
        fields: Optional[List[str]] = None,
        headers: Optional[List[str]] = None
    ) -> str:
        """
        Export data as CSV.
        
        Args:
            data: Document data (single dict or list of dicts)
            output_path: Optional output file path
            fields: Field names to include
            headers: Custom header names
            
        Returns:
            CSV string
        """
        try:
            # Normalize data to list
            if isinstance(data, dict):
                rows = [data]
            else:
                rows = data
            
            if not rows:
                return ""
            
            # Process rows
            processed_rows = []
            for row in rows:
                processed_row = self._process_row(row)
                if processed_row:
                    processed_rows.append(processed_row)
            
            if not processed_rows:
                return ""
            
            # Determine fields
            if not fields:
                fields = self._determine_fields(processed_rows)
            
            # Apply field order
            if self.field_order:
                ordered_fields = []
                for field in self.field_order:
                    if field in fields:
                        ordered_fields.append(field)
                # Add remaining fields
                for field in fields:
                    if field not in ordered_fields:
                        ordered_fields.append(field)
                fields = ordered_fields
            
            # Determine headers
            if not headers:
                headers = self._get_headers(fields)
            
            # Create CSV
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=fields,
                delimiter=self.delimiter,
                quotechar=self.quotechar,
                quoting=self.quoting,
                lineterminator=self.lineterminator,
                extrasaction='ignore'
            )
            
            # Write headers
            if self.include_headers:
                writer.writerow(dict(zip(fields, headers)))
            
            # Write data
            for row in processed_rows:
                # Ensure all fields have values
                row_data = {field: row.get(field, self.null_value) for field in fields}
                writer.writerow(row_data)
            
            csv_content = output.getvalue()
            
            # Save to file if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding=self.encoding, newline='') as f:
                    f.write(csv_content)
                
                self.logger.info(f"CSV exported to {output_path}")
            
            return csv_content
            
        except Exception as e:
            self.logger.error(f"CSV export failed: {e}")
            raise ProcessingError(f"Failed to export CSV: {e}")
    
    def export_table(
        self,
        headers: List[str],
        rows: List[List[Any]],
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Export table data as CSV.
        
        Args:
            headers: Column headers
            rows: Table rows
            output_path: Optional output file path
            
        Returns:
            CSV string
        """
        try:
            output = io.StringIO()
            writer = csv.writer(
                output,
                delimiter=self.delimiter,
                quotechar=self.quotechar,
                quoting=self.quoting,
                lineterminator=self.lineterminator
            )
            
            # Write headers
            if self.include_headers:
                writer.writerow(headers)
            
            # Write rows
            for row in rows:
                processed_row = [self._convert_value(value) for value in row]
                writer.writerow(processed_row)
            
            csv_content = output.getvalue()
            
            # Save to file if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding=self.encoding, newline='') as f:
                    f.write(csv_content)
                
                self.logger.info(f"Table exported to {output_path}")
            
            return csv_content
            
        except Exception as e:
            self.logger.error(f"Table export failed: {e}")
            raise ProcessingError(f"Failed to export table: {e}")
    
    def export_batch(
        self,
        documents: List[Dict[str, Any]],
        output_path: Union[str, Path],
        separate_files: bool = False,
        filename_template: str = "document_{index}.csv"
    ) -> Union[Path, List[Path]]:
        """
        Export multiple documents.
        
        Args:
            documents: List of documents
            output_path: Output path (file or directory)
            separate_files: Export each document separately
            filename_template: Template for separate files
            
        Returns:
            Path or list of paths
        """
        output_path = Path(output_path)
        
        if separate_files:
            # Export to separate files
            output_path.mkdir(parents=True, exist_ok=True)
            exported_files = []
            
            for i, doc in enumerate(documents):
                filename = filename_template.format(
                    index=i,
                    **doc.get('metadata', {})
                )
                file_path = output_path / filename
                self.export_document(doc, file_path)
                exported_files.append(file_path)
            
            self.logger.info(f"Exported {len(documents)} CSV files to {output_path}")
            return exported_files
        else:
            # Export to single file
            self.export_document(documents, output_path)
            return output_path
    
    def _process_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single row of data"""
        processed = {}
        
        if self.flatten_nested:
            # Flatten nested structures
            flattened = self._flatten_dict(row)
            for key, value in flattened.items():
                if key not in self.excluded_fields:
                    # Apply field mapping
                    mapped_key = self.field_mapping.get(key, key)
                    processed[mapped_key] = self._convert_value(value)
        else:
            # Process without flattening
            for key, value in row.items():
                if key not in self.excluded_fields:
                    mapped_key = self.field_mapping.get(key, key)
                    processed[mapped_key] = self._convert_value(value)
        
        return processed
    
    def _flatten_dict(
        self,
        data: Dict[str, Any],
        parent_key: str = '',
        separator: str = '.'
    ) -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        
        for key, value in data.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.extend(
                    self._flatten_dict(value, new_key, separator).items()
                )
            elif isinstance(value, list):
                # Handle lists
                if value and isinstance(value[0], dict):
                    # List of dicts - take first item or create multiple rows
                    items.extend(
                        self._flatten_dict(value[0], new_key, separator).items()
                    )
                else:
                    # Simple list - convert to string
                    items.append((new_key, self._list_to_string(value)))
            else:
                items.append((new_key, value))
        
        return dict(items)
    
    def _determine_fields(self, rows: List[Dict[str, Any]]) -> List[str]:
        """Determine all unique fields from rows"""
        all_fields = set()
        
        for row in rows:
            all_fields.update(row.keys())
        
        # Remove excluded fields
        fields = [f for f in all_fields if f not in self.excluded_fields]
        
        return sorted(fields)
    
    def _get_headers(self, fields: List[str]) -> List[str]:
        """Get header names for fields"""
        headers = []
        
        for field in fields:
            # Use mapped name if available
            if field in self.field_mapping:
                headers.append(self.field_mapping[field])
            else:
                # Convert field name to readable header
                header = field.replace('_', ' ').replace('.', ' - ')
                header = header.title()
                headers.append(header)
        
        return headers
    
    def _convert_value(self, value: Any) -> str:
        """Convert value to CSV-compatible string"""
        if value is None:
            return self.null_value
        elif isinstance(value, bool):
            return 'true' if value else 'false'
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, datetime):
            return value.strftime(self.datetime_format)
        elif isinstance(value, date):
            return value.strftime(self.date_format)
        elif isinstance(value, (list, tuple)):
            return self._list_to_string(value)
        elif isinstance(value, dict):
            return json.dumps(value, separators=(',', ':'))
        else:
            return str(value)
    
    def _list_to_string(self, items: List[Any]) -> str:
        """Convert list to string representation"""
        if not items:
            return ""
        
        # Check if simple list
        if all(isinstance(item, (str, int, float, bool)) for item in items):
            return '; '.join(str(item) for item in items)
        else:
            # Complex list - use JSON
            return json.dumps(items, separators=(',', ':'))
    
    def export_from_dataframe(
        self,
        df: Any,  # pandas DataFrame
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> str:
        """
        Export pandas DataFrame as CSV.
        
        Args:
            df: pandas DataFrame
            output_path: Optional output file path
            **kwargs: Additional pandas to_csv arguments
            
        Returns:
            CSV string
        """
        try:
            # Prepare kwargs
            csv_kwargs = {
                'sep': self.delimiter,
                'encoding': self.encoding,
                'index': False,
                'date_format': self.datetime_format,
                'na_rep': self.null_value
            }
            csv_kwargs.update(kwargs)
            
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, **csv_kwargs)
                
                with open(output_path, 'r', encoding=self.encoding) as f:
                    return f.read()
            else:
                return df.to_csv(**csv_kwargs)
                
        except Exception as e:
            self.logger.error(f"DataFrame export failed: {e}")
            raise ProcessingError(f"Failed to export DataFrame: {e}")
    
    def create_pivot_table(
        self,
        data: List[Dict[str, Any]],
        index: str,
        columns: str,
        values: str,
        aggfunc: str = 'sum',
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Create pivot table from data.
        
        Args:
            data: List of records
            index: Field for row index
            columns: Field for columns
            values: Field for values
            aggfunc: Aggregation function
            output_path: Optional output path
            
        Returns:
            CSV string of pivot table
        """
        try:
            # Create pivot structure
            pivot_data = {}
            column_names = set()
            
            for record in data:
                row_key = record.get(index, 'Unknown')
                col_key = record.get(columns, 'Unknown')
                value = record.get(values, 0)
                
                if row_key not in pivot_data:
                    pivot_data[row_key] = {}
                
                column_names.add(col_key)
                
                if col_key not in pivot_data[row_key]:
                    pivot_data[row_key][col_key] = []
                
                pivot_data[row_key][col_key].append(value)
            
            # Apply aggregation
            column_names = sorted(column_names)
            rows = []
            
            for row_key in sorted(pivot_data.keys()):
                row = {index: row_key}
                
                for col_key in column_names:
                    values_list = pivot_data[row_key].get(col_key, [])
                    
                    if aggfunc == 'sum':
                        row[col_key] = sum(values_list)
                    elif aggfunc == 'mean':
                        row[col_key] = sum(values_list) / len(values_list) if values_list else 0
                    elif aggfunc == 'count':
                        row[col_key] = len(values_list)
                    elif aggfunc == 'min':
                        row[col_key] = min(values_list) if values_list else 0
                    elif aggfunc == 'max':
                        row[col_key] = max(values_list) if values_list else 0
                    else:
                        row[col_key] = values_list[0] if values_list else 0
                
                rows.append(row)
            
            # Export as CSV
            fields = [index] + column_names
            return self.export_document(rows, output_path, fields=fields)
            
        except Exception as e:
            self.logger.error(f"Pivot table creation failed: {e}")
            raise ProcessingError(f"Failed to create pivot table: {e}")


# Factory function
def create_csv_exporter(config: Optional[Dict[str, Any]] = None) -> CSVExporter:
    """Create and return a CSV exporter instance"""
    return CSVExporter(config)