#!/usr/bin/env python3
"""
JSON utilities for handling JSON data operations.

Provides robust JSON handling with schema validation, transformation,
patching, and streaming capabilities.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator, TextIO, Type, Callable
from datetime import datetime, date
from decimal import Decimal
import re
from collections import OrderedDict
import jsonschema
from jsonschema import validate, ValidationError as JsonSchemaValidationError
import jsonpointer
import jsonpatch

from ..core.logging import get_logger
from ..core.exceptions import ValidationError, ProcessingError


logger = get_logger(__name__)


class JSONEncoder(json.JSONEncoder):
    """Extended JSON encoder with support for additional types"""
    
    def default(self, obj):
        """Handle additional object types"""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, 'to_json'):
            return obj.to_json()
        return super().default(obj)


class JSONDecoder(json.JSONDecoder):
    """Extended JSON decoder with custom object hooks"""
    
    def __init__(self, *args, **kwargs):
        """Initialize decoder with custom object hook"""
        kwargs['object_hook'] = self.object_hook
        super().__init__(*args, **kwargs)
    
    def object_hook(self, obj):
        """Custom object deserialization"""
        # Handle ISO date strings
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    # Try to parse ISO datetime
                    if re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', value):
                        try:
                            obj[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                        except ValueError:
                            pass
        return obj


def load_json(
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    cls: Optional[Type[json.JSONDecoder]] = None,
    **kwargs
) -> Any:
    """
    Load JSON from file with error handling.
    
    Args:
        file_path: Path to JSON file
        encoding: File encoding
        cls: Custom decoder class
        **kwargs: Additional arguments for json.load
        
    Returns:
        Parsed JSON data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return json.load(f, cls=cls or JSONDecoder, **kwargs)
    except json.JSONDecodeError as e:
        raise ProcessingError(f"Invalid JSON in {file_path}: {e}")
    except Exception as e:
        raise ProcessingError(f"Error loading JSON from {file_path}: {e}")


def save_json(
    data: Any,
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    indent: int = 2,
    sort_keys: bool = False,
    cls: Optional[Type[json.JSONEncoder]] = None,
    ensure_ascii: bool = False,
    **kwargs
) -> None:
    """
    Save data as JSON to file.
    
    Args:
        data: Data to save
        file_path: Output file path
        encoding: File encoding
        indent: JSON indentation
        sort_keys: Sort dictionary keys
        cls: Custom encoder class
        ensure_ascii: Ensure ASCII output
        **kwargs: Additional arguments for json.dump
    """
    file_path = Path(file_path)
    
    # Create parent directory if needed
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(
                data,
                f,
                cls=cls or JSONEncoder,
                indent=indent,
                sort_keys=sort_keys,
                ensure_ascii=ensure_ascii,
                **kwargs
            )
    except Exception as e:
        raise ProcessingError(f"Error saving JSON to {file_path}: {e}")


def parse_json(
    json_string: str,
    cls: Optional[Type[json.JSONDecoder]] = None,
    **kwargs
) -> Any:
    """
    Parse JSON string with error handling.
    
    Args:
        json_string: JSON string to parse
        cls: Custom decoder class
        **kwargs: Additional arguments for json.loads
        
    Returns:
        Parsed data
    """
    try:
        return json.loads(json_string, cls=cls or JSONDecoder, **kwargs)
    except json.JSONDecodeError as e:
        raise ProcessingError(f"Invalid JSON: {e}")


def to_json(
    data: Any,
    indent: Optional[int] = None,
    sort_keys: bool = False,
    cls: Optional[Type[json.JSONEncoder]] = None,
    **kwargs
) -> str:
    """
    Convert data to JSON string.
    
    Args:
        data: Data to convert
        indent: JSON indentation
        sort_keys: Sort dictionary keys
        cls: Custom encoder class
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string
    """
    try:
        return json.dumps(
            data,
            cls=cls or JSONEncoder,
            indent=indent,
            sort_keys=sort_keys,
            **kwargs
        )
    except Exception as e:
        raise ProcessingError(f"Error converting to JSON: {e}")


def validate_json_schema(
    data: Any,
    schema: Dict[str, Any],
    raise_on_error: bool = True
) -> Tuple[bool, Optional[List[str]]]:
    """
    Validate data against JSON schema.
    
    Args:
        data: Data to validate
        schema: JSON schema
        raise_on_error: Raise exception on validation error
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    try:
        validate(instance=data, schema=schema)
        return True, None
    except JsonSchemaValidationError as e:
        errors = [e.message]
        
        # Collect all validation errors
        if hasattr(e, 'context'):
            for suberror in e.context:
                errors.append(f"  - {suberror.message}")
        
        if raise_on_error:
            raise ValidationError(f"JSON schema validation failed: {'; '.join(errors)}")
        
        return False, errors


def merge_json(
    base: Dict[str, Any],
    *updates: Dict[str, Any],
    deep: bool = True
) -> Dict[str, Any]:
    """
    Merge multiple JSON objects.
    
    Args:
        base: Base dictionary
        *updates: Dictionaries to merge
        deep: Perform deep merge
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for update in updates:
        if deep:
            _deep_merge(result, update)
        else:
            result.update(update)
    
    return result


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> None:
    """Recursively merge dictionaries"""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def flatten_json(
    data: Dict[str, Any],
    separator: str = '.',
    prefix: str = ''
) -> Dict[str, Any]:
    """
    Flatten nested JSON structure.
    
    Args:
        data: Nested dictionary
        separator: Key separator
        prefix: Key prefix
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            items.extend(flatten_json(value, separator, new_key).items())
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    items.extend(flatten_json(item, separator, f"{new_key}[{i}]").items())
                else:
                    items.append((f"{new_key}[{i}]", item))
        else:
            items.append((new_key, value))
    
    return dict(items)


def unflatten_json(
    data: Dict[str, Any],
    separator: str = '.'
) -> Dict[str, Any]:
    """
    Unflatten a flattened JSON structure.
    
    Args:
        data: Flattened dictionary
        separator: Key separator
        
    Returns:
        Nested dictionary
    """
    result = {}
    
    for key, value in data.items():
        parts = key.split(separator)
        current = result
        
        for i, part in enumerate(parts[:-1]):
            # Handle array indices
            if '[' in part and ']' in part:
                array_key, index = part.split('[')
                index = int(index.rstrip(']'))
                
                if array_key not in current:
                    current[array_key] = []
                
                # Extend array if needed
                while len(current[array_key]) <= index:
                    current[array_key].append({})
                
                current = current[array_key][index]
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        # Set the value
        final_key = parts[-1]
        if '[' in final_key and ']' in final_key:
            array_key, index = final_key.split('[')
            index = int(index.rstrip(']'))
            
            if array_key not in current:
                current[array_key] = []
            
            while len(current[array_key]) <= index:
                current[array_key].append(None)
            
            current[array_key][index] = value
        else:
            current[final_key] = value
    
    return result


def json_pointer_get(
    data: Any,
    pointer: str,
    default: Any = None
) -> Any:
    """
    Get value using JSON Pointer.
    
    Args:
        data: JSON data
        pointer: JSON Pointer string
        default: Default value if not found
        
    Returns:
        Value at pointer location
    """
    try:
        return jsonpointer.resolve_pointer(data, pointer)
    except jsonpointer.JsonPointerException:
        return default


def json_pointer_set(
    data: Any,
    pointer: str,
    value: Any
) -> Any:
    """
    Set value using JSON Pointer.
    
    Args:
        data: JSON data
        pointer: JSON Pointer string
        value: Value to set
        
    Returns:
        Modified data
    """
    try:
        return jsonpointer.set_pointer(data, pointer, value)
    except jsonpointer.JsonPointerException as e:
        raise ProcessingError(f"Invalid JSON Pointer: {e}")


def json_patch_apply(
    data: Any,
    patch: Union[str, List[Dict[str, Any]]]
) -> Any:
    """
    Apply JSON Patch to data.
    
    Args:
        data: Original data
        patch: JSON Patch document
        
    Returns:
        Patched data
    """
    try:
        if isinstance(patch, str):
            patch = json.loads(patch)
        
        patch_obj = jsonpatch.JsonPatch(patch)
        return patch_obj.apply(data)
    except Exception as e:
        raise ProcessingError(f"Failed to apply JSON Patch: {e}")


def json_patch_create(
    source: Any,
    target: Any
) -> List[Dict[str, Any]]:
    """
    Create JSON Patch from source to target.
    
    Args:
        source: Source data
        target: Target data
        
    Returns:
        JSON Patch operations
    """
    try:
        patch = jsonpatch.make_patch(source, target)
        return patch.patch
    except Exception as e:
        raise ProcessingError(f"Failed to create JSON Patch: {e}")


def stream_json_lines(
    file_path: Union[str, Path],
    encoding: str = 'utf-8'
) -> Iterator[Any]:
    """
    Stream JSON Lines file (one JSON object per line).
    
    Args:
        file_path: Path to JSON Lines file
        encoding: File encoding
        
    Yields:
        Parsed JSON objects
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding=encoding) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                yield parse_json(line)
            except ProcessingError as e:
                logger.warning(f"Invalid JSON on line {line_num}: {e}")


def write_json_lines(
    data: Iterator[Any],
    file_path: Union[str, Path],
    encoding: str = 'utf-8'
) -> int:
    """
    Write data as JSON Lines file.
    
    Args:
        data: Iterator of objects
        file_path: Output file path
        encoding: File encoding
        
    Returns:
        Number of lines written
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(file_path, 'w', encoding=encoding) as f:
        for item in data:
            f.write(to_json(item) + '\n')
            count += 1
    
    return count


def json_transform(
    data: Any,
    mapping: Dict[str, Union[str, Callable]],
    remove_unmapped: bool = False
) -> Any:
    """
    Transform JSON data using field mapping.
    
    Args:
        data: Input data
        mapping: Field mapping (old_key -> new_key or transform function)
        remove_unmapped: Remove fields not in mapping
        
    Returns:
        Transformed data
    """
    if isinstance(data, dict):
        result = {}
        
        for key, value in data.items():
            if key in mapping:
                map_value = mapping[key]
                
                if callable(map_value):
                    # Apply transformation function
                    transformed = map_value(value)
                    if isinstance(transformed, dict):
                        result.update(transformed)
                    else:
                        result[key] = transformed
                else:
                    # Simple key rename
                    result[map_value] = value
            elif not remove_unmapped:
                result[key] = json_transform(value, mapping, remove_unmapped)
        
        return result
    
    elif isinstance(data, list):
        return [json_transform(item, mapping, remove_unmapped) for item in data]
    
    else:
        return data


def json_filter(
    data: Any,
    condition: Callable[[Any], bool],
    path: Optional[str] = None
) -> Any:
    """
    Filter JSON data based on condition.
    
    Args:
        data: Input data
        condition: Filter condition function
        path: Optional JSON Pointer path to filter at
        
    Returns:
        Filtered data
    """
    if path:
        target = json_pointer_get(data, path)
        filtered = json_filter(target, condition)
        return json_pointer_set(data, path, filtered)
    
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if condition(v)}
    
    elif isinstance(data, list):
        return [item for item in data if condition(item)]
    
    else:
        return data if condition(data) else None


def json_sort(
    data: Any,
    key: Optional[Union[str, Callable]] = None,
    reverse: bool = False
) -> Any:
    """
    Sort JSON arrays or object keys.
    
    Args:
        data: Input data
        key: Sort key (field name or function)
        reverse: Reverse sort order
        
    Returns:
        Sorted data
    """
    if isinstance(data, dict):
        # Sort dictionary keys
        sorted_items = sorted(data.items(), reverse=reverse)
        return OrderedDict(sorted_items)
    
    elif isinstance(data, list):
        if key is None:
            return sorted(data, reverse=reverse)
        elif isinstance(key, str):
            return sorted(data, key=lambda x: x.get(key) if isinstance(x, dict) else x, reverse=reverse)
        else:
            return sorted(data, key=key, reverse=reverse)
    
    else:
        return data


def json_pretty_print(
    data: Any,
    indent: int = 2,
    sort_keys: bool = True,
    colors: bool = False
) -> str:
    """
    Pretty print JSON with optional colors.
    
    Args:
        data: JSON data
        indent: Indentation level
        sort_keys: Sort dictionary keys
        colors: Enable color output
        
    Returns:
        Formatted JSON string
    """
    json_str = to_json(data, indent=indent, sort_keys=sort_keys)
    
    if colors:
        try:
            from pygments import highlight
            from pygments.lexers import JsonLexer
            from pygments.formatters import TerminalFormatter
            
            return highlight(json_str, JsonLexer(), TerminalFormatter())
        except ImportError:
            logger.warning("Pygments not available for colored output")
    
    return json_str


def json_diff(
    old_data: Any,
    new_data: Any,
    ignore_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate difference between two JSON objects.
    
    Args:
        old_data: Original data
        new_data: New data
        ignore_keys: Keys to ignore in comparison
        
    Returns:
        Difference summary
    """
    patch = json_patch_create(old_data, new_data)
    
    # Filter out ignored keys
    if ignore_keys:
        patch = [
            op for op in patch
            if not any(key in op.get('path', '') for key in ignore_keys)
        ]
    
    # Summarize changes
    summary = {
        'added': [],
        'removed': [],
        'modified': [],
        'patch': patch
    }
    
    for op in patch:
        if op['op'] == 'add':
            summary['added'].append(op['path'])
        elif op['op'] == 'remove':
            summary['removed'].append(op['path'])
        elif op['op'] == 'replace':
            summary['modified'].append(op['path'])
    
    return summary


def json_to_csv_rows(
    data: List[Dict[str, Any]],
    fields: Optional[List[str]] = None
) -> List[List[Any]]:
    """
    Convert JSON array to CSV rows.
    
    Args:
        data: Array of JSON objects
        fields: Field names to include
        
    Returns:
        List of CSV rows
    """
    if not data:
        return []
    
    # Auto-detect fields if not provided
    if fields is None:
        fields = list(data[0].keys())
    
    rows = [fields]  # Header row
    
    for item in data:
        row = []
        for field in fields:
            value = item.get(field, '')
            
            # Handle nested objects/arrays
            if isinstance(value, (dict, list)):
                value = to_json(value)
            
            row.append(value)
        
        rows.append(row)
    
    return rows


def json_safe_parse(
    text: str,
    repair: bool = True
) -> Any:
    """
    Parse potentially malformed JSON with repair attempts.
    
    Args:
        text: JSON-like text
        repair: Attempt to repair malformed JSON
        
    Returns:
        Parsed data
    """
    # Try standard parsing first
    try:
        return parse_json(text)
    except ProcessingError:
        if not repair:
            raise
    
    # Attempt repairs
    repaired = text
    
    # Fix common issues
    repairs = [
        # Trailing commas
        (r',\s*}', '}'),
        (r',\s*]', ']'),
        # Single quotes
        (r"'([^']*)'", r'"\1"'),
        # Unquoted keys
        (r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":'),
    ]
    
    for pattern, replacement in repairs:
        repaired = re.sub(pattern, replacement, repaired)
    
    try:
        return parse_json(repaired)
    except ProcessingError:
        raise ProcessingError("Unable to parse or repair JSON")


# Convenience functions
def read_json(path: Union[str, Path]) -> Any:
    """Alias for load_json"""
    return load_json(path)


def write_json(data: Any, path: Union[str, Path], **kwargs) -> None:
    """Alias for save_json"""
    save_json(data, path, **kwargs)