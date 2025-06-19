#!/usr/bin/env python3
"""
Validation utilities for data and input sanitization.

Provides functions for validating emails, URLs, file paths, images,
documents, and sanitizing user inputs for security.
"""

import os
import re
import string
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jsonschema
from PIL import Image

from ..core import get_logger

logger = get_logger(__name__)


def validate_email(email: str) -> Dict[str, Any]:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
    
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': False,
        'email': email,
        'errors': [],
        'warnings': []
    }
    
    try:
        # Basic format check
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not email:
            result['errors'].append('Email is empty')
            return result
        
        if not re.match(email_pattern, email):
            result['errors'].append('Invalid email format')
            return result
        
        # Split into local and domain parts
        local, domain = email.rsplit('@', 1)
        
        # Check local part length
        if len(local) > 64:
            result['errors'].append('Local part too long (max 64 characters)')
            return result
        
        # Check domain part length
        if len(domain) > 253:
            result['errors'].append('Domain part too long (max 253 characters)')
            return result
        
        # Check for consecutive dots
        if '..' in email:
            result['errors'].append('Consecutive dots not allowed')
            return result
        
        # Check for valid characters in local part
        valid_local_chars = string.ascii_letters + string.digits + '._%+-'
        if not all(c in valid_local_chars for c in local):
            result['errors'].append('Invalid characters in local part')
            return result
        
        # Check domain format
        domain_pattern = r'^[a-zA-Z0-9.-]+$'
        if not re.match(domain_pattern, domain):
            result['errors'].append('Invalid domain format')
            return result
        
        # Check for valid TLD
        if '.' not in domain:
            result['errors'].append('Domain must have a TLD')
            return result
        
        tld = domain.split('.')[-1]
        if len(tld) < 2:
            result['errors'].append('Invalid TLD length')
            return result
        
        result['valid'] = True
        result['local_part'] = local
        result['domain_part'] = domain
        result['tld'] = tld
        
    except Exception as e:
        result['errors'].append(f'Validation error: {str(e)}')
    
    return result


def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate URL format and structure.
    
    Args:
        url: URL to validate
        allowed_schemes: List of allowed URL schemes
    
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': False,
        'url': url,
        'errors': [],
        'warnings': []
    }
    
    if allowed_schemes is None:
        allowed_schemes = ['http', 'https', 'ftp', 'ftps']
    
    try:
        if not url:
            result['errors'].append('URL is empty')
            return result
        
        # Parse URL
        parsed = urllib.parse.urlparse(url)
        
        # Check scheme
        if not parsed.scheme:
            result['errors'].append('URL missing scheme')
            return result
        
        if parsed.scheme.lower() not in allowed_schemes:
            result['errors'].append(f'Scheme "{parsed.scheme}" not allowed')
            return result
        
        # Check netloc (domain)
        if not parsed.netloc:
            result['errors'].append('URL missing domain')
            return result
        
        # Basic domain validation
        domain = parsed.netloc.split(':')[0]  # Remove port if present
        if not re.match(r'^[a-zA-Z0-9.-]+$', domain):
            result['errors'].append('Invalid domain format')
            return result
        
        # Check for valid domain structure
        if '..' in domain or domain.startswith('.') or domain.endswith('.'):
            result['errors'].append('Invalid domain structure')
            return result
        
        result['valid'] = True
        result['scheme'] = parsed.scheme.lower()
        result['domain'] = domain
        result['port'] = parsed.port
        result['path'] = parsed.path
        result['query'] = parsed.query
        result['fragment'] = parsed.fragment
        
    except Exception as e:
        result['errors'].append(f'URL parsing error: {str(e)}')
    
    return result


def validate_json_schema(data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate data against JSON schema.
    
    Args:
        data: Data to validate
        schema: JSON schema definition
    
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': False,
        'errors': [],
        'warnings': []
    }
    
    try:
        jsonschema.validate(data, schema)
        result['valid'] = True
        
    except jsonschema.ValidationError as e:
        result['errors'].append(f'Schema validation error: {e.message}')
        if e.path:
            result['errors'].append(f'Error path: {" -> ".join(str(p) for p in e.path)}')
    except jsonschema.SchemaError as e:
        result['errors'].append(f'Invalid schema: {e.message}')
    except Exception as e:
        result['errors'].append(f'Validation error: {str(e)}')
    
    return result


def validate_file_path(file_path: Union[str, Path], 
                      must_exist: bool = False,
                      must_be_file: bool = False,
                      must_be_dir: bool = False,
                      allowed_extensions: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate file path and properties.
    
    Args:
        file_path: Path to validate
        must_exist: Whether path must exist
        must_be_file: Whether path must be a file
        must_be_dir: Whether path must be a directory
        allowed_extensions: List of allowed file extensions
    
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': False,
        'path': str(file_path),
        'errors': [],
        'warnings': []
    }
    
    try:
        path = Path(file_path)
        
        # Check if path is absolute or relative
        result['is_absolute'] = path.is_absolute()
        
        # Check path components for security issues
        path_str = str(path)
        if '..' in path_str:
            result['warnings'].append('Path contains ".." which may be unsafe')
        
        # Check for null bytes (security issue)
        if '\x00' in path_str:
            result['errors'].append('Path contains null bytes')
            return result
        
        # Check existence if required
        if must_exist and not path.exists():
            result['errors'].append('Path does not exist')
            return result
        
        if path.exists():
            result['exists'] = True
            result['is_file'] = path.is_file()
            result['is_dir'] = path.is_dir()
            result['is_symlink'] = path.is_symlink()
            
            # Check type requirements
            if must_be_file and not path.is_file():
                result['errors'].append('Path is not a file')
                return result
            
            if must_be_dir and not path.is_dir():
                result['errors'].append('Path is not a directory')
                return result
            
            # Get file info if it's a file
            if path.is_file():
                stat_info = path.stat()
                result['size'] = stat_info.st_size
                result['modified'] = stat_info.st_mtime
        
        # Check file extension if specified
        if allowed_extensions and path.suffix:
            extension = path.suffix.lower()
            if extension not in [ext.lower() for ext in allowed_extensions]:
                result['errors'].append(f'Extension "{extension}" not allowed')
                return result
        
        result['valid'] = True
        result['name'] = path.name
        result['stem'] = path.stem
        result['suffix'] = path.suffix
        result['parent'] = str(path.parent)
        
    except Exception as e:
        result['errors'].append(f'Path validation error: {str(e)}')
    
    return result


def validate_image(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate image file and get metadata.
    
    Args:
        file_path: Path to image file
    
    Returns:
        Dictionary with validation results and image info
    """
    result = {
        'valid': False,
        'path': str(file_path),
        'errors': [],
        'warnings': []
    }
    
    try:
        path = Path(file_path)
        
        if not path.exists():
            result['errors'].append('Image file does not exist')
            return result
        
        if not path.is_file():
            result['errors'].append('Path is not a file')
            return result
        
        # Try to open and validate image
        with Image.open(path) as img:
            result['valid'] = True
            result['format'] = img.format
            result['mode'] = img.mode
            result['size'] = img.size
            result['width'] = img.size[0]
            result['height'] = img.size[1]
            
            # Check for potential issues
            if img.size[0] * img.size[1] > 100000000:  # 100MP
                result['warnings'].append('Very large image (>100MP)')
            
            if img.mode not in ['RGB', 'RGBA', 'L', 'P']:
                result['warnings'].append(f'Uncommon color mode: {img.mode}')
            
            # Get additional info if available
            if hasattr(img, 'info'):
                result['info'] = img.info
            
            # Check for EXIF data
            if hasattr(img, '_getexif') and img._getexif():
                result['has_exif'] = True
            
    except Exception as e:
        result['errors'].append(f'Image validation error: {str(e)}')
    
    return result


def validate_document(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate document file format and basic properties.
    
    Args:
        file_path: Path to document file
    
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': False,
        'path': str(file_path),
        'errors': [],
        'warnings': []
    }
    
    try:
        path = Path(file_path)
        
        if not path.exists():
            result['errors'].append('Document file does not exist')
            return result
        
        if not path.is_file():
            result['errors'].append('Path is not a file')
            return result
        
        # Get file info
        stat_info = path.stat()
        result['size'] = stat_info.st_size
        result['extension'] = path.suffix.lower()
        
        # Check file size (warn if very large)
        if stat_info.st_size > 100 * 1024 * 1024:  # 100MB
            result['warnings'].append('Very large document file (>100MB)')
        
        # Check for supported document formats
        supported_extensions = {
            '.pdf', '.docx', '.doc', '.txt', '.rtf', '.odt',
            '.json', '.xml', '.csv', '.yaml', '.yml', '.html'
        }
        
        if result['extension'] not in supported_extensions:
            result['warnings'].append(f'Unsupported document format: {result["extension"]}')
        
        # Try to read file header for format verification
        with open(path, 'rb') as f:
            header = f.read(32)
        
        # Check for common file signatures
        if header.startswith(b'%PDF'):
            result['detected_format'] = 'pdf'
        elif header.startswith(b'PK\x03\x04'):  # ZIP-based formats like DOCX
            result['detected_format'] = 'zip_based'
        elif header.startswith(b'\xd0\xcf\x11\xe0'):  # MS Office legacy
            result['detected_format'] = 'ms_office_legacy'
        else:
            try:
                # Try to decode as text
                header.decode('utf-8')
                result['detected_format'] = 'text'
            except UnicodeDecodeError:
                result['detected_format'] = 'binary'
        
        result['valid'] = True
        
    except Exception as e:
        result['errors'].append(f'Document validation error: {str(e)}')
    
    return result


def sanitize_filename(filename: str, replacement: str = '_') -> str:
    """
    Sanitize filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        replacement: Character to replace invalid chars with
    
    Returns:
        Sanitized filename
    """
    # Invalid characters for most filesystems
    invalid_chars = '<>:"/\\|?*'
    
    # Remove or replace invalid characters
    sanitized = ''.join(
        replacement if c in invalid_chars or ord(c) < 32 else c
        for c in filename
    )
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    
    # Handle reserved names on Windows
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    name_without_ext = sanitized.split('.')[0].upper()
    if name_without_ext in reserved_names:
        sanitized = f"{replacement}{sanitized}"
    
    # Ensure filename is not empty
    if not sanitized:
        sanitized = 'unnamed'
    
    # Limit length (most filesystems support 255 chars)
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        max_name_len = 255 - len(ext)
        sanitized = name[:max_name_len] + ext
    
    return sanitized


def sanitize_input(text: str, allow_html: bool = False, 
                  max_length: Optional[int] = None) -> str:
    """
    Sanitize user input for security.
    
    Args:
        text: Input text to sanitize
        allow_html: Whether to allow HTML tags
        max_length: Maximum allowed length
    
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove or escape HTML if not allowed
    if not allow_html:
        # Simple HTML escaping
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&#x27;')
    
    # Remove control characters except newlines and tabs
    text = ''.join(
        c for c in text 
        if ord(c) >= 32 or c in '\n\t\r'
    )
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Limit length if specified
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    return text


def validate_password(password: str, min_length: int = 8,
                     require_uppercase: bool = True,
                     require_lowercase: bool = True,
                     require_digits: bool = True,
                     require_special: bool = True) -> Dict[str, Any]:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
        min_length: Minimum password length
        require_uppercase: Require uppercase letters
        require_lowercase: Require lowercase letters
        require_digits: Require digits
        require_special: Require special characters
    
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': False,
        'score': 0,
        'errors': [],
        'warnings': []
    }
    
    if not password:
        result['errors'].append('Password is empty')
        return result
    
    # Check length
    if len(password) < min_length:
        result['errors'].append(f'Password too short (minimum {min_length} characters)')
    else:
        result['score'] += 1
    
    # Check character requirements
    if require_uppercase and not re.search(r'[A-Z]', password):
        result['errors'].append('Password must contain uppercase letters')
    elif require_uppercase:
        result['score'] += 1
    
    if require_lowercase and not re.search(r'[a-z]', password):
        result['errors'].append('Password must contain lowercase letters')
    elif require_lowercase:
        result['score'] += 1
    
    if require_digits and not re.search(r'\d', password):
        result['errors'].append('Password must contain digits')
    elif require_digits:
        result['score'] += 1
    
    if require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        result['errors'].append('Password must contain special characters')
    elif require_special:
        result['score'] += 1
    
    # Check for common patterns
    if re.search(r'(.)\1{2,}', password):  # Repeated characters
        result['warnings'].append('Password contains repeated characters')
        result['score'] -= 0.5
    
    if re.search(r'(012|123|234|345|456|567|678|789|890)', password):
        result['warnings'].append('Password contains sequential numbers')
        result['score'] -= 0.5
    
    if re.search(r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)', password.lower()):
        result['warnings'].append('Password contains sequential letters')
        result['score'] -= 0.5
    
    # Common passwords check (simplified)
    common_passwords = {
        'password', '123456', 'password123', 'admin', 'letmein',
        'welcome', 'monkey', '1234567890', 'qwerty', 'abc123'
    }
    if password.lower() in common_passwords:
        result['errors'].append('Password is too common')
        result['score'] = 0
    
    # Set validity based on errors
    result['valid'] = len(result['errors']) == 0
    
    # Normalize score
    max_score = 5  # Length + 4 character types
    result['score'] = max(0, min(result['score'] / max_score, 1.0))
    
    return result


__all__ = [
    'validate_email',
    'validate_url',
    'validate_json_schema',
    'validate_file_path',
    'validate_image',
    'validate_document',
    'validate_password',
    'sanitize_filename',
    'sanitize_input'
]