#!/usr/bin/env python3
"""
String manipulation and processing utilities.

Provides comprehensive string handling functions for text processing,
formatting, validation, and transformation.
"""

import re
import string
import unicodedata
from typing import List, Optional, Union, Dict, Any, Pattern, Tuple, Callable
from textwrap import wrap, dedent, indent
import hashlib
import base64
from difflib import SequenceMatcher
import random
from collections import Counter

from ..core.logging import get_logger
from ..core.exceptions import ValidationError


logger = get_logger(__name__)


# Common regex patterns
PATTERNS = {
    'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
    'url': re.compile(r'^https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)$'),
    'phone': re.compile(r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{4,6}$'),
    'ipv4': re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'),
    'ipv6': re.compile(r'^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))$'),
    'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.IGNORECASE),
    'credit_card': re.compile(r'^[0-9]{13,19}$'),
    'ssn': re.compile(r'^\d{3}-\d{2}-\d{4}$'),
    'hex_color': re.compile(r'^#(?:[0-9a-fA-F]{3}){1,2}$'),
    'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
    'alpha': re.compile(r'^[a-zA-Z]+$'),
    'numeric': re.compile(r'^[0-9]+$'),
    'whitespace': re.compile(r'\s+'),
}


def clean_text(
    text: str,
    normalize_whitespace: bool = True,
    strip: bool = True,
    remove_empty_lines: bool = False
) -> str:
    """
    Clean text by normalizing whitespace and removing unwanted characters.
    
    Args:
        text: Input text
        normalize_whitespace: Normalize multiple spaces to single space
        strip: Strip leading/trailing whitespace
        remove_empty_lines: Remove empty lines
        
    Returns:
        Cleaned text
    """
    if not text:
        return text
    
    result = text
    
    if normalize_whitespace:
        result = PATTERNS['whitespace'].sub(' ', result)
    
    if strip:
        result = result.strip()
    
    if remove_empty_lines:
        lines = [line for line in result.splitlines() if line.strip()]
        result = '\n'.join(lines)
    
    return result


def truncate(
    text: str,
    max_length: int,
    suffix: str = '...',
    whole_words: bool = True
) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated
        whole_words: Truncate at word boundaries
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    truncated_length = max_length - len(suffix)
    
    if truncated_length <= 0:
        return suffix[:max_length]
    
    if whole_words:
        # Find last space before limit
        truncated = text[:truncated_length]
        last_space = truncated.rfind(' ')
        
        if last_space > 0:
            truncated = truncated[:last_space]
    else:
        truncated = text[:truncated_length]
    
    return truncated + suffix


def wrap_text(
    text: str,
    width: int = 80,
    indent_str: str = '',
    subsequent_indent: Optional[str] = None
) -> str:
    """
    Wrap text to specified width.
    
    Args:
        text: Text to wrap
        width: Maximum line width
        indent_str: Indentation for all lines
        subsequent_indent: Indentation for continuation lines
        
    Returns:
        Wrapped text
    """
    if subsequent_indent is None:
        subsequent_indent = indent_str
    
    lines = []
    for paragraph in text.split('\n'):
        if paragraph.strip():
            wrapped = wrap(
                paragraph,
                width=width,
                initial_indent=indent_str,
                subsequent_indent=subsequent_indent
            )
            lines.extend(wrapped)
        else:
            lines.append('')
    
    return '\n'.join(lines)


def snake_case(text: str) -> str:
    """Convert text to snake_case"""
    # Replace non-alphanumeric with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Replace spaces and convert to lowercase
    text = re.sub(r'\s+', '_', text.strip().lower())
    # Remove consecutive underscores
    text = re.sub(r'_+', '_', text)
    return text


def camel_case(text: str, pascal: bool = False) -> str:
    """
    Convert text to camelCase or PascalCase.
    
    Args:
        text: Input text
        pascal: Use PascalCase (first letter uppercase)
        
    Returns:
        Converted text
    """
    words = re.sub(r'[^\w\s]', ' ', text).split()
    
    if not words:
        return ''
    
    if pascal:
        return ''.join(word.capitalize() for word in words)
    else:
        return words[0].lower() + ''.join(word.capitalize() for word in words[1:])


def kebab_case(text: str) -> str:
    """Convert text to kebab-case"""
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', '-', text.strip().lower())
    text = re.sub(r'-+', '-', text)
    return text


def title_case(text: str, exceptions: Optional[List[str]] = None) -> str:
    """
    Convert text to Title Case with exceptions.
    
    Args:
        text: Input text
        exceptions: Words to keep lowercase (e.g., 'a', 'the', 'of')
        
    Returns:
        Title cased text
    """
    if exceptions is None:
        exceptions = ['a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 
                      'from', 'in', 'into', 'of', 'on', 'or', 'the', 'to',
                      'with']
    
    words = text.split()
    result = []
    
    for i, word in enumerate(words):
        if i == 0 or word.lower() not in exceptions:
            result.append(word.capitalize())
        else:
            result.append(word.lower())
    
    return ' '.join(result)


def remove_accents(text: str) -> str:
    """Remove accents from Unicode text"""
    nfd_form = unicodedata.normalize('NFD', text)
    return ''.join(char for char in nfd_form if unicodedata.category(char) != 'Mn')


def slugify(
    text: str,
    separator: str = '-',
    lowercase: bool = True,
    max_length: Optional[int] = None
) -> str:
    """
    Create URL-safe slug from text.
    
    Args:
        text: Input text
        separator: Word separator
        lowercase: Convert to lowercase
        max_length: Maximum slug length
        
    Returns:
        Slugified text
    """
    # Remove accents
    text = remove_accents(text)
    
    # Replace non-alphanumeric with separator
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', separator, text)
    
    # Clean up
    text = text.strip(separator)
    
    if lowercase:
        text = text.lower()
    
    if max_length:
        text = text[:max_length].rstrip(separator)
    
    return text


def validate_pattern(
    text: str,
    pattern: Union[str, Pattern],
    pattern_name: Optional[str] = None
) -> bool:
    """
    Validate text against regex pattern.
    
    Args:
        text: Text to validate
        pattern: Regex pattern or pattern name
        pattern_name: Optional pattern name for error messages
        
    Returns:
        True if valid
    """
    if isinstance(pattern, str):
        if pattern in PATTERNS:
            pattern_obj = PATTERNS[pattern]
            pattern_name = pattern_name or pattern
        else:
            pattern_obj = re.compile(pattern)
    else:
        pattern_obj = pattern
    
    return bool(pattern_obj.match(text))


def extract_pattern(
    text: str,
    pattern: Union[str, Pattern],
    group: int = 0
) -> Optional[str]:
    """
    Extract first match of pattern from text.
    
    Args:
        text: Text to search
        pattern: Regex pattern
        group: Capture group to return
        
    Returns:
        Matched text or None
    """
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    
    match = pattern.search(text)
    if match:
        return match.group(group)
    
    return None


def extract_all_patterns(
    text: str,
    pattern: Union[str, Pattern],
    group: int = 0
) -> List[str]:
    """
    Extract all matches of pattern from text.
    
    Args:
        text: Text to search
        pattern: Regex pattern
        group: Capture group to return
        
    Returns:
        List of matched texts
    """
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    
    return [match.group(group) for match in pattern.finditer(text)]


def mask_sensitive(
    text: str,
    patterns: Optional[Dict[str, str]] = None,
    mask_char: str = '*',
    partial: bool = True
) -> str:
    """
    Mask sensitive information in text.
    
    Args:
        text: Text containing sensitive data
        patterns: Pattern name -> replacement mapping
        mask_char: Character to use for masking
        partial: Partially mask (show first/last chars)
        
    Returns:
        Masked text
    """
    if patterns is None:
        patterns = {
            'email': lambda m: _mask_email(m.group(), mask_char),
            'phone': lambda m: _mask_phone(m.group(), mask_char),
            'credit_card': lambda m: _mask_credit_card(m.group(), mask_char),
            'ssn': lambda m: mask_char * len(m.group())
        }
    
    result = text
    
    for pattern_name, replacement in patterns.items():
        if pattern_name in PATTERNS:
            pattern = PATTERNS[pattern_name]
            if callable(replacement):
                result = pattern.sub(replacement, result)
            else:
                result = pattern.sub(replacement, result)
    
    return result


def _mask_email(email: str, mask_char: str) -> str:
    """Mask email address"""
    parts = email.split('@')
    if len(parts) != 2:
        return email
    
    username, domain = parts
    if len(username) > 2:
        masked_username = username[0] + mask_char * (len(username) - 2) + username[-1]
    else:
        masked_username = mask_char * len(username)
    
    return f"{masked_username}@{domain}"


def _mask_phone(phone: str, mask_char: str) -> str:
    """Mask phone number"""
    digits = re.sub(r'\D', '', phone)
    if len(digits) >= 4:
        visible_end = digits[-4:]
        masked = mask_char * (len(digits) - 4) + visible_end
        return masked
    return mask_char * len(digits)


def _mask_credit_card(card: str, mask_char: str) -> str:
    """Mask credit card number"""
    digits = re.sub(r'\D', '', card)
    if len(digits) >= 4:
        visible_end = digits[-4:]
        masked = mask_char * (len(digits) - 4) + visible_end
        return masked
    return mask_char * len(digits)


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        
        previous_row = current_row
    
    return previous_row[-1]


def similarity_ratio(s1: str, s2: str) -> float:
    """
    Calculate similarity ratio between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity ratio (0.0 to 1.0)
    """
    return SequenceMatcher(None, s1, s2).ratio()


def find_similar(
    target: str,
    candidates: List[str],
    threshold: float = 0.6,
    limit: Optional[int] = None
) -> List[Tuple[str, float]]:
    """
    Find similar strings from candidates.
    
    Args:
        target: Target string
        candidates: List of candidate strings
        threshold: Minimum similarity threshold
        limit: Maximum results to return
        
    Returns:
        List of (candidate, similarity) tuples
    """
    similarities = []
    
    for candidate in candidates:
        ratio = similarity_ratio(target, candidate)
        if ratio >= threshold:
            similarities.append((candidate, ratio))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    if limit:
        similarities = similarities[:limit]
    
    return similarities


def expand_template(
    template: str,
    variables: Dict[str, Any],
    safe: bool = True
) -> str:
    """
    Expand template with variables.
    
    Args:
        template: Template string with {variable} placeholders
        variables: Variable values
        safe: Use safe substitution (ignore missing variables)
        
    Returns:
        Expanded string
    """
    if safe:
        # Use safe substitution
        from string import Template
        
        # Convert {var} to $var format
        template_str = re.sub(r'\{(\w+)\}', r'$\1', template)
        tmpl = Template(template_str)
        
        try:
            return tmpl.safe_substitute(variables)
        except Exception:
            return template
    else:
        try:
            return template.format(**variables)
        except KeyError as e:
            raise ValidationError(f"Missing template variable: {e}")


def generate_random_string(
    length: int,
    charset: str = string.ascii_letters + string.digits,
    prefix: str = '',
    suffix: str = ''
) -> str:
    """
    Generate random string.
    
    Args:
        length: String length (excluding prefix/suffix)
        charset: Characters to use
        prefix: String prefix
        suffix: String suffix
        
    Returns:
        Random string
    """
    random_part = ''.join(random.choice(charset) for _ in range(length))
    return prefix + random_part + suffix


def hash_string(
    text: str,
    algorithm: str = 'sha256',
    encoding: str = 'utf-8'
) -> str:
    """
    Hash string using specified algorithm.
    
    Args:
        text: Text to hash
        algorithm: Hash algorithm
        encoding: Text encoding
        
    Returns:
        Hex digest
    """
    hasher = hashlib.new(algorithm)
    hasher.update(text.encode(encoding))
    return hasher.hexdigest()


def encode_base64(
    text: str,
    encoding: str = 'utf-8',
    urlsafe: bool = False
) -> str:
    """
    Encode string to base64.
    
    Args:
        text: Text to encode
        encoding: Text encoding
        urlsafe: Use URL-safe encoding
        
    Returns:
        Base64 encoded string
    """
    text_bytes = text.encode(encoding)
    
    if urlsafe:
        return base64.urlsafe_b64encode(text_bytes).decode('ascii')
    else:
        return base64.b64encode(text_bytes).decode('ascii')


def decode_base64(
    encoded: str,
    encoding: str = 'utf-8',
    urlsafe: bool = False
) -> str:
    """
    Decode base64 string.
    
    Args:
        encoded: Base64 encoded string
        encoding: Output text encoding
        urlsafe: Use URL-safe decoding
        
    Returns:
        Decoded string
    """
    try:
        if urlsafe:
            decoded_bytes = base64.urlsafe_b64decode(encoded)
        else:
            decoded_bytes = base64.b64decode(encoded)
        
        return decoded_bytes.decode(encoding)
    except Exception as e:
        raise ValidationError(f"Invalid base64 string: {e}")


def count_words(text: str) -> int:
    """Count words in text"""
    return len(text.split())


def count_sentences(text: str) -> int:
    """Count sentences in text"""
    # Simple sentence counting
    sentence_endings = re.findall(r'[.!?]+', text)
    return len(sentence_endings)


def extract_numbers(
    text: str,
    integers_only: bool = False
) -> List[Union[int, float]]:
    """
    Extract numbers from text.
    
    Args:
        text: Input text
        integers_only: Extract only integers
        
    Returns:
        List of numbers
    """
    if integers_only:
        pattern = r'-?\d+'
        converter = int
    else:
        pattern = r'-?\d+\.?\d*'
        converter = lambda x: int(x) if '.' not in x else float(x)
    
    matches = re.findall(pattern, text)
    return [converter(match) for match in matches]


def word_frequency(
    text: str,
    ignore_case: bool = True,
    min_length: int = 1
) -> Dict[str, int]:
    """
    Calculate word frequency in text.
    
    Args:
        text: Input text
        ignore_case: Case-insensitive counting
        min_length: Minimum word length
        
    Returns:
        Word frequency dictionary
    """
    if ignore_case:
        text = text.lower()
    
    # Extract words
    words = re.findall(r'\b\w+\b', text)
    
    # Filter by length
    words = [w for w in words if len(w) >= min_length]
    
    return dict(Counter(words))


def highlight_text(
    text: str,
    highlights: List[str],
    prefix: str = '**',
    suffix: str = '**',
    ignore_case: bool = True
) -> str:
    """
    Highlight specific text portions.
    
    Args:
        text: Input text
        highlights: Text to highlight
        prefix: Highlight prefix
        suffix: Highlight suffix
        ignore_case: Case-insensitive matching
        
    Returns:
        Text with highlights
    """
    result = text
    
    for highlight in highlights:
        flags = re.IGNORECASE if ignore_case else 0
        pattern = re.escape(highlight)
        replacement = f"{prefix}{highlight}{suffix}"
        result = re.sub(pattern, replacement, result, flags=flags)
    
    return result


def normalize_spacing(text: str) -> str:
    """Normalize all spacing in text"""
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Normalize spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def reverse_string(text: str) -> str:
    """Reverse a string"""
    return text[::-1]


def is_palindrome(text: str, ignore_case: bool = True, ignore_spaces: bool = True) -> bool:
    """Check if text is a palindrome"""
    if ignore_case:
        text = text.lower()
    
    if ignore_spaces:
        text = re.sub(r'\s+', '', text)
    
    return text == text[::-1]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    
    # Clean up
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def join_lines(
    lines: List[str],
    separator: str = ' ',
    strip: bool = True
) -> str:
    """
    Join lines of text.
    
    Args:
        lines: Lines to join
        separator: Line separator
        strip: Strip each line
        
    Returns:
        Joined text
    """
    if strip:
        lines = [line.strip() for line in lines]
    
    return separator.join(lines)


def indent_text(
    text: str,
    spaces: int = 4,
    first_line: bool = True
) -> str:
    """
    Indent text.
    
    Args:
        text: Text to indent
        spaces: Number of spaces
        first_line: Indent first line
        
    Returns:
        Indented text
    """
    indent_str = ' ' * spaces
    
    if first_line:
        return indent(text, indent_str)
    else:
        lines = text.splitlines()
        if lines:
            result = [lines[0]]
            result.extend(indent_str + line for line in lines[1:])
            return '\n'.join(result)
        return text


def remove_extra_whitespace(text: str) -> str:
    """Remove all extra whitespace from text"""
    # Remove extra spaces
    text = re.sub(r' +', ' ', text)
    
    # Remove extra tabs
    text = re.sub(r'\t+', '\t', text)
    
    # Remove extra newlines
    text = re.sub(r'\n+', '\n', text)
    
    # Remove trailing whitespace from lines
    lines = [line.rstrip() for line in text.splitlines()]
    
    return '\n'.join(lines).strip()