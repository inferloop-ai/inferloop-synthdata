"""
Utility functions for the structured documents synthetic data system.

Provides core helper functions for cryptography, file operations,
format handling, validation, and testing.
"""

from .crypto_utils import (
    hash_string, encrypt_data, decrypt_data, generate_salt,
    create_secure_token, verify_signature, sign_data
)
from .file_utils import (
    read_file, write_file, copy_file, move_file, delete_file,
    create_directory, list_directory, get_file_info, compress_file,
    extract_archive, watch_directory
)
from .format_utils import (
    detect_format, convert_format, validate_format,
    parse_json, parse_xml, parse_csv, format_json,
    format_xml, format_csv
)
from .validation_utils import (
    validate_email, validate_url, validate_json_schema,
    validate_file_path, validate_image, validate_document,
    sanitize_filename, sanitize_input
)
from .test_utils import (
    create_mock_document, create_test_data, generate_test_files,
    setup_test_environment, cleanup_test_environment,
    assert_file_exists, assert_valid_json
)

__all__ = [
    # Crypto utilities
    'hash_string', 'encrypt_data', 'decrypt_data', 'generate_salt',
    'create_secure_token', 'verify_signature', 'sign_data',
    
    # File utilities
    'read_file', 'write_file', 'copy_file', 'move_file', 'delete_file',
    'create_directory', 'list_directory', 'get_file_info', 'compress_file',
    'extract_archive', 'watch_directory',
    
    # Format utilities
    'detect_format', 'convert_format', 'validate_format',
    'parse_json', 'parse_xml', 'parse_csv', 'format_json',
    'format_xml', 'format_csv',
    
    # Validation utilities
    'validate_email', 'validate_url', 'validate_json_schema',
    'validate_file_path', 'validate_image', 'validate_document',
    'sanitize_filename', 'sanitize_input',
    
    # Test utilities
    'create_mock_document', 'create_test_data', 'generate_test_files',
    'setup_test_environment', 'cleanup_test_environment',
    'assert_file_exists', 'assert_valid_json'
]