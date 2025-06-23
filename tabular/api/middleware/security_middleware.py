"""
Security middleware for file upload validation and general security
"""

import os
import hashlib
import magic
from typing import List, Optional, Dict, Any, Set
from pathlib import Path
import tempfile
import aiofiles

from fastapi import HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse


class SecurityMiddleware:
    """Security middleware for the API"""
    
    def __init__(self, 
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB default
                 allowed_mime_types: Optional[List[str]] = None,
                 allowed_extensions: Optional[List[str]] = None,
                 scan_for_threats: bool = True):
        
        self.max_file_size = max_file_size
        self.allowed_mime_types = allowed_mime_types or [
            'text/csv',
            'text/plain',
            'application/csv',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/json',
            'application/x-parquet',
            'application/octet-stream'  # For parquet files
        ]
        self.allowed_extensions = allowed_extensions or [
            '.csv', '.txt', '.json', '.xlsx', '.xls', '.parquet', '.tsv'
        ]
        self.scan_for_threats = scan_for_threats
        
        # Initialize magic for MIME type detection
        self.mime_detector = magic.Magic(mime=True)
        
        # Malicious patterns to check
        self.malicious_patterns = [
            b'<?php',  # PHP code
            b'<script',  # JavaScript
            b'eval(',  # Eval functions
            b'exec(',  # Exec functions
            b'system(',  # System calls
            b'__import__',  # Python imports
            b'subprocess',  # Python subprocess
            b'os.system',  # OS commands
        ]
    
    async def validate_file_upload(self, file: UploadFile) -> Dict[str, Any]:
        """Validate uploaded file for security threats"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        # Check file size
        file_size = 0
        temp_file = None
        
        try:
            # Save to temporary file for scanning
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                temp_file = tmp.name
                
                # Read and write in chunks
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    
                    file_size += len(chunk)
                    
                    # Check size limit
                    if file_size > self.max_file_size:
                        validation_result['valid'] = False
                        validation_result['errors'].append(
                            f"File size ({file_size} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)"
                        )
                        return validation_result
                    
                    tmp.write(chunk)
            
            # Reset file position
            await file.seek(0)
            
            # Get file info
            validation_result['file_info'] = {
                'filename': file.filename,
                'size': file_size,
                'content_type': file.content_type
            }
            
            # Check file extension
            file_ext = Path(file.filename).suffix.lower() if file.filename else ''
            if file_ext not in self.allowed_extensions:
                validation_result['valid'] = False
                validation_result['errors'].append(
                    f"File extension '{file_ext}' not allowed. Allowed: {', '.join(self.allowed_extensions)}"
                )
            
            # Check MIME type
            detected_mime = self.mime_detector.from_file(temp_file)
            validation_result['file_info']['detected_mime'] = detected_mime
            
            if detected_mime not in self.allowed_mime_types:
                validation_result['warnings'].append(
                    f"Detected MIME type '{detected_mime}' not in allowed list"
                )
            
            # Scan for malicious content
            if self.scan_for_threats:
                threats_found = await self._scan_for_threats(temp_file)
                if threats_found:
                    validation_result['valid'] = False
                    validation_result['errors'].extend(threats_found)
            
            # Calculate file hash
            file_hash = await self._calculate_file_hash(temp_file)
            validation_result['file_info']['hash'] = file_hash
            
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
        
        return validation_result
    
    async def _scan_for_threats(self, file_path: str) -> List[str]:
        """Scan file for potential threats"""
        threats = []
        
        try:
            # Read file content for pattern matching
            async with aiofiles.open(file_path, 'rb') as f:
                # Read first 10KB for pattern matching
                content = await f.read(10240)
                
                # Check for malicious patterns
                for pattern in self.malicious_patterns:
                    if pattern in content:
                        threats.append(f"Potentially malicious pattern detected: {pattern.decode('utf-8', errors='ignore')}")
            
            # Additional checks based on file type
            if file_path.endswith(('.csv', '.txt', '.tsv')):
                # Check for formula injection in CSV
                async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    first_lines = await f.read(1024)
                    
                    # Check for formula injection patterns
                    formula_patterns = ['=', '@', '+', '-', '|']
                    for line in first_lines.split('\n')[:10]:  # Check first 10 lines
                        if line and line[0] in formula_patterns:
                            threats.append("Potential CSV injection detected")
                            break
            
        except Exception as e:
            threats.append(f"Error scanning file: {str(e)}")
        
        return threats
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        
        async with aiofiles.open(file_path, 'rb') as f:
            while True:
                chunk = await f.read(8192)
                if not chunk:
                    break
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    async def __call__(self, request: Request, call_next):
        """Middleware to add security headers and checks"""
        # Add security headers
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response


class FileUploadValidator:
    """Standalone file upload validator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.max_file_size = config.get('max_file_size', 100 * 1024 * 1024)
        self.allowed_extensions = set(config.get('allowed_extensions', [
            '.csv', '.txt', '.json', '.xlsx', '.xls', '.parquet', '.tsv'
        ]))
        self.allowed_mime_types = set(config.get('allowed_mime_types', [
            'text/csv',
            'text/plain',
            'application/csv',
            'application/json',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/x-parquet',
            'application/octet-stream'
        ]))
        
        # Column count limits
        self.max_columns = config.get('max_columns', 1000)
        self.max_rows_preview = config.get('max_rows_preview', 10000)
    
    async def validate_csv_structure(self, file_path: str) -> Dict[str, Any]:
        """Validate CSV file structure"""
        import pandas as pd
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'structure': {}
        }
        
        try:
            # Read first few rows to check structure
            df_preview = pd.read_csv(file_path, nrows=self.max_rows_preview)
            
            # Check column count
            num_columns = len(df_preview.columns)
            if num_columns > self.max_columns:
                validation_result['valid'] = False
                validation_result['errors'].append(
                    f"Too many columns ({num_columns}). Maximum allowed: {self.max_columns}"
                )
            
            # Check for empty column names
            empty_cols = [col for col in df_preview.columns if not str(col).strip()]
            if empty_cols:
                validation_result['warnings'].append(
                    f"Found {len(empty_cols)} empty column names"
                )
            
            # Get structure info
            validation_result['structure'] = {
                'columns': list(df_preview.columns),
                'column_count': num_columns,
                'row_count_preview': len(df_preview),
                'dtypes': {col: str(dtype) for col, dtype in df_preview.dtypes.items()},
                'null_counts': df_preview.isnull().sum().to_dict()
            }
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Error parsing CSV: {str(e)}")
        
        return validation_result