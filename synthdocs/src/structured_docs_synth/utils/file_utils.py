#!/usr/bin/env python3
"""
File and directory utilities for system operations.

Provides functions for file I/O, directory management, compression,
monitoring, and safe file operations with error handling.
"""

import asyncio
import gzip
import json
import os
import shutil
import stat
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from ..core import get_logger

logger = get_logger(__name__)


def read_file(file_path: Union[str, Path], encoding: str = 'utf-8', 
             binary: bool = False) -> Union[str, bytes]:
    """
    Read file contents safely.
    
    Args:
        file_path: Path to file
        encoding: Text encoding (ignored if binary=True)
        binary: Whether to read in binary mode
    
    Returns:
        File contents as string or bytes
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        mode = 'rb' if binary else 'r'
        kwargs = {} if binary else {'encoding': encoding}
        
        with open(path, mode, **kwargs) as f:
            return f.read()
            
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise


def write_file(file_path: Union[str, Path], content: Union[str, bytes],
              encoding: str = 'utf-8', binary: bool = False,
              create_dirs: bool = True, backup: bool = False) -> bool:
    """
    Write content to file safely.
    
    Args:
        file_path: Path to file
        content: Content to write
        encoding: Text encoding (ignored if binary=True)
        binary: Whether to write in binary mode
        create_dirs: Create parent directories if they don't exist
        backup: Create backup of existing file
    
    Returns:
        True if successful
    """
    try:
        path = Path(file_path)
        
        # Create parent directories if needed
        if create_dirs and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if requested and file exists
        if backup and path.exists():
            backup_path = path.with_suffix(path.suffix + '.bak')
            shutil.copy2(path, backup_path)
            logger.debug(f"Created backup: {backup_path}")
        
        # Write content
        mode = 'wb' if binary else 'w'
        kwargs = {} if binary else {'encoding': encoding}
        
        with open(path, mode, **kwargs) as f:
            f.write(content)
        
        logger.debug(f"Successfully wrote to file: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        raise


def copy_file(src: Union[str, Path], dst: Union[str, Path], 
             preserve_metadata: bool = True) -> bool:
    """
    Copy file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        preserve_metadata: Whether to preserve file metadata
    
    Returns:
        True if successful
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src}")
        
        if not src_path.is_file():
            raise ValueError(f"Source is not a file: {src}")
        
        # Create destination directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        if preserve_metadata:
            shutil.copy2(src_path, dst_path)
        else:
            shutil.copy(src_path, dst_path)
        
        logger.debug(f"Successfully copied {src} to {dst}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to copy file {src} to {dst}: {e}")
        raise


def move_file(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """
    Move file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
    
    Returns:
        True if successful
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src}")
        
        # Create destination directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file
        shutil.move(str(src_path), str(dst_path))
        
        logger.debug(f"Successfully moved {src} to {dst}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to move file {src} to {dst}: {e}")
        raise


def delete_file(file_path: Union[str, Path], force: bool = False) -> bool:
    """
    Delete file safely.
    
    Args:
        file_path: Path to file
        force: Force deletion even if file is read-only
    
    Returns:
        True if successful or file doesn't exist
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            return True  # File doesn't exist, consider it successful
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Remove read-only attribute if force is True
        if force and not os.access(path, os.W_OK):
            path.chmod(stat.S_IWRITE)
        
        path.unlink()
        logger.debug(f"Successfully deleted file: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e}")
        raise


def create_directory(dir_path: Union[str, Path], parents: bool = True,
                    exist_ok: bool = True) -> bool:
    """
    Create directory.
    
    Args:
        dir_path: Directory path
        parents: Create parent directories
        exist_ok: Don't raise error if directory exists
    
    Returns:
        True if successful
    """
    try:
        path = Path(dir_path)
        path.mkdir(parents=parents, exist_ok=exist_ok)
        
        logger.debug(f"Successfully created directory: {dir_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create directory {dir_path}: {e}")
        raise


def list_directory(dir_path: Union[str, Path], pattern: str = "*",
                  recursive: bool = False, files_only: bool = False,
                  dirs_only: bool = False) -> List[Path]:
    """
    List directory contents.
    
    Args:
        dir_path: Directory path
        pattern: Glob pattern to match
        recursive: Search recursively
        files_only: Return only files
        dirs_only: Return only directories
    
    Returns:
        List of paths matching criteria
    """
    try:
        path = Path(dir_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")
        
        # Get all matching paths
        if recursive:
            paths = list(path.rglob(pattern))
        else:
            paths = list(path.glob(pattern))
        
        # Filter by type if requested
        if files_only:
            paths = [p for p in paths if p.is_file()]
        elif dirs_only:
            paths = [p for p in paths if p.is_dir()]
        
        return sorted(paths)
        
    except Exception as e:
        logger.error(f"Failed to list directory {dir_path}: {e}")
        raise


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get file information and metadata.
    
    Args:
        file_path: Path to file
    
    Returns:
        Dictionary with file information
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat_info = path.stat()
        
        return {
            'path': str(path.absolute()),
            'name': path.name,
            'stem': path.stem,
            'suffix': path.suffix,
            'size': stat_info.st_size,
            'created': stat_info.st_ctime,
            'modified': stat_info.st_mtime,
            'accessed': stat_info.st_atime,
            'is_file': path.is_file(),
            'is_dir': path.is_dir(),
            'is_symlink': path.is_symlink(),
            'permissions': oct(stat_info.st_mode)[-3:],
            'owner_readable': os.access(path, os.R_OK),
            'owner_writable': os.access(path, os.W_OK),
            'owner_executable': os.access(path, os.X_OK)
        }
        
    except Exception as e:
        logger.error(f"Failed to get file info for {file_path}: {e}")
        raise


def compress_file(file_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None,
                 compression: str = 'gzip') -> Path:
    """
    Compress file using specified compression.
    
    Args:
        file_path: Path to file to compress
        output_path: Output path (auto-generated if not provided)
        compression: Compression type ('gzip', 'zip')
    
    Returns:
        Path to compressed file
    """
    try:
        src_path = Path(file_path)
        
        if not src_path.exists() or not src_path.is_file():
            raise ValueError(f"Invalid source file: {file_path}")
        
        # Generate output path if not provided
        if output_path is None:
            if compression == 'gzip':
                output_path = src_path.with_suffix(src_path.suffix + '.gz')
            elif compression == 'zip':
                output_path = src_path.with_suffix('.zip')
            else:
                raise ValueError(f"Unsupported compression: {compression}")
        else:
            output_path = Path(output_path)
        
        # Compress file
        if compression == 'gzip':
            with open(src_path, 'rb') as f_in:
                with gzip.open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif compression == 'zip':
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(src_path, src_path.name)
        else:
            raise ValueError(f"Unsupported compression: {compression}")
        
        logger.info(f"Compressed {file_path} to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to compress file {file_path}: {e}")
        raise


def extract_archive(archive_path: Union[str, Path], 
                   extract_to: Optional[Union[str, Path]] = None) -> Path:
    """
    Extract archive file.
    
    Args:
        archive_path: Path to archive file
        extract_to: Directory to extract to (auto-generated if not provided)
    
    Returns:
        Path to extraction directory
    """
    try:
        archive_path = Path(archive_path)
        
        if not archive_path.exists() or not archive_path.is_file():
            raise ValueError(f"Invalid archive file: {archive_path}")
        
        # Generate extraction path if not provided
        if extract_to is None:
            extract_to = archive_path.parent / archive_path.stem
        else:
            extract_to = Path(extract_to)
        
        # Create extraction directory
        extract_to.mkdir(parents=True, exist_ok=True)
        
        # Extract based on file extension
        suffix = archive_path.suffix.lower()
        
        if suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(extract_to)
        elif suffix in ['.tar', '.gz', '.bz2', '.xz']:
            with tarfile.open(archive_path, 'r:*') as tarf:
                tarf.extractall(extract_to)
        elif suffix == '.gz' and archive_path.stem.endswith('.tar'):
            with tarfile.open(archive_path, 'r:gz') as tarf:
                tarf.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {suffix}")
        
        logger.info(f"Extracted {archive_path} to {extract_to}")
        return extract_to
        
    except Exception as e:
        logger.error(f"Failed to extract archive {archive_path}: {e}")
        raise


class DirectoryWatcher(FileSystemEventHandler):
    """File system event handler for directory watching"""
    
    def __init__(self, callback: callable):
        self.callback = callback
        super().__init__()
    
    def on_any_event(self, event: FileSystemEvent):
        try:
            self.callback(event)
        except Exception as e:
            logger.error(f"Error in directory watcher callback: {e}")


def watch_directory(dir_path: Union[str, Path], callback: callable,
                   recursive: bool = True) -> Observer:
    """
    Watch directory for file system changes.
    
    Args:
        dir_path: Directory to watch
        callback: Function to call on events
        recursive: Watch subdirectories
    
    Returns:
        Observer instance (call .stop() to stop watching)
    """
    try:
        path = Path(dir_path)
        
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid directory: {dir_path}")
        
        # Create and start observer
        event_handler = DirectoryWatcher(callback)
        observer = Observer()
        observer.schedule(event_handler, str(path), recursive=recursive)
        observer.start()
        
        logger.info(f"Started watching directory: {dir_path}")
        return observer
        
    except Exception as e:
        logger.error(f"Failed to start directory watcher for {dir_path}: {e}")
        raise


def safe_filename(filename: str, replace_char: str = '_') -> str:
    """
    Create safe filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        replace_char: Character to replace invalid chars with
    
    Returns:
        Safe filename
    """
    # Invalid characters for most filesystems
    invalid_chars = '<>:"/\\|?*'
    
    # Replace invalid characters
    safe_name = ''.join(
        replace_char if c in invalid_chars else c 
        for c in filename
    )
    
    # Remove control characters
    safe_name = ''.join(
        c for c in safe_name 
        if ord(c) >= 32
    )
    
    # Trim and remove multiple consecutive replace_chars
    safe_name = safe_name.strip()
    while replace_char + replace_char in safe_name:
        safe_name = safe_name.replace(replace_char + replace_char, replace_char)
    
    # Ensure filename is not empty
    if not safe_name:
        safe_name = 'unnamed'
    
    return safe_name


def create_temp_file(suffix: str = '', prefix: str = 'tmp', 
                    dir: Optional[Union[str, Path]] = None,
                    delete: bool = True) -> tempfile.NamedTemporaryFile:
    """
    Create temporary file.
    
    Args:
        suffix: File suffix
        prefix: File prefix
        dir: Directory for temp file
        delete: Whether to delete file when closed
    
    Returns:
        NamedTemporaryFile object
    """
    return tempfile.NamedTemporaryFile(
        suffix=suffix,
        prefix=prefix,
        dir=str(dir) if dir else None,
        delete=delete
    )


def create_temp_directory(suffix: str = '', prefix: str = 'tmp',
                         dir: Optional[Union[str, Path]] = None) -> str:
    """
    Create temporary directory.
    
    Args:
        suffix: Directory suffix
        prefix: Directory prefix
        dir: Parent directory for temp dir
    
    Returns:
        Path to temporary directory
    """
    return tempfile.mkdtemp(
        suffix=suffix,
        prefix=prefix,
        dir=str(dir) if dir else None
    )


def disk_usage(path: Union[str, Path]) -> Dict[str, int]:
    """
    Get disk usage statistics for path.
    
    Args:
        path: Path to check
    
    Returns:
        Dictionary with total, used, and free space in bytes
    """
    try:
        usage = shutil.disk_usage(path)
        return {
            'total': usage.total,
            'used': usage.used,
            'free': usage.free
        }
    except Exception as e:
        logger.error(f"Failed to get disk usage for {path}: {e}")
        raise


__all__ = [
    'read_file',
    'write_file',
    'copy_file',
    'move_file',
    'delete_file',
    'create_directory',
    'list_directory',
    'get_file_info',
    'compress_file',
    'extract_archive',
    'watch_directory',
    'DirectoryWatcher',
    'safe_filename',
    'create_temp_file',
    'create_temp_directory',
    'disk_usage'
]