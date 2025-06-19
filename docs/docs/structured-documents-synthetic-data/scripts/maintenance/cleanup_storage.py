#!/usr/bin/env python3
"""
Storage cleanup script for removing old files and optimizing disk usage.

Provides automated cleanup of temporary files, logs, caches,
and old generated content based on configurable policies.
"""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import json
import sqlite3

# Cleanup configuration
CLEANUP_CONFIG = {
    'temp_file_age_hours': 24,
    'log_retention_days': 30,
    'cache_retention_days': 7,
    'generated_content_retention_days': 90,
    'max_log_size_mb': 100,
    'max_cache_size_mb': 500,
    'vacuum_databases': True,
    'compress_old_logs': True
}

DEFAULT_DATA_DIR = Path.home() / '.structured_docs_synth'


class StorageCleanup:
    """Comprehensive storage cleanup system"""
    
    def __init__(self, data_dir: Optional[Path] = None, 
                 config: Optional[Dict[str, Any]] = None):
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.config = {**CLEANUP_CONFIG, **(config or {})}
        self.cleanup_stats = {
            'files_removed': 0,
            'directories_removed': 0,
            'space_freed_mb': 0,
            'errors': []
        }
    
    async def run_full_cleanup(self) -> Dict[str, Any]:
        """
        Run complete storage cleanup.
        
        Returns:
            Cleanup results
        """
        print(">ù Starting full storage cleanup...")
        
        cleanup_tasks = [
            ('temp_files', self._cleanup_temp_files()),
            ('logs', self._cleanup_logs()),
            ('cache', self._cleanup_cache()),
            ('generated_content', self._cleanup_generated_content()),
            ('databases', self._vacuum_databases()),
            ('empty_directories', self._remove_empty_directories())
        ]
        
        results = {}
        
        for task_name, task_coro in cleanup_tasks:
            try:
                print(f"= Running {task_name} cleanup...")
                result = await task_coro
                results[task_name] = result
                
                if result.get('success', True):
                    freed_mb = result.get('space_freed_mb', 0)
                    files_removed = result.get('files_removed', 0)
                    print(f" {task_name}: {files_removed} files, {freed_mb:.1f} MB freed")
                else:
                    print(f"   {task_name}: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                error_msg = f"Error in {task_name}: {str(e)}"
                print(f"L {error_msg}")
                self.cleanup_stats['errors'].append(error_msg)
                results[task_name] = {'success': False, 'error': str(e)}
        
        # Summary
        total_space_freed = sum(
            result.get('space_freed_mb', 0) 
            for result in results.values()
        )
        total_files_removed = sum(
            result.get('files_removed', 0) 
            for result in results.values()
        )
        
        print(f"\n=Ê Cleanup Summary:")
        print(f"Total files removed: {total_files_removed}")
        print(f"Total space freed: {total_space_freed:.1f} MB")
        print(f"Errors: {len(self.cleanup_stats['errors'])}")
        
        return {
            'success': True,
            'total_files_removed': total_files_removed,
            'total_space_freed_mb': total_space_freed,
            'task_results': results,
            'errors': self.cleanup_stats['errors']
        }
    
    async def _cleanup_temp_files(self) -> Dict[str, Any]:
        """Clean up temporary files"""
        temp_patterns = [
            '*.tmp', '*.temp', '*~', '.DS_Store', 'Thumbs.db',
            '*.pyc', '__pycache__', '*.log.tmp'
        ]
        
        cutoff_time = datetime.now() - timedelta(hours=self.config['temp_file_age_hours'])
        
        files_removed = 0
        space_freed = 0
        
        # Clean from data directory
        for pattern in temp_patterns:
            for file_path in self.data_dir.rglob(pattern):
                try:
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_time:
                            size = file_path.stat().st_size
                            file_path.unlink()
                            files_removed += 1
                            space_freed += size
                    elif file_path.is_dir() and pattern == '__pycache__':
                        size = self._get_directory_size(file_path)
                        shutil.rmtree(file_path)
                        files_removed += 1
                        space_freed += size
                except Exception as e:
                    self.cleanup_stats['errors'].append(f"Error removing {file_path}: {e}")
        
        # Clean system temp directory
        system_temp = Path(tempfile.gettempdir())
        for temp_file in system_temp.glob('structured_docs_synth_*'):
            try:
                file_time = datetime.fromtimestamp(temp_file.stat().st_mtime)
                if file_time < cutoff_time:
                    size = temp_file.stat().st_size if temp_file.is_file() else self._get_directory_size(temp_file)
                    
                    if temp_file.is_file():
                        temp_file.unlink()
                    else:
                        shutil.rmtree(temp_file)
                    
                    files_removed += 1
                    space_freed += size
            except Exception as e:
                self.cleanup_stats['errors'].append(f"Error removing temp {temp_file}: {e}")
        
        return {
            'success': True,
            'files_removed': files_removed,
            'space_freed_mb': space_freed / (1024 * 1024)
        }
    
    async def _cleanup_logs(self) -> Dict[str, Any]:
        """Clean up old log files"""
        logs_dir = self.data_dir / 'logs'
        
        if not logs_dir.exists():
            return {'success': True, 'files_removed': 0, 'space_freed_mb': 0}
        
        cutoff_time = datetime.now() - timedelta(days=self.config['log_retention_days'])
        max_size = self.config['max_log_size_mb'] * 1024 * 1024
        
        files_removed = 0
        space_freed = 0
        
        # Remove old log files
        for log_file in logs_dir.rglob('*.log'):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                file_size = log_file.stat().st_size
                
                should_remove = False
                
                # Remove if too old
                if file_time < cutoff_time:
                    should_remove = True
                
                # Remove if too large
                elif file_size > max_size:
                    should_remove = True
                
                if should_remove:
                    space_freed += file_size
                    log_file.unlink()
                    files_removed += 1
                
                # Compress old logs if enabled
                elif (self.config['compress_old_logs'] and 
                      file_time < datetime.now() - timedelta(days=7) and
                      not log_file.name.endswith('.gz')):
                    
                    compressed_size = await self._compress_file(log_file)
                    if compressed_size > 0:
                        space_freed += file_size - compressed_size
            
            except Exception as e:
                self.cleanup_stats['errors'].append(f"Error processing log {log_file}: {e}")
        
        return {
            'success': True,
            'files_removed': files_removed,
            'space_freed_mb': space_freed / (1024 * 1024)
        }
    
    async def _cleanup_cache(self) -> Dict[str, Any]:
        """Clean up cache files"""
        cache_dirs = [
            self.data_dir / 'cache',
            self.data_dir / '.cache',
            self.data_dir / 'tmp'
        ]
        
        cutoff_time = datetime.now() - timedelta(days=self.config['cache_retention_days'])
        max_cache_size = self.config['max_cache_size_mb'] * 1024 * 1024
        
        files_removed = 0
        space_freed = 0
        
        for cache_dir in cache_dirs:
            if not cache_dir.exists():
                continue
            
            # Calculate current cache size
            current_size = self._get_directory_size(cache_dir)
            
            # If cache is too large, remove oldest files first
            if current_size > max_cache_size:
                cache_files = [
                    (f, f.stat().st_mtime) 
                    for f in cache_dir.rglob('*') 
                    if f.is_file()
                ]
                cache_files.sort(key=lambda x: x[1])  # Sort by modification time
                
                for file_path, _ in cache_files:
                    if current_size <= max_cache_size:
                        break
                    
                    try:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        files_removed += 1
                        space_freed += file_size
                        current_size -= file_size
                    except Exception as e:
                        self.cleanup_stats['errors'].append(f"Error removing cache {file_path}: {e}")
            
            # Remove old cache files
            for cache_file in cache_dir.rglob('*'):
                if cache_file.is_file():
                    try:
                        file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                        if file_time < cutoff_time:
                            file_size = cache_file.stat().st_size
                            cache_file.unlink()
                            files_removed += 1
                            space_freed += file_size
                    except Exception as e:
                        self.cleanup_stats['errors'].append(f"Error removing old cache {cache_file}: {e}")
        
        return {
            'success': True,
            'files_removed': files_removed,
            'space_freed_mb': space_freed / (1024 * 1024)
        }
    
    async def _cleanup_generated_content(self) -> Dict[str, Any]:
        """Clean up old generated content"""
        content_dirs = [
            self.data_dir / 'generated',
            self.data_dir / 'output',
            self.data_dir / 'content' / 'generated'
        ]
        
        cutoff_time = datetime.now() - timedelta(days=self.config['generated_content_retention_days'])
        
        files_removed = 0
        space_freed = 0
        
        for content_dir in content_dirs:
            if not content_dir.exists():
                continue
            
            for content_file in content_dir.rglob('*'):
                if content_file.is_file():
                    try:
                        file_time = datetime.fromtimestamp(content_file.stat().st_mtime)
                        if file_time < cutoff_time:
                            file_size = content_file.stat().st_size
                            content_file.unlink()
                            files_removed += 1
                            space_freed += file_size
                    except Exception as e:
                        self.cleanup_stats['errors'].append(f"Error removing content {content_file}: {e}")
        
        return {
            'success': True,
            'files_removed': files_removed,
            'space_freed_mb': space_freed / (1024 * 1024)
        }
    
    async def _vacuum_databases(self) -> Dict[str, Any]:
        """Vacuum SQLite databases to reclaim space"""
        if not self.config['vacuum_databases']:
            return {'success': True, 'databases_vacuumed': 0, 'space_freed_mb': 0}
        
        db_files = list(self.data_dir.rglob('*.db')) + list(self.data_dir.rglob('*.sqlite'))
        
        databases_vacuumed = 0
        space_freed = 0
        
        for db_file in db_files:
            try:
                # Get size before vacuum
                size_before = db_file.stat().st_size
                
                # Vacuum database
                conn = sqlite3.connect(db_file)
                conn.execute('VACUUM')
                conn.close()
                
                # Get size after vacuum
                size_after = db_file.stat().st_size
                
                space_freed += size_before - size_after
                databases_vacuumed += 1
                
                print(f"=Ã  Vacuumed {db_file.name}: {(size_before - size_after) / (1024 * 1024):.1f} MB freed")
                
            except Exception as e:
                self.cleanup_stats['errors'].append(f"Error vacuuming {db_file}: {e}")
        
        return {
            'success': True,
            'databases_vacuumed': databases_vacuumed,
            'space_freed_mb': space_freed / (1024 * 1024)
        }
    
    async def _remove_empty_directories(self) -> Dict[str, Any]:
        """Remove empty directories"""
        directories_removed = 0
        
        # Walk directories bottom-up to catch nested empty dirs
        for dirpath, dirnames, filenames in os.walk(self.data_dir, topdown=False):
            dir_path = Path(dirpath)
            
            # Skip backup and system directories
            if any(part in dir_path.parts for part in ['backups', '.git', '__pycache__']):
                continue
            
            try:
                # Check if directory is empty
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    directories_removed += 1
            except (OSError, PermissionError):
                # Directory not empty or permission issue
                pass
            except Exception as e:
                self.cleanup_stats['errors'].append(f"Error removing empty dir {dir_path}: {e}")
        
        return {
            'success': True,
            'directories_removed': directories_removed,
            'space_freed_mb': 0  # Empty directories don't take significant space
        }
    
    async def _compress_file(self, file_path: Path) -> int:
        """Compress file and return compressed size"""
        import gzip
        
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file
            file_path.unlink()
            
            return compressed_path.stat().st_size
            
        except Exception as e:
            self.cleanup_stats['errors'].append(f"Error compressing {file_path}: {e}")
            return 0
    
    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        
        return total_size
    
    async def analyze_storage_usage(self) -> Dict[str, Any]:
        """Analyze current storage usage"""
        print("= Analyzing storage usage...")
        
        analysis = {
            'total_size_mb': 0,
            'breakdown': {},
            'recommendations': []
        }
        
        # Analyze major directories
        major_dirs = [
            'datasets', 'generated', 'output', 'content', 
            'templates', 'cache', 'logs', 'backups'
        ]
        
        for dir_name in major_dirs:
            dir_path = self.data_dir / dir_name
            if dir_path.exists():
                size_mb = self._get_directory_size(dir_path) / (1024 * 1024)
                analysis['breakdown'][dir_name] = {
                    'size_mb': size_mb,
                    'file_count': len(list(dir_path.rglob('*')))
                }
                analysis['total_size_mb'] += size_mb
        
        # Database files
        db_files = list(self.data_dir.rglob('*.db')) + list(self.data_dir.rglob('*.sqlite'))
        if db_files:
            db_size_mb = sum(f.stat().st_size for f in db_files) / (1024 * 1024)
            analysis['breakdown']['databases'] = {
                'size_mb': db_size_mb,
                'file_count': len(db_files)
            }
            analysis['total_size_mb'] += db_size_mb
        
        # Generate recommendations
        if analysis['breakdown'].get('logs', {}).get('size_mb', 0) > 100:
            analysis['recommendations'].append("Consider cleaning up old log files (>100MB)")
        
        if analysis['breakdown'].get('cache', {}).get('size_mb', 0) > 500:
            analysis['recommendations'].append("Cache directory is large (>500MB), consider cleanup")
        
        if analysis['breakdown'].get('generated', {}).get('size_mb', 0) > 1000:
            analysis['recommendations'].append("Generated content is large (>1GB), consider archiving old files")
        
        # Temp files check
        temp_files = list(self.data_dir.rglob('*.tmp')) + list(self.data_dir.rglob('*.temp'))
        if temp_files:
            analysis['recommendations'].append(f"Found {len(temp_files)} temporary files that can be cleaned")
        
        return analysis
    
    async def get_cleanup_preview(self) -> Dict[str, Any]:
        """Preview what would be cleaned without actually cleaning"""
        print("= Generating cleanup preview...")
        
        preview = {
            'temp_files': {'count': 0, 'size_mb': 0},
            'old_logs': {'count': 0, 'size_mb': 0},
            'old_cache': {'count': 0, 'size_mb': 0},
            'old_content': {'count': 0, 'size_mb': 0},
            'empty_dirs': {'count': 0}
        }
        
        # Preview temp files
        temp_patterns = ['*.tmp', '*.temp', '*~', '.DS_Store', 'Thumbs.db']
        cutoff_time = datetime.now() - timedelta(hours=self.config['temp_file_age_hours'])
        
        for pattern in temp_patterns:
            for file_path in self.data_dir.rglob(pattern):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        preview['temp_files']['count'] += 1
                        preview['temp_files']['size_mb'] += file_path.stat().st_size / (1024 * 1024)
        
        # Preview old logs
        logs_dir = self.data_dir / 'logs'
        if logs_dir.exists():
            log_cutoff = datetime.now() - timedelta(days=self.config['log_retention_days'])
            for log_file in logs_dir.rglob('*.log'):
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < log_cutoff:
                    preview['old_logs']['count'] += 1
                    preview['old_logs']['size_mb'] += log_file.stat().st_size / (1024 * 1024)
        
        # Preview old cache
        cache_dirs = [self.data_dir / 'cache', self.data_dir / '.cache']
        cache_cutoff = datetime.now() - timedelta(days=self.config['cache_retention_days'])
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                for cache_file in cache_dir.rglob('*'):
                    if cache_file.is_file():
                        file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                        if file_time < cache_cutoff:
                            preview['old_cache']['count'] += 1
                            preview['old_cache']['size_mb'] += cache_file.stat().st_size / (1024 * 1024)
        
        # Preview old generated content
        content_cutoff = datetime.now() - timedelta(days=self.config['generated_content_retention_days'])
        content_dirs = [self.data_dir / 'generated', self.data_dir / 'output']
        
        for content_dir in content_dirs:
            if content_dir.exists():
                for content_file in content_dir.rglob('*'):
                    if content_file.is_file():
                        file_time = datetime.fromtimestamp(content_file.stat().st_mtime)
                        if file_time < content_cutoff:
                            preview['old_content']['count'] += 1
                            preview['old_content']['size_mb'] += content_file.stat().st_size / (1024 * 1024)
        
        # Count empty directories
        for dirpath, dirnames, filenames in os.walk(self.data_dir, topdown=False):
            dir_path = Path(dirpath)
            if not any(dir_path.iterdir()) and dir_path != self.data_dir:
                preview['empty_dirs']['count'] += 1
        
        return preview


async def main():
    """
    Main cleanup script function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Clean up storage for structured document synthesis'
    )
    parser.add_argument(
        'action',
        choices=['cleanup', 'analyze', 'preview'],
        help='Action to perform'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        help='Custom data directory'
    )
    parser.add_argument(
        '--temp-age-hours',
        type=int,
        help='Age threshold for temp files (hours)'
    )
    parser.add_argument(
        '--log-retention-days',
        type=int,
        help='Log retention period (days)'
    )
    parser.add_argument(
        '--cache-retention-days',
        type=int,
        help='Cache retention period (days)'
    )
    parser.add_argument(
        '--content-retention-days',
        type=int,
        help='Generated content retention period (days)'
    )
    parser.add_argument(
        '--no-vacuum',
        action='store_true',
        help='Skip database vacuum'
    )
    parser.add_argument(
        '--no-compress',
        action='store_true',
        help='Skip log compression'
    )
    
    args = parser.parse_args()
    
    # Build config
    config = CLEANUP_CONFIG.copy()
    if args.temp_age_hours:
        config['temp_file_age_hours'] = args.temp_age_hours
    if args.log_retention_days:
        config['log_retention_days'] = args.log_retention_days
    if args.cache_retention_days:
        config['cache_retention_days'] = args.cache_retention_days
    if args.content_retention_days:
        config['generated_content_retention_days'] = args.content_retention_days
    if args.no_vacuum:
        config['vacuum_databases'] = False
    if args.no_compress:
        config['compress_old_logs'] = False
    
    cleanup = StorageCleanup(data_dir=args.data_dir, config=config)
    
    if args.action == 'cleanup':
        result = await cleanup.run_full_cleanup()
        
        if result['success']:
            print(f"\n Cleanup completed successfully")
            print(f"=Ú Files removed: {result['total_files_removed']}")
            print(f"=Á Space freed: {result['total_space_freed_mb']:.1f} MB")
            
            if result['errors']:
                print(f"\n   Errors encountered:")
                for error in result['errors']:
                    print(f"  - {error}")
        else:
            print(f"\nL Cleanup failed")
            return 1
    
    elif args.action == 'analyze':
        analysis = await cleanup.analyze_storage_usage()
        
        print(f"\n=Ê Storage Analysis:")
        print(f"Total size: {analysis['total_size_mb']:.1f} MB")
        print("\nBreakdown by category:")
        
        for category, info in analysis['breakdown'].items():
            print(f"  {category}: {info['size_mb']:.1f} MB ({info['file_count']} files)")
        
        if analysis['recommendations']:
            print("\n=¡ Recommendations:")
            for rec in analysis['recommendations']:
                print(f"  - {rec}")
    
    elif args.action == 'preview':
        preview = await cleanup.get_cleanup_preview()
        
        print(f"\n= Cleanup Preview:")
        print(f"Temporary files: {preview['temp_files']['count']} files, {preview['temp_files']['size_mb']:.1f} MB")
        print(f"Old logs: {preview['old_logs']['count']} files, {preview['old_logs']['size_mb']:.1f} MB")
        print(f"Old cache: {preview['old_cache']['count']} files, {preview['old_cache']['size_mb']:.1f} MB")
        print(f"Old content: {preview['old_content']['count']} files, {preview['old_content']['size_mb']:.1f} MB")
        print(f"Empty directories: {preview['empty_dirs']['count']} directories")
        
        total_files = (preview['temp_files']['count'] + preview['old_logs']['count'] + 
                      preview['old_cache']['count'] + preview['old_content']['count'])
        total_size = (preview['temp_files']['size_mb'] + preview['old_logs']['size_mb'] + 
                     preview['old_cache']['size_mb'] + preview['old_content']['size_mb'])
        
        print(f"\nTotal: {total_files} files, {total_size:.1f} MB would be freed")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(asyncio.run(main()))