#!/usr/bin/env python3
"""
Dataset download script for external data sources.

Provides automated downloading and management of external datasets
for training and validation purposes.
"""

import asyncio
import json
import os
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
import hashlib
from datetime import datetime

import aiohttp
import aiofiles
from tqdm.asyncio import tqdm

# Configuration
DATASETS_CONFIG = {
    'funsd': {
        'name': 'FUNSD Dataset',
        'url': 'https://guillaumejaume.github.io/FUNSD/dataset.zip',
        'description': 'Form Understanding in Noisy Scanned Documents',
        'size_mb': 15,
        'checksum': 'sha256:a1b2c3d4e5f6...',
        'extract': True,
        'format': 'zip'
    },
    'sroie': {
        'name': 'SROIE Dataset',
        'url': 'https://rrc.cvc.uab.es/downloads/SROIE.zip',
        'description': 'Scanned Receipts OCR and Information Extraction',
        'size_mb': 250,
        'checksum': 'sha256:b2c3d4e5f6g7...',
        'extract': True,
        'format': 'zip'
    },
    'docvqa': {
        'name': 'DocVQA Dataset',
        'url': 'https://www.docvqa.org/datasets/docvqa_train.tar.gz',
        'description': 'Document Visual Question Answering',
        'size_mb': 1200,
        'checksum': 'sha256:c3d4e5f6g7h8...',
        'extract': True,
        'format': 'tar.gz'
    },
    'publaynet': {
        'name': 'PubLayNet Dataset',
        'url': 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz',
        'description': 'Document Layout Analysis Dataset',
        'size_mb': 18000,
        'checksum': 'sha256:d4e5f6g7h8i9...',
        'extract': True,
        'format': 'tar.gz'
    },
    'cord': {
        'name': 'CORD Dataset',
        'url': 'https://www.dropbox.com/s/abc123/CORD.zip',
        'description': 'Consolidated Receipt Dataset',
        'size_mb': 45,
        'checksum': 'sha256:e5f6g7h8i9j0...',
        'extract': True,
        'format': 'zip'
    }
}

DEFAULT_DATA_DIR = Path.home() / '.structured_docs_synth' / 'datasets'
MAX_CONCURRENT_DOWNLOADS = 3
CHUNK_SIZE = 8192


class DatasetDownloader:
    """Automated dataset downloader with progress tracking"""
    
    def __init__(self, data_dir: Optional[Path] = None, 
                 max_concurrent: int = MAX_CONCURRENT_DOWNLOADS):
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.max_concurrent = max_concurrent
        self.session: Optional[aiohttp.ClientSession] = None
        self.download_stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_size_mb': 0
        }
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3600),  # 1 hour timeout
            connector=aiohttp.TCPConnector(limit=self.max_concurrent)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def download_dataset(self, dataset_id: str, 
                             force_redownload: bool = False) -> bool:
        """
        Download single dataset.
        
        Args:
            dataset_id: Dataset identifier
            force_redownload: Force redownload even if exists
        
        Returns:
            True if download successful
        """
        if dataset_id not in DATASETS_CONFIG:
            print(f"L Unknown dataset: {dataset_id}")
            print(f"Available datasets: {', '.join(DATASETS_CONFIG.keys())}")
            return False
        
        config = DATASETS_CONFIG[dataset_id]
        dataset_dir = self.data_dir / dataset_id
        
        # Check if already exists
        if dataset_dir.exists() and not force_redownload:
            print(f" Dataset '{dataset_id}' already exists at {dataset_dir}")
            return True
        
        print(f"=å Downloading {config['name']}...")
        print(f"   Description: {config['description']}")
        print(f"   Size: ~{config['size_mb']} MB")
        print(f"   URL: {config['url']}")
        
        try:
            # Create dataset directory
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Download file
            filename = self._get_filename_from_url(config['url'])
            file_path = dataset_dir / filename
            
            success = await self._download_file(
                config['url'], file_path, config['size_mb']
            )
            
            if not success:
                return False
            
            # Verify checksum if provided
            if 'checksum' in config and config['checksum']:
                if not await self._verify_checksum(file_path, config['checksum']):
                    print(f"L Checksum verification failed for {dataset_id}")
                    return False
                print(f" Checksum verified for {dataset_id}")
            
            # Extract if needed
            if config.get('extract', False):
                success = await self._extract_file(file_path, dataset_dir, config['format'])
                if success:
                    # Remove archive after extraction
                    file_path.unlink()
                    print(f"=Ã  Extracted and cleaned up archive for {dataset_id}")
            
            # Create metadata file
            await self._create_metadata_file(dataset_dir, config)
            
            self.download_stats['successful_downloads'] += 1
            self.download_stats['total_size_mb'] += config['size_mb']
            
            print(f" Successfully downloaded {config['name']}")
            return True
            
        except Exception as e:
            print(f"L Failed to download {dataset_id}: {e}")
            self.download_stats['failed_downloads'] += 1
            return False
        
        finally:
            self.download_stats['total_downloads'] += 1
    
    async def download_multiple(self, dataset_ids: List[str], 
                              force_redownload: bool = False) -> Dict[str, bool]:
        """
        Download multiple datasets concurrently.
        
        Args:
            dataset_ids: List of dataset identifiers
            force_redownload: Force redownload even if exists
        
        Returns:
            Dictionary of dataset_id -> success status
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def download_with_semaphore(dataset_id: str) -> tuple[str, bool]:
            async with semaphore:
                success = await self.download_dataset(dataset_id, force_redownload)
                return dataset_id, success
        
        print(f"=€ Starting download of {len(dataset_ids)} datasets...")
        
        tasks = [download_with_semaphore(dataset_id) for dataset_id in dataset_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        download_results = {}
        for result in results:
            if isinstance(result, Exception):
                print(f"L Download error: {result}")
            else:
                dataset_id, success = result
                download_results[dataset_id] = success
        
        return download_results
    
    async def _download_file(self, url: str, file_path: Path, 
                           expected_size_mb: float) -> bool:
        """
        Download file with progress bar.
        
        Args:
            url: Download URL
            file_path: Local file path
            expected_size_mb: Expected file size in MB
        
        Returns:
            True if download successful
        """
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    print(f"L HTTP {response.status}: {response.reason}")
                    return False
                
                # Get content length
                content_length = response.headers.get('content-length')
                total_size = int(content_length) if content_length else None
                
                # Create progress bar
                progress_bar = tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=file_path.name
                )
                
                # Download with progress
                async with aiofiles.open(file_path, 'wb') as f:
                    downloaded = 0
                    async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                        await f.write(chunk)
                        downloaded += len(chunk)
                        progress_bar.update(len(chunk))
                
                progress_bar.close()
                
                # Verify size
                actual_size_mb = file_path.stat().st_size / (1024 * 1024)
                size_diff = abs(actual_size_mb - expected_size_mb)
                
                if size_diff > expected_size_mb * 0.1:  # Allow 10% variance
                    print(f"   Size mismatch: expected {expected_size_mb}MB, got {actual_size_mb:.1f}MB")
                
                print(f"=Á Downloaded {actual_size_mb:.1f}MB to {file_path}")
                return True
                
        except Exception as e:
            print(f"L Download failed: {e}")
            # Clean up partial file
            if file_path.exists():
                file_path.unlink()
            return False
    
    async def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """
        Verify file checksum.
        
        Args:
            file_path: Path to file
            expected_checksum: Expected checksum (format: algorithm:hash)
        
        Returns:
            True if checksum matches
        """
        try:
            algorithm, expected_hash = expected_checksum.split(':', 1)
            
            hasher = hashlib.new(algorithm)
            
            async with aiofiles.open(file_path, 'rb') as f:
                while chunk := await f.read(CHUNK_SIZE):
                    hasher.update(chunk)
            
            actual_hash = hasher.hexdigest()
            return actual_hash == expected_hash
            
        except Exception as e:
            print(f"   Checksum verification error: {e}")
            return False
    
    async def _extract_file(self, file_path: Path, extract_dir: Path, 
                          format_type: str) -> bool:
        """
        Extract archive file.
        
        Args:
            file_path: Path to archive file
            extract_dir: Directory to extract to
            format_type: Archive format (zip, tar.gz, etc.)
        
        Returns:
            True if extraction successful
        """
        try:
            print(f"=æ Extracting {file_path.name}...")
            
            if format_type == 'zip':
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            
            elif format_type in ['tar.gz', 'tgz']:
                with tarfile.open(file_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_dir)
            
            elif format_type in ['tar.bz2', 'tbz2']:
                with tarfile.open(file_path, 'r:bz2') as tar_ref:
                    tar_ref.extractall(extract_dir)
            
            elif format_type == 'tar':
                with tarfile.open(file_path, 'r') as tar_ref:
                    tar_ref.extractall(extract_dir)
            
            else:
                print(f"L Unsupported archive format: {format_type}")
                return False
            
            return True
            
        except Exception as e:
            print(f"L Extraction failed: {e}")
            return False
    
    async def _create_metadata_file(self, dataset_dir: Path, config: Dict[str, Any]):
        """
        Create metadata file for dataset.
        
        Args:
            dataset_dir: Dataset directory
            config: Dataset configuration
        """
        metadata = {
            'name': config['name'],
            'description': config['description'],
            'url': config['url'],
            'downloaded_at': datetime.now().isoformat(),
            'size_mb': config['size_mb'],
            'format': config.get('format', 'unknown'),
            'checksum': config.get('checksum', ''),
            'version': '1.0'
        }
        
        metadata_file = dataset_dir / 'metadata.json'
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(metadata, indent=2))
    
    def _get_filename_from_url(self, url: str) -> str:
        """
        Extract filename from URL.
        
        Args:
            url: Download URL
        
        Returns:
            Filename
        """
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        return filename if filename else 'download'
    
    def list_available_datasets(self) -> None:
        """
        List all available datasets.
        """
        print("=Ê Available Datasets:")
        print("=" * 50)
        
        for dataset_id, config in DATASETS_CONFIG.items():
            status = " Downloaded" if (self.data_dir / dataset_id).exists() else " Not downloaded"
            print(f"=9 {dataset_id}")
            print(f"   Name: {config['name']}")
            print(f"   Description: {config['description']}")
            print(f"   Size: ~{config['size_mb']} MB")
            print(f"   Status: {status}")
            print()
    
    def get_download_stats(self) -> Dict[str, Any]:
        """
        Get download statistics.
        
        Returns:
            Download statistics
        """
        return {
            **self.download_stats,
            'success_rate': (
                self.download_stats['successful_downloads'] / 
                max(1, self.download_stats['total_downloads']) * 100
            ),
            'data_directory': str(self.data_dir)
        }


async def main():
    """
    Main download script function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download datasets for structured document synthesis'
    )
    parser.add_argument(
        'datasets', 
        nargs='*', 
        help='Dataset IDs to download (default: all)'
    )
    parser.add_argument(
        '--list', 
        action='store_true',
        help='List available datasets'
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force redownload even if dataset exists'
    )
    parser.add_argument(
        '--data-dir', 
        type=Path,
        help='Custom data directory'
    )
    parser.add_argument(
        '--concurrent', 
        type=int, 
        default=MAX_CONCURRENT_DOWNLOADS,
        help='Maximum concurrent downloads'
    )
    
    args = parser.parse_args()
    
    async with DatasetDownloader(
        data_dir=args.data_dir,
        max_concurrent=args.concurrent
    ) as downloader:
        
        if args.list:
            downloader.list_available_datasets()
            return
        
        # Determine datasets to download
        if args.datasets:
            dataset_ids = args.datasets
        else:
            dataset_ids = list(DATASETS_CONFIG.keys())
            print(f"< No specific datasets specified, downloading all {len(dataset_ids)} datasets")
        
        # Download datasets
        results = await downloader.download_multiple(dataset_ids, args.force)
        
        # Print summary
        print("\n" + "="*50)
        print("=Ê Download Summary:")
        print("="*50)
        
        successful = sum(1 for success in results.values() if success)
        failed = len(results) - successful
        
        for dataset_id, success in results.items():
            status = "" if success else "L"
            print(f"{status} {dataset_id}")
        
        print(f"\n=È Results: {successful} successful, {failed} failed")
        
        # Print statistics
        stats = downloader.get_download_stats()
        print(f"=Ê Total size downloaded: {stats['total_size_mb']:.1f} MB")
        print(f"=Â Data directory: {stats['data_directory']}")
        print(f"( Success rate: {stats['success_rate']:.1f}%")


if __name__ == '__main__':
    asyncio.run(main())