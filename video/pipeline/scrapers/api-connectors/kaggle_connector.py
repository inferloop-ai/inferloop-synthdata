#!/usr/bin/env python3
"""
Kaggle API Connector

This module provides functionality to access and download video datasets from Kaggle
for use in the Inferloop Synthetic Data pipeline.
"""

import os
import json
import logging
import time
import zipfile
import shutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

# Third-party libraries
from kaggle.api.kaggle_api_extended import KaggleApi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class KaggleDataset:
    """Class to store Kaggle dataset metadata"""
    ref: str  # Format: 'username/dataset-name'
    title: str
    subtitle: str
    description: str
    url: str
    download_count: int
    view_count: int
    vote_count: int
    size: int  # in bytes
    last_updated: datetime
    tags: List[str]
    license_name: str
    files: List[str] = field(default_factory=list)
    video_files: List[str] = field(default_factory=list)
    local_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class KaggleConnector:
    """Connector for accessing and downloading video datasets from Kaggle"""
    
    def __init__(self, username: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize the Kaggle connector
        
        Args:
            username: Kaggle username (optional, can be set via KAGGLE_USERNAME env var)
            key: Kaggle API key (optional, can be set via KAGGLE_KEY env var)
        """
        # Kaggle API looks for credentials in ~/.kaggle/kaggle.json or env vars
        if username and key:
            os.environ['KAGGLE_USERNAME'] = username
            os.environ['KAGGLE_KEY'] = key
        
        self.api = KaggleApi()
        try:
            self.api.authenticate()
            logger.info("Successfully authenticated with Kaggle API")
        except Exception as e:
            logger.error(f"Failed to authenticate with Kaggle API: {e}")
            raise
        
        self.output_dir = os.environ.get('KAGGLE_OUTPUT_DIR', 'data/kaggle')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def search_video_datasets(self, query: str, max_results: int = 10) -> List[KaggleDataset]:
        """
        Search for video datasets on Kaggle
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of KaggleDataset objects
        """
        logger.info(f"Searching Kaggle for '{query}' datasets")
        
        # Search for datasets with the query and filter by file type
        search_query = f"{query} file:mp4 OR file:avi OR file:mov OR file:webm"
        datasets = self.api.dataset_list(search=search_query, max_size=None, file_type=None)
        
        results = []
        for dataset in datasets[:max_results]:
            # Convert Kaggle dataset object to our KaggleDataset class
            ds = KaggleDataset(
                ref=dataset.ref,
                title=dataset.title,
                subtitle=dataset.subtitle,
                description=dataset.description,
                url=dataset.url,
                download_count=dataset.downloadCount,
                view_count=dataset.viewCount,
                vote_count=dataset.voteCount,
                size=dataset.size,
                last_updated=datetime.strptime(dataset.lastUpdated, '%Y-%m-%dT%H:%M:%S.%fZ'),
                tags=dataset.tags,
                license_name=dataset.licenseName
            )
            results.append(ds)
        
        logger.info(f"Found {len(results)} video datasets matching '{query}'")
        return results
    
    def get_dataset_details(self, dataset_ref: str) -> Optional[KaggleDataset]:
        """
        Get detailed information about a specific Kaggle dataset
        
        Args:
            dataset_ref: Dataset reference in format 'username/dataset-name'
            
        Returns:
            KaggleDataset object or None if not found
        """
        try:
            # Get dataset metadata
            dataset = self.api.dataset_view(dataset_ref)
            
            # Get file list
            files = self.api.dataset_list_files(dataset_ref).files
            
            # Filter for video files
            video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
            video_files = [
                f for f in files 
                if any(f.name.lower().endswith(ext) for ext in video_extensions)
            ]
            
            ds = KaggleDataset(
                ref=dataset.ref,
                title=dataset.title,
                subtitle=dataset.subtitle,
                description=dataset.description,
                url=dataset.url,
                download_count=dataset.downloadCount,
                view_count=dataset.viewCount,
                vote_count=dataset.voteCount,
                size=dataset.size,
                last_updated=datetime.strptime(dataset.lastUpdated, '%Y-%m-%dT%H:%M:%S.%fZ'),
                tags=dataset.tags,
                license_name=dataset.licenseName,
                files=[f.name for f in files],
                video_files=[f.name for f in video_files],
                metadata={
                    'owner_name': dataset.ownerName,
                    'total_files': len(files),
                    'total_video_files': len(video_files)
                }
            )
            
            return ds
            
        except Exception as e:
            logger.error(f"Error getting dataset details for {dataset_ref}: {e}")
            return None
    
    def download_dataset(self, dataset_ref: str, output_path: Optional[str] = None, 
                         unzip: bool = True, video_only: bool = True) -> Optional[str]:
        """
        Download a Kaggle dataset
        
        Args:
            dataset_ref: Dataset reference in format 'username/dataset-name'
            output_path: Directory to save the dataset (defaults to self.output_dir/dataset_name)
            unzip: Whether to unzip the downloaded dataset
            video_only: Whether to extract only video files
            
        Returns:
            Path to downloaded dataset or None if download failed
        """
        try:
            # Create dataset-specific output directory
            dataset_name = dataset_ref.split('/')[-1]
            if not output_path:
                output_path = os.path.join(self.output_dir, dataset_name)
            
            os.makedirs(output_path, exist_ok=True)
            
            # Download the dataset
            logger.info(f"Downloading dataset: {dataset_ref}")
            self.api.dataset_download_files(dataset_ref, path=output_path, unzip=unzip)
            
            # If we downloaded as zip and need to unzip
            zip_path = os.path.join(output_path, f"{dataset_name}.zip")
            if not unzip and os.path.exists(zip_path):
                # We're done, return the zip path
                return zip_path
            
            # If we want to extract only video files
            if video_only and unzip:
                # Get all files in the directory
                all_files = []
                for root, _, files in os.walk(output_path):
                    for file in files:
                        all_files.append(os.path.join(root, file))
                
                # Filter for video files
                video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
                video_files = [
                    f for f in all_files 
                    if any(f.lower().endswith(ext) for ext in video_extensions)
                ]
                
                # Create videos subdirectory
                videos_dir = os.path.join(output_path, 'videos')
                os.makedirs(videos_dir, exist_ok=True)
                
                # Move video files to videos subdirectory
                for video_file in video_files:
                    filename = os.path.basename(video_file)
                    dest_path = os.path.join(videos_dir, filename)
                    shutil.copy2(video_file, dest_path)
                
                logger.info(f"Extracted {len(video_files)} video files to {videos_dir}")
            
            # Get dataset details and save metadata
            dataset = self.get_dataset_details(dataset_ref)
            if dataset:
                dataset.local_path = output_path
                
                # Save metadata
                metadata_path = os.path.join(output_path, f"{dataset_name}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump({
                        'ref': dataset.ref,
                        'title': dataset.title,
                        'description': dataset.description,
                        'url': dataset.url,
                        'download_count': dataset.download_count,
                        'size': dataset.size,
                        'last_updated': dataset.last_updated.isoformat(),
                        'license_name': dataset.license_name,
                        'video_files': dataset.video_files,
                        'download_date': datetime.now().isoformat()
                    }, f, indent=2)
            
            logger.info(f"Dataset downloaded to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_ref}: {e}")
            return None
    
    def export_to_csv(self, datasets: List[KaggleDataset], output_path: Optional[str] = None) -> str:
        """
        Export dataset metadata to CSV
        
        Args:
            datasets: List of KaggleDataset objects
            output_path: Path to save CSV file
            
        Returns:
            Path to the saved CSV file
        """
        if not output_path:
            output_path = os.path.join(self.output_dir, f"kaggle_datasets_{int(time.time())}.csv")
        
        # Convert to DataFrame
        data = [
            {
                'ref': ds.ref,
                'title': ds.title,
                'subtitle': ds.subtitle,
                'url': ds.url,
                'download_count': ds.download_count,
                'view_count': ds.view_count,
                'vote_count': ds.vote_count,
                'size_bytes': ds.size,
                'last_updated': ds.last_updated,
                'license': ds.license_name,
                'video_file_count': len(ds.video_files),
                'local_path': ds.local_path
            } for ds in datasets
        ]
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(datasets)} datasets to {output_path}")
        return output_path

def main():
    """Example usage of the KaggleConnector"""
    # Check if Kaggle credentials are available
    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')
    
    if not username or not key:
        print("Warning: Kaggle credentials not found. Set the KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        print("You can find your API credentials at https://www.kaggle.com/account")
        return
    
    connector = KaggleConnector()
    
    # Example: Search for video datasets
    datasets = connector.search_video_datasets(
        query="video dataset",
        max_results=5
    )
    
    if datasets:
        print(f"Found {len(datasets)} video datasets:")
        for ds in datasets:
            print(f"- {ds.title} (Ref: {ds.ref}, Size: {ds.size/1024/1024:.2f} MB)")
        
        # Example: Get details for the first dataset
        first_dataset = datasets[0]
        detailed_dataset = connector.get_dataset_details(first_dataset.ref)
        
        if detailed_dataset:
            print(f"\nDetails for {detailed_dataset.title}:")
            print(f"Description: {detailed_dataset.description[:100]}...")
            print(f"Video files: {len(detailed_dataset.video_files)}")
            
            # Example: Download the dataset
            download_path = connector.download_dataset(
                detailed_dataset.ref,
                video_only=True
            )
            
            if download_path:
                print(f"Downloaded dataset to {download_path}")
        
        # Example: Export metadata to CSV
        csv_path = connector.export_to_csv(datasets)
        print(f"Exported metadata to {csv_path}")
    else:
        print("No datasets found or API error occurred.")

if __name__ == "__main__":
    main()
