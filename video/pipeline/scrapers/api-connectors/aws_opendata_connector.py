#!/usr/bin/env python3
"""
AWS Open Data Connector

This module provides functionality to access and download video datasets from AWS Open Data Registry
for use in the Inferloop Synthetic Data pipeline.
"""

import os
import json
import logging
import time
import requests
import boto3
import yaml
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AWSOpenDataset:
    """Class to store AWS Open Data dataset metadata"""
    name: str
    description: str
    documentation: str
    contact: str
    mla: str  # Machine Learning Application
    update_frequency: str
    tags: List[str]
    resources: Dict[str, Any]
    license: str
    bucket_name: Optional[str] = None
    bucket_region: Optional[str] = None
    video_files: List[str] = field(default_factory=list)
    local_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class AWSOpenDataConnector:
    """Connector for accessing and downloading video datasets from AWS Open Data Registry"""
    
    def __init__(self, aws_access_key: Optional[str] = None, aws_secret_key: Optional[str] = None):
        """
        Initialize the AWS Open Data connector
        
        Args:
            aws_access_key: AWS access key ID (optional, can be set via AWS_ACCESS_KEY_ID env var)
            aws_secret_key: AWS secret access key (optional, can be set via AWS_SECRET_ACCESS_KEY env var)
        """
        # Set AWS credentials if provided
        if aws_access_key and aws_secret_key:
            os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
            os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
        
        # AWS Open Data Registry GitHub repository URL
        self.registry_url = "https://github.com/awslabs/open-data-registry"
        self.registry_raw_url = "https://raw.githubusercontent.com/awslabs/open-data-registry/main/datasets"
        
        # Initialize AWS S3 client
        self.s3 = boto3.client('s3')
        
        self.output_dir = os.environ.get('AWS_OPENDATA_OUTPUT_DIR', 'data/aws_opendata')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def list_datasets(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available datasets in the AWS Open Data Registry
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List of dataset metadata dictionaries
        """
        # Get the list of datasets from the registry
        try:
            # Clone or update the repository to get the latest data
            # For simplicity, we'll use the raw GitHub URL instead
            response = requests.get("https://api.github.com/repos/awslabs/open-data-registry/contents/datasets")
            if response.status_code != 200:
                logger.error(f"Failed to get dataset list: {response.text}")
                return []
            
            datasets = []
            for item in response.json():
                if item['type'] == 'file' and item['name'].endswith('.yaml'):
                    # Get the dataset YAML file
                    yaml_url = item['download_url']
                    yaml_response = requests.get(yaml_url)
                    if yaml_response.status_code == 200:
                        try:
                            dataset = yaml.safe_load(yaml_response.text)
                            # Filter by category if specified
                            if category:
                                if 'tags' in dataset and category.lower() in [tag.lower() for tag in dataset['tags']]:
                                    datasets.append(dataset)
                            else:
                                datasets.append(dataset)
                        except yaml.YAMLError as e:
                            logger.warning(f"Error parsing YAML for {item['name']}: {e}")
            
            logger.info(f"Found {len(datasets)} datasets in the AWS Open Data Registry")
            return datasets
            
        except Exception as e:
            logger.error(f"Error listing AWS Open Data datasets: {e}")
            return []
    
    def search_video_datasets(self, query: str = None, max_results: int = 10) -> List[AWSOpenDataset]:
        """
        Search for video datasets in the AWS Open Data Registry
        
        Args:
            query: Search query string (searches in name, description, and tags)
            max_results: Maximum number of results to return
            
        Returns:
            List of AWSOpenDataset objects
        """
        logger.info(f"Searching AWS Open Data Registry for '{query}' datasets")
        
        # Get all datasets
        all_datasets = self.list_datasets()
        
        # Filter for video-related datasets
        video_keywords = ['video', 'movie', 'film', 'footage', 'camera', 'surveillance', 'stream']
        video_formats = ['mp4', 'avi', 'mov', 'webm', 'mkv']
        
        filtered_datasets = []
        for dataset in all_datasets:
            # Check if dataset is video-related
            is_video_related = False
            
            # Check name and description
            name = dataset.get('name', '').lower()
            description = dataset.get('description', '').lower()
            
            # Check tags
            tags = [tag.lower() for tag in dataset.get('tags', [])]
            
            # Check resources
            resources = dataset.get('resources', [])
            resource_types = []
            for resource in resources:
                if 'type' in resource:
                    resource_types.append(resource['type'].lower())
            
            # Check if video-related
            for keyword in video_keywords:
                if (keyword in name or keyword in description or 
                    keyword in ' '.join(tags) or keyword in ' '.join(resource_types)):
                    is_video_related = True
                    break
            
            # Check if it matches the query
            matches_query = True
            if query:
                query_lower = query.lower()
                matches_query = (query_lower in name or query_lower in description or 
                                any(query_lower in tag for tag in tags))
            
            if is_video_related and matches_query:
                filtered_datasets.append(dataset)
        
        # Convert to AWSOpenDataset objects
        results = []
        for dataset in filtered_datasets[:max_results]:
            # Extract S3 bucket information if available
            bucket_name = None
            bucket_region = None
            resources = dataset.get('resources', [])
            for resource in resources:
                if resource.get('type') == 's3':
                    bucket_name = resource.get('ARN', '').split(':::')[-1]
                    bucket_region = resource.get('Region')
                    break
            
            ds = AWSOpenDataset(
                name=dataset.get('name', ''),
                description=dataset.get('description', ''),
                documentation=dataset.get('documentation', ''),
                contact=dataset.get('contact', ''),
                mla=dataset.get('ManagedBy', ''),
                update_frequency=dataset.get('UpdateFrequency', ''),
                tags=dataset.get('tags', []),
                resources=dataset.get('resources', {}),
                license=dataset.get('License', ''),
                bucket_name=bucket_name,
                bucket_region=bucket_region
            )
            results.append(ds)
        
        logger.info(f"Found {len(results)} video datasets matching '{query}'")
        return results
    
    def get_dataset_details(self, dataset_name: str) -> Optional[AWSOpenDataset]:
        """
        Get detailed information about a specific AWS Open Data dataset
        
        Args:
            dataset_name: Dataset name or identifier
            
        Returns:
            AWSOpenDataset object or None if not found
        """
        try:
            # Find the dataset YAML file
            yaml_url = f"{self.registry_raw_url}/{dataset_name}.yaml"
            response = requests.get(yaml_url)
            
            if response.status_code != 200:
                # Try searching by name
                all_datasets = self.list_datasets()
                dataset_info = None
                for ds in all_datasets:
                    if ds.get('name') == dataset_name:
                        dataset_info = ds
                        break
                
                if not dataset_info:
                    logger.warning(f"Dataset {dataset_name} not found")
                    return None
            else:
                dataset_info = yaml.safe_load(response.text)
            
            # Extract S3 bucket information if available
            bucket_name = None
            bucket_region = None
            resources = dataset_info.get('resources', [])
            for resource in resources:
                if resource.get('type') == 's3':
                    bucket_name = resource.get('ARN', '').split(':::')[-1]
                    bucket_region = resource.get('Region')
                    break
            
            # Create dataset object
            dataset = AWSOpenDataset(
                name=dataset_info.get('name', ''),
                description=dataset_info.get('description', ''),
                documentation=dataset_info.get('documentation', ''),
                contact=dataset_info.get('contact', ''),
                mla=dataset_info.get('ManagedBy', ''),
                update_frequency=dataset_info.get('UpdateFrequency', ''),
                tags=dataset_info.get('tags', []),
                resources=dataset_info.get('resources', {}),
                license=dataset_info.get('License', ''),
                bucket_name=bucket_name,
                bucket_region=bucket_region
            )
            
            # If we have a bucket name, list video files
            if bucket_name:
                try:
                    # List objects in the bucket (limited to 1000 by default)
                    response = self.s3.list_objects_v2(Bucket=bucket_name)
                    
                    # Filter for video files
                    video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
                    video_files = [
                        obj['Key'] for obj in response.get('Contents', [])
                        if any(obj['Key'].lower().endswith(ext) for ext in video_extensions)
                    ]
                    
                    dataset.video_files = video_files
                    dataset.metadata['total_video_files'] = len(video_files)
                    
                except Exception as e:
                    logger.warning(f"Error listing objects in bucket {bucket_name}: {e}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error getting dataset details for {dataset_name}: {e}")
            return None
    
    def download_video_files(self, dataset: AWSOpenDataset, output_path: Optional[str] = None, 
                           max_files: int = 10) -> Optional[str]:
        """
        Download video files from an AWS Open Data dataset
        
        Args:
            dataset: AWSOpenDataset object
            output_path: Directory to save the files (defaults to self.output_dir/dataset_name)
            max_files: Maximum number of files to download
            
        Returns:
            Path to downloaded files or None if download failed
        """
        if not dataset.bucket_name or not dataset.video_files:
            logger.error(f"Dataset {dataset.name} has no S3 bucket or video files")
            return None
        
        try:
            # Create dataset-specific output directory
            if not output_path:
                output_path = os.path.join(self.output_dir, dataset.name.replace(' ', '_'))
            
            os.makedirs(output_path, exist_ok=True)
            
            # Download video files
            logger.info(f"Downloading up to {max_files} video files from {dataset.name}")
            
            downloaded_files = []
            for i, file_key in enumerate(dataset.video_files[:max_files]):
                try:
                    # Create subdirectories if needed
                    file_path = os.path.join(output_path, file_key)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    # Download the file
                    logger.info(f"Downloading {file_key}")
                    self.s3.download_file(dataset.bucket_name, file_key, file_path)
                    downloaded_files.append(file_path)
                    
                except Exception as e:
                    logger.error(f"Error downloading {file_key}: {e}")
            
            dataset.local_path = output_path
            
            # Save metadata
            metadata_path = os.path.join(output_path, f"{dataset.name.replace(' ', '_')}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump({
                    'name': dataset.name,
                    'description': dataset.description,
                    'documentation': dataset.documentation,
                    'license': dataset.license,
                    'bucket_name': dataset.bucket_name,
                    'bucket_region': dataset.bucket_region,
                    'video_files': dataset.video_files,
                    'downloaded_files': [os.path.basename(f) for f in downloaded_files],
                    'download_date': datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"Downloaded {len(downloaded_files)} files to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading files from {dataset.name}: {e}")
            return None
    
    def export_to_csv(self, datasets: List[AWSOpenDataset], output_path: Optional[str] = None) -> str:
        """
        Export dataset metadata to CSV
        
        Args:
            datasets: List of AWSOpenDataset objects
            output_path: Path to save CSV file
            
        Returns:
            Path to the saved CSV file
        """
        if not output_path:
            output_path = os.path.join(self.output_dir, f"aws_opendata_datasets_{int(time.time())}.csv")
        
        # Convert to DataFrame
        data = [
            {
                'name': ds.name,
                'description': ds.description[:100] + '...' if len(ds.description) > 100 else ds.description,
                'documentation': ds.documentation,
                'license': ds.license,
                'tags': ', '.join(ds.tags),
                'bucket_name': ds.bucket_name,
                'bucket_region': ds.bucket_region,
                'video_file_count': len(ds.video_files),
                'local_path': ds.local_path
            } for ds in datasets
        ]
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(datasets)} datasets to {output_path}")
        return output_path

def main():
    """Example usage of the AWSOpenDataConnector"""
    connector = AWSOpenDataConnector()
    
    # Example: Search for video datasets
    datasets = connector.search_video_datasets(
        query="video",
        max_results=5
    )
    
    if datasets:
        print(f"Found {len(datasets)} video datasets:")
        for ds in datasets:
            print(f"- {ds.name}")
            print(f"  Description: {ds.description[:100]}...")
            print(f"  Bucket: {ds.bucket_name}")
            print(f"  Tags: {', '.join(ds.tags)}")
            print()
        
        # Example: Get details for the first dataset
        first_dataset = datasets[0]
        detailed_dataset = connector.get_dataset_details(first_dataset.name)
        
        if detailed_dataset and detailed_dataset.video_files:
            print(f"\nDetails for {detailed_dataset.name}:")
            print(f"Video files: {len(detailed_dataset.video_files)}")
            
            # Example: Download video files
            download_path = connector.download_video_files(
                detailed_dataset,
                max_files=2  # Limit to 2 files for example
            )
            
            if download_path:
                print(f"Downloaded files to {download_path}")
        
        # Example: Export metadata to CSV
        csv_path = connector.export_to_csv(datasets)
        print(f"Exported metadata to {csv_path}")
    else:
        print("No datasets found or API error occurred.")

if __name__ == "__main__":
    main()
