#!/usr/bin/env python3
"""
Vimeo Video Scraper

This module provides functionality to scrape video data from Vimeo
for use in the Inferloop Synthetic Data pipeline.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests

# Third-party libraries
import vimeo
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VimeoVideo:
    """Class to store Vimeo video metadata"""
    video_id: str
    title: str
    description: str
    user_id: str
    user_name: str
    created_time: datetime
    tags: List[str]
    duration: int  # in seconds
    width: int
    height: int
    view_count: int
    like_count: int
    comment_count: int
    thumbnail_url: str
    download_url: Optional[str] = None
    local_path: Optional[str] = None
    fps: Optional[float] = None
    privacy: Optional[Dict[str, Any]] = None

class VimeoScraper:
    """Scraper for Vimeo videos using Vimeo API"""
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None, access_token: Optional[str] = None):
        """
        Initialize the Vimeo scraper
        
        Args:
            client_id: Vimeo API client ID (optional, can be set via VIMEO_CLIENT_ID env var)
            client_secret: Vimeo API client secret (optional, can be set via VIMEO_CLIENT_SECRET env var)
            access_token: Vimeo API access token (optional, can be set via VIMEO_ACCESS_TOKEN env var)
        """
        self.client_id = client_id or os.environ.get('VIMEO_CLIENT_ID')
        self.client_secret = client_secret or os.environ.get('VIMEO_CLIENT_SECRET')
        self.access_token = access_token or os.environ.get('VIMEO_ACCESS_TOKEN')
        
        if not all([self.client_id, self.client_secret, self.access_token]):
            logger.warning("Missing Vimeo API credentials. Some functionality will be limited.")
            self.client = None
        else:
            self.client = vimeo.VimeoClient(
                token=self.access_token,
                key=self.client_id,
                secret=self.client_secret
            )
        
        self.output_dir = os.environ.get('VIMEO_OUTPUT_DIR', 'data/vimeo')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def search_videos(self, query: str, max_results: int = 10, 
                      sort: str = 'relevant', direction: str = 'desc',
                      categories: Optional[List[str]] = None) -> List[VimeoVideo]:
        """
        Search for Vimeo videos matching the query
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sort: Sort order ('relevant', 'date', 'plays', 'likes', 'comments', 'duration')
            direction: Sort direction ('asc' or 'desc')
            categories: List of category URIs to filter by
            
        Returns:
            List of VimeoVideo objects
        """
        if not self.client:
            raise ValueError("Vimeo API credentials are required for search functionality")
        
        params = {
            'query': query,
            'per_page': min(max_results, 100),  # API limit is 100 per request
            'sort': sort,
            'direction': direction
        }
        
        if categories:
            params['category'] = categories
        
        logger.info(f"Searching Vimeo for '{query}'")
        
        try:
            response = self.client.get('/videos', params=params)
            
            if response.status_code != 200:
                logger.error(f"Vimeo API error: {response.text}")
                return []
            
            data = response.json()
            results = []
            
            for item in data['data']:
                # Extract video metadata
                video = self._parse_video_data(item)
                results.append(video)
            
            logger.info(f"Found {len(results)} videos matching '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Vimeo API error: {e}")
            return []
    
    def get_video_details(self, video_id: str) -> Optional[VimeoVideo]:
        """
        Get detailed information about a specific Vimeo video
        
        Args:
            video_id: Vimeo video ID
            
        Returns:
            VimeoVideo object or None if not found
        """
        if not self.client:
            raise ValueError("Vimeo API credentials are required")
        
        try:
            response = self.client.get(f'/videos/{video_id}')
            
            if response.status_code != 200:
                logger.warning(f"No video found with ID {video_id}")
                return None
            
            data = response.json()
            video = self._parse_video_data(data)
            
            # Get download links if available
            if 'download' in data and data['download']:
                download_links = data['download']
                if download_links:
                    # Get the highest quality download link
                    best_download = max(download_links, key=lambda x: x.get('width', 0) if x.get('width') else 0)
                    video.download_url = best_download.get('link')
            
            return video
            
        except Exception as e:
            logger.error(f"Error getting video details: {e}")
            return None
    
    def _parse_video_data(self, data: Dict[str, Any]) -> VimeoVideo:
        """
        Parse Vimeo API response into VimeoVideo object
        
        Args:
            data: Video data from Vimeo API
            
        Returns:
            VimeoVideo object
        """
        # Extract video ID from URI (format: "/videos/123456789")
        video_id = data['uri'].split('/')[-1]
        
        # Parse created time
        created_time = datetime.fromisoformat(data['created_time'].replace('Z', '+00:00'))
        
        # Extract tags
        tags = [tag['name'] for tag in data.get('tags', [])]
        
        # Get thumbnail URL (use the largest available)
        pictures = data.get('pictures', {}).get('sizes', [])
        thumbnail_url = pictures[-1]['link'] if pictures else ""
        
        # Extract metadata
        video = VimeoVideo(
            video_id=video_id,
            title=data['name'],
            description=data.get('description', ''),
            user_id=data['user']['uri'].split('/')[-1],
            user_name=data['user']['name'],
            created_time=created_time,
            tags=tags,
            duration=data['duration'],
            width=data.get('width', 0),
            height=data.get('height', 0),
            view_count=data.get('stats', {}).get('plays', 0),
            like_count=data.get('metadata', {}).get('connections', {}).get('likes', {}).get('total', 0),
            comment_count=data.get('metadata', {}).get('connections', {}).get('comments', {}).get('total', 0),
            thumbnail_url=thumbnail_url,
            privacy=data.get('privacy', {})
        )
        
        return video
    
    def download_video(self, video: VimeoVideo, output_path: Optional[str] = None) -> Optional[str]:
        """
        Download a Vimeo video
        
        Args:
            video: VimeoVideo object
            output_path: Directory to save the video (defaults to self.output_dir)
            
        Returns:
            Path to downloaded file or None if download failed
        """
        if not video.download_url:
            logger.error(f"No download URL available for video {video.video_id}")
            return None
            
        if not output_path:
            output_path = self.output_dir
            
        os.makedirs(output_path, exist_ok=True)
        
        filename = f"vimeo_{video.video_id}_{video.title[:50].replace('/', '_')}.mp4"
        filepath = os.path.join(output_path, filename)
        
        try:
            logger.info(f"Downloading video: {video.title}")
            
            # Download the file
            response = requests.get(video.download_url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            video.local_path = filepath
            
            # Save metadata alongside video
            metadata_path = os.path.join(output_path, f"vimeo_{video.video_id}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump({
                    'video_id': video.video_id,
                    'title': video.title,
                    'user': video.user_name,
                    'created_time': video.created_time.isoformat(),
                    'duration': video.duration,
                    'view_count': video.view_count,
                    'resolution': f"{video.width}x{video.height}",
                    'download_date': datetime.now().isoformat()
                }, f, indent=2)
                
            logger.info(f"Video downloaded to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error downloading video {video.video_id}: {e}")
            return None
    
    def export_to_csv(self, videos: List[VimeoVideo], output_path: Optional[str] = None) -> str:
        """
        Export video metadata to CSV
        
        Args:
            videos: List of VimeoVideo objects
            output_path: Path to save CSV file
            
        Returns:
            Path to the saved CSV file
        """
        if not output_path:
            output_path = os.path.join(self.output_dir, f"vimeo_videos_{int(time.time())}.csv")
        
        # Convert to DataFrame
        data = [
            {
                'video_id': v.video_id,
                'title': v.title,
                'user_name': v.user_name,
                'created_time': v.created_time,
                'duration': v.duration,
                'view_count': v.view_count,
                'like_count': v.like_count,
                'comment_count': v.comment_count,
                'resolution': f"{v.width}x{v.height}" if v.width and v.height else None,
                'local_path': v.local_path
            } for v in videos
        ]
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(videos)} videos to {output_path}")
        return output_path

def main():
    """Example usage of the VimeoScraper"""
    # Get API credentials from environment variables
    client_id = os.environ.get('VIMEO_CLIENT_ID')
    client_secret = os.environ.get('VIMEO_CLIENT_SECRET')
    access_token = os.environ.get('VIMEO_ACCESS_TOKEN')
    
    if not all([client_id, client_secret, access_token]):
        print("Warning: Missing Vimeo API credentials. Set the VIMEO_CLIENT_ID, VIMEO_CLIENT_SECRET, and VIMEO_ACCESS_TOKEN environment variables.")
        return
    
    scraper = VimeoScraper(client_id=client_id, client_secret=client_secret, access_token=access_token)
    
    # Example: Search for videos
    videos = scraper.search_videos(
        query="synthetic data generation",
        max_results=5,
        sort="relevant"
    )
    
    if videos:
        print(f"Found {len(videos)} videos:")
        for video in videos:
            print(f"- {video.title} (ID: {video.video_id}, Views: {video.view_count})")
        
        # Example: Get details for the first video
        first_video = videos[0]
        detailed_video = scraper.get_video_details(first_video.video_id)
        
        if detailed_video and detailed_video.download_url:
            # Example: Download the video
            download_path = scraper.download_video(detailed_video)
            
            if download_path:
                print(f"Downloaded video to {download_path}")
        
        # Example: Export metadata to CSV
        csv_path = scraper.export_to_csv(videos)
        print(f"Exported metadata to {csv_path}")
    else:
        print("No videos found or API error occurred.")

if __name__ == "__main__":
    main()
