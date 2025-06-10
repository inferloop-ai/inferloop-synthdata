#!/usr/bin/env python3
"""
YouTube Video Scraper

This module provides functionality to scrape video data from YouTube
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
import googleapiclient.discovery
from googleapiclient.errors import HttpError
from pytube import YouTube
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class YouTubeVideo:
    """Class to store YouTube video metadata"""
    video_id: str
    title: str
    description: str
    channel_id: str
    channel_title: str
    published_at: datetime
    tags: List[str]
    category_id: str
    duration: str  # ISO 8601 duration
    view_count: int
    like_count: int
    comment_count: int
    thumbnail_url: str
    resolution: Optional[Tuple[int, int]] = None
    fps: Optional[float] = None
    audio_quality: Optional[str] = None
    download_url: Optional[str] = None
    local_path: Optional[str] = None

class YouTubeScraper:
    """Scraper for YouTube videos using YouTube Data API and pytube"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the YouTube scraper
        
        Args:
            api_key: YouTube Data API key (optional, can be set via YOUTUBE_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get('YOUTUBE_API_KEY')
        if not self.api_key:
            logger.warning("No YouTube API key provided. Some functionality will be limited.")
            self.youtube = None
        else:
            self.youtube = googleapiclient.discovery.build(
                'youtube', 'v3', developerKey=self.api_key, cache_discovery=False
            )
        
        self.output_dir = os.environ.get('YOUTUBE_OUTPUT_DIR', 'data/youtube')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def search_videos(self, query: str, max_results: int = 10, 
                      published_after: Optional[datetime] = None,
                      category_id: Optional[str] = None) -> List[YouTubeVideo]:
        """
        Search for YouTube videos matching the query
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            published_after: Only include videos published after this date
            category_id: Filter by YouTube category ID
            
        Returns:
            List of YouTubeVideo objects
        """
        if not self.youtube:
            raise ValueError("YouTube API key is required for search functionality")
        
        search_params = {
            'q': query,
            'type': 'video',
            'part': 'id,snippet',
            'maxResults': min(max_results, 50)  # API limit is 50 per request
        }
        
        if published_after:
            search_params['publishedAfter'] = published_after.isoformat() + 'Z'
        
        if category_id:
            search_params['videoCategoryId'] = category_id
        
        logger.info(f"Searching YouTube for '{query}'")
        
        try:
            search_response = self.youtube.search().list(**search_params).execute()
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            
            # Get detailed video information
            videos_response = self.youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=','.join(video_ids)
            ).execute()
            
            results = []
            for item in videos_response['items']:
                snippet = item['snippet']
                statistics = item.get('statistics', {})
                content_details = item.get('contentDetails', {})
                
                video = YouTubeVideo(
                    video_id=item['id'],
                    title=snippet['title'],
                    description=snippet['description'],
                    channel_id=snippet['channelId'],
                    channel_title=snippet['channelTitle'],
                    published_at=datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00')),
                    tags=snippet.get('tags', []),
                    category_id=snippet.get('categoryId', ''),
                    duration=content_details.get('duration', ''),
                    view_count=int(statistics.get('viewCount', 0)),
                    like_count=int(statistics.get('likeCount', 0)),
                    comment_count=int(statistics.get('commentCount', 0)),
                    thumbnail_url=snippet.get('thumbnails', {}).get('high', {}).get('url', '')
                )
                results.append(video)
            
            logger.info(f"Found {len(results)} videos matching '{query}'")
            return results
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            return []
    
    def get_video_details(self, video_id: str) -> Optional[YouTubeVideo]:
        """
        Get detailed information about a specific YouTube video
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            YouTubeVideo object or None if not found
        """
        if not self.youtube:
            # Fall back to pytube if no API key
            return self._get_video_details_pytube(video_id)
        
        try:
            video_response = self.youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=video_id
            ).execute()
            
            if not video_response['items']:
                logger.warning(f"No video found with ID {video_id}")
                return None
            
            item = video_response['items'][0]
            snippet = item['snippet']
            statistics = item.get('statistics', {})
            content_details = item.get('contentDetails', {})
            
            video = YouTubeVideo(
                video_id=item['id'],
                title=snippet['title'],
                description=snippet['description'],
                channel_id=snippet['channelId'],
                channel_title=snippet['channelTitle'],
                published_at=datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00')),
                tags=snippet.get('tags', []),
                category_id=snippet.get('categoryId', ''),
                duration=content_details.get('duration', ''),
                view_count=int(statistics.get('viewCount', 0)),
                like_count=int(statistics.get('likeCount', 0)),
                comment_count=int(statistics.get('commentCount', 0)),
                thumbnail_url=snippet.get('thumbnails', {}).get('high', {}).get('url', '')
            )
            
            # Enhance with pytube data
            self._enhance_with_pytube(video)
            
            return video
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            return None
    
    def _get_video_details_pytube(self, video_id: str) -> Optional[YouTubeVideo]:
        """
        Get video details using pytube (no API key required)
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            YouTubeVideo object or None if error
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        try:
            yt = YouTube(url)
            
            video = YouTubeVideo(
                video_id=video_id,
                title=yt.title,
                description=yt.description,
                channel_id=yt.channel_id,
                channel_title=yt.author,
                published_at=datetime.now(),  # pytube doesn't provide publish date
                tags=yt.keywords,
                category_id='',
                duration=str(yt.length),
                view_count=yt.views,
                like_count=0,  # Not available in pytube
                comment_count=0,  # Not available in pytube
                thumbnail_url=yt.thumbnail_url
            )
            
            # Get resolution info from streams
            streams = yt.streams.filter(progressive=True, file_extension='mp4')
            if streams:
                best_stream = streams.get_highest_resolution()
                if best_stream:
                    video.resolution = (best_stream.resolution.split('p')[0], None)
                    video.fps = best_stream.fps
                    video.download_url = best_stream.url
            
            return video
            
        except Exception as e:
            logger.error(f"Error getting video details with pytube: {e}")
            return None
    
    def _enhance_with_pytube(self, video: YouTubeVideo) -> None:
        """
        Enhance video object with additional data from pytube
        
        Args:
            video: YouTubeVideo object to enhance
        """
        url = f"https://www.youtube.com/watch?v={video.video_id}"
        
        try:
            yt = YouTube(url)
            streams = yt.streams.filter(progressive=True, file_extension='mp4')
            if streams:
                best_stream = streams.get_highest_resolution()
                if best_stream:
                    video.resolution = (int(best_stream.resolution.split('p')[0]), None)
                    video.fps = best_stream.fps
                    video.download_url = best_stream.url
        except Exception as e:
            logger.warning(f"Could not enhance video with pytube: {e}")
    
    def download_video(self, video: YouTubeVideo, output_path: Optional[str] = None) -> Optional[str]:
        """
        Download a YouTube video
        
        Args:
            video: YouTubeVideo object
            output_path: Directory to save the video (defaults to self.output_dir)
            
        Returns:
            Path to downloaded file or None if download failed
        """
        if not output_path:
            output_path = self.output_dir
            
        os.makedirs(output_path, exist_ok=True)
        
        url = f"https://www.youtube.com/watch?v={video.video_id}"
        filename = f"{video.video_id}_{video.title[:50].replace('/', '_')}.mp4"
        filepath = os.path.join(output_path, filename)
        
        try:
            logger.info(f"Downloading video: {video.title}")
            yt = YouTube(url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').get_highest_resolution()
            
            if not stream:
                logger.error(f"No suitable stream found for {video.video_id}")
                return None
                
            downloaded_path = stream.download(output_path=output_path, filename=filename)
            video.local_path = downloaded_path
            
            # Save metadata alongside video
            metadata_path = os.path.join(output_path, f"{video.video_id}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump({
                    'video_id': video.video_id,
                    'title': video.title,
                    'channel': video.channel_title,
                    'published_at': video.published_at.isoformat(),
                    'duration': video.duration,
                    'view_count': video.view_count,
                    'resolution': video.resolution,
                    'fps': video.fps,
                    'download_date': datetime.now().isoformat()
                }, f, indent=2)
                
            logger.info(f"Video downloaded to {downloaded_path}")
            return downloaded_path
            
        except Exception as e:
            logger.error(f"Error downloading video {video.video_id}: {e}")
            return None
    
    def export_to_csv(self, videos: List[YouTubeVideo], output_path: Optional[str] = None) -> str:
        """
        Export video metadata to CSV
        
        Args:
            videos: List of YouTubeVideo objects
            output_path: Path to save CSV file
            
        Returns:
            Path to the saved CSV file
        """
        if not output_path:
            output_path = os.path.join(self.output_dir, f"youtube_videos_{int(time.time())}.csv")
        
        # Convert to DataFrame
        data = [{
            'video_id': v.video_id,
            'title': v.title,
            'channel': v.channel_title,
            'published_at': v.published_at,
            'duration': v.duration,
            'view_count': v.view_count,
            'like_count': v.like_count,
            'comment_count': v.comment_count,
            'resolution': str(v.resolution) if v.resolution else None,
            'fps': v.fps,
            'local_path': v.local_path
        } for v in videos]
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(videos)} videos to {output_path}")
        return output_path


def main():
    """Example usage of the YouTubeScraper"""
    # Get API key from environment variable
    api_key = os.environ.get('YOUTUBE_API_KEY')
    
    if not api_key:
        print("Warning: No YouTube API key found. Set the YOUTUBE_API_KEY environment variable.")
        print("Limited functionality will be available.")
    
    scraper = YouTubeScraper(api_key=api_key)
    
    # Example: Search for videos
    videos = scraper.search_videos(
        query="synthetic data generation", 
        max_results=5
    )
    
    if videos:
        print(f"Found {len(videos)} videos:")
        for video in videos:
            print(f"- {video.title} (ID: {video.video_id}, Views: {video.view_count})")
        
        # Example: Download the first video
        first_video = videos[0]
        download_path = scraper.download_video(first_video)
        
        if download_path:
            print(f"Downloaded video to {download_path}")
        
        # Example: Export metadata to CSV
        csv_path = scraper.export_to_csv(videos)
        print(f"Exported metadata to {csv_path}")
    else:
        print("No videos found or API error occurred.")


if __name__ == "__main__":
    main()
