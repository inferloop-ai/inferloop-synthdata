#!/usr/bin/env python3
"""
Upload Handler

This module provides functionality to process uploaded video files
for use in the Inferloop Synthetic Data pipeline.
"""

import os
import json
import logging
import time
import shutil
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import cv2
import numpy as np
import ffmpeg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VideoFile:
    """Class to store video file metadata"""
    file_path: str
    file_name: str
    file_size: int
    file_hash: str
    upload_time: datetime
    processed: bool = False
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    duration: Optional[float] = None
    format: Optional[str] = None
    bitrate: Optional[int] = None
    codec: Optional[str] = None
    audio_codec: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    error: Optional[str] = None

class UploadHandler:
    """Handler for processing uploaded video files"""
    
    def __init__(self, upload_dir: str = 'uploads', processed_dir: str = 'processed'):
        """
        Initialize the upload handler
        
        Args:
            upload_dir: Directory for incoming uploads
            processed_dir: Directory for processed files
        """
        self.upload_dir = upload_dir
        self.processed_dir = processed_dir
        
        # Create directories if they don't exist
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Create subdirectories for different processing stages
        self.thumbnail_dir = os.path.join(processed_dir, 'thumbnails')
        self.metadata_dir = os.path.join(processed_dir, 'metadata')
        self.error_dir = os.path.join(processed_dir, 'errors')
        
        os.makedirs(self.thumbnail_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)
    
    def process_upload(self, file_path: str, move_file: bool = False) -> VideoFile:
        """
        Process an uploaded video file
        
        Args:
            file_path: Path to the uploaded file
            move_file: Whether to move the file to the processed directory
            
        Returns:
            VideoFile object with extracted metadata
        """
        logger.info(f"Processing uploaded file: {file_path}")
        
        try:
            # Basic file information
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            file_hash = self._calculate_file_hash(file_path)
            upload_time = datetime.fromtimestamp(os.path.getctime(file_path))
            
            # Create VideoFile object
            video_file = VideoFile(
                file_path=file_path,
                file_name=file_name,
                file_size=file_size,
                file_hash=file_hash,
                upload_time=upload_time
            )
            
            # Extract video metadata
            self._extract_video_metadata(video_file)
            
            # Generate thumbnail
            self._generate_thumbnail(video_file)
            
            # Move file if requested
            if move_file:
                new_path = os.path.join(self.processed_dir, file_name)
                shutil.move(file_path, new_path)
                video_file.file_path = new_path
                video_file.processed_path = new_path
            
            # Save metadata
            self._save_metadata(video_file)
            
            video_file.processed = True
            logger.info(f"Successfully processed {file_name}")
            
            return video_file
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            
            # Create error record
            video_file = VideoFile(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                file_hash="",
                upload_time=datetime.now(),
                processed=False,
                error=str(e)
            )
            
            # Move to error directory if requested
            if move_file and os.path.exists(file_path):
                error_path = os.path.join(self.error_dir, os.path.basename(file_path))
                shutil.move(file_path, error_path)
                video_file.file_path = error_path
            
            return video_file
    
    def process_directory(self, directory: str = None, recursive: bool = False) -> List[VideoFile]:
        """
        Process all video files in a directory
        
        Args:
            directory: Directory to process (defaults to self.upload_dir)
            recursive: Whether to process subdirectories
            
        Returns:
            List of VideoFile objects
        """
        if directory is None:
            directory = self.upload_dir
        
        logger.info(f"Processing directory: {directory}")
        
        video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
        results = []
        
        # Find all video files
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    file_path = os.path.join(root, file)
                    video_file = self.process_upload(file_path, move_file=True)
                    results.append(video_file)
            
            if not recursive:
                break
        
        logger.info(f"Processed {len(results)} files in {directory}")
        return results
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate MD5 hash of a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            MD5 hash as hexadecimal string
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _extract_video_metadata(self, video_file: VideoFile) -> None:
        """
        Extract metadata from a video file
        
        Args:
            video_file: VideoFile object to update
        """
        try:
            # Use ffmpeg to probe the file
            probe = ffmpeg.probe(video_file.file_path)
            
            # Get video stream info
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            # Extract basic video properties
            video_file.width = int(video_stream['width'])
            video_file.height = int(video_stream['height'])
            video_file.fps = eval(video_stream['r_frame_rate'])
            video_file.duration = float(video_stream.get('duration', 0))
            video_file.codec = video_stream.get('codec_name', '')
            video_file.format = probe['format']['format_name']
            video_file.bitrate = int(probe['format'].get('bit_rate', 0))
            
            # Get audio stream info if available
            audio_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
                None
            )
            
            if audio_stream:
                video_file.audio_codec = audio_stream.get('codec_name', '')
                video_file.metadata['audio'] = {
                    'codec': audio_stream.get('codec_name', ''),
                    'channels': audio_stream.get('channels', 0),
                    'sample_rate': audio_stream.get('sample_rate', 0)
                }
            
            # Extract additional metadata
            video_file.metadata['format'] = probe['format']
            video_file.metadata['streams'] = probe['streams']
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {video_file.file_name}: {e}")
            video_file.error = f"Metadata extraction error: {str(e)}"
    
    def _generate_thumbnail(self, video_file: VideoFile) -> None:
        """
        Generate a thumbnail for a video file
        
        Args:
            video_file: VideoFile object to update
        """
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_file.file_path)
            
            # Check if video opened successfully
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get total number of frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Seek to 10% of the video
            target_frame = min(int(total_frames * 0.1), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            # Read the frame
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Could not read frame")
            
            # Generate thumbnail filename
            thumbnail_name = f"{os.path.splitext(video_file.file_name)[0]}_thumbnail.jpg"
            thumbnail_path = os.path.join(self.thumbnail_dir, thumbnail_name)
            
            # Resize the frame to a reasonable thumbnail size
            height, width = frame.shape[:2]
            max_dim = 320
            if height > width:
                new_height = max_dim
                new_width = int(width * (max_dim / height))
            else:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
            
            thumbnail = cv2.resize(frame, (new_width, new_height))
            
            # Save the thumbnail
            cv2.imwrite(thumbnail_path, thumbnail)
            
            # Update the video file object
            video_file.thumbnail_path = thumbnail_path
            
            # Release the video capture object
            cap.release()
            
        except Exception as e:
            logger.error(f"Error generating thumbnail for {video_file.file_name}: {e}")
            video_file.error = f"Thumbnail generation error: {str(e)}"
    
    def _save_metadata(self, video_file: VideoFile) -> None:
        """
        Save video metadata to a JSON file
        
        Args:
            video_file: VideoFile object to save
        """
        try:
            # Generate metadata filename
            metadata_name = f"{os.path.splitext(video_file.file_name)[0]}_metadata.json"
            metadata_path = os.path.join(self.metadata_dir, metadata_name)
            
            # Prepare metadata dictionary
            metadata = {
                'file_name': video_file.file_name,
                'file_size': video_file.file_size,
                'file_hash': video_file.file_hash,
                'upload_time': video_file.upload_time.isoformat(),
                'processed': video_file.processed,
                'width': video_file.width,
                'height': video_file.height,
                'fps': video_file.fps,
                'duration': video_file.duration,
                'format': video_file.format,
                'bitrate': video_file.bitrate,
                'codec': video_file.codec,
                'audio_codec': video_file.audio_codec,
                'processed_path': video_file.processed_path,
                'thumbnail_path': video_file.thumbnail_path,
                'error': video_file.error,
                'additional_metadata': video_file.metadata
            }
            
            # Save to JSON file
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving metadata for {video_file.file_name}: {e}")
    
    def export_to_csv(self, video_files: List[VideoFile], output_path: Optional[str] = None) -> str:
        """
        Export video file metadata to CSV
        
        Args:
            video_files: List of VideoFile objects
            output_path: Path to save CSV file
            
        Returns:
            Path to the saved CSV file
        """
        if not output_path:
            output_path = os.path.join(self.processed_dir, f"video_files_{int(time.time())}.csv")
        
        try:
            import pandas as pd
            
            # Convert to DataFrame
            data = [
                {
                    'file_name': vf.file_name,
                    'file_size': vf.file_size,
                    'file_hash': vf.file_hash,
                    'upload_time': vf.upload_time,
                    'processed': vf.processed,
                    'width': vf.width,
                    'height': vf.height,
                    'fps': vf.fps,
                    'duration': vf.duration,
                    'format': vf.format,
                    'codec': vf.codec,
                    'audio_codec': vf.audio_codec,
                    'processed_path': vf.processed_path,
                    'thumbnail_path': vf.thumbnail_path,
                    'error': vf.error
                } for vf in video_files
            ]
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Exported {len(video_files)} video files to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return None

def main():
    """Example usage of the UploadHandler"""
    # Create an upload handler
    handler = UploadHandler(
        upload_dir='data/uploads',
        processed_dir='data/processed'
    )
    
    # Create directories if they don't exist
    os.makedirs('data/uploads', exist_ok=True)
    
    # Example: Process a single file
    # (Uncomment and modify path if you want to test with a specific file)
    # video_file = handler.process_upload('path/to/video.mp4', move_file=True)
    # print(f"Processed: {video_file.file_name}, Resolution: {video_file.width}x{video_file.height}")
    
    # Example: Process all files in the upload directory
    video_files = handler.process_directory(recursive=True)
    
    if video_files:
        print(f"Processed {len(video_files)} files:")
        for vf in video_files:
            status = "Success" if vf.processed else f"Error: {vf.error}"
            print(f"- {vf.file_name}: {status}")
        
        # Example: Export metadata to CSV
        csv_path = handler.export_to_csv(video_files)
        if csv_path:
            print(f"Exported metadata to {csv_path}")
    else:
        print("No video files found in the upload directory.")

if __name__ == "__main__":
    main()
