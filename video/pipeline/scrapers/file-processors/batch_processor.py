#!/usr/bin/env python3
"""
Batch Video Processor

This module provides functionality to process batches of video files
for use in the Inferloop Synthetic Data pipeline.
"""

import os
import json
import logging
import time
import shutil
import concurrent.futures
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import cv2
import numpy as np
from tqdm import tqdm

# Import local modules
from .upload_handler import UploadHandler, VideoFile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BatchProcessingResult:
    """Class to store batch processing results"""
    total_files: int
    successful_files: int
    failed_files: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    processed_files: List[VideoFile] = field(default_factory=list)
    error_files: List[VideoFile] = field(default_factory=list)
    output_directory: Optional[str] = None
    summary_file: Optional[str] = None

class BatchProcessor:
    """Processor for handling batches of video files"""
    
    def __init__(self, input_dir: str = 'batch_input', output_dir: str = 'batch_output',
                 max_workers: int = 4):
        """
        Initialize the batch processor
        
        Args:
            input_dir: Directory for incoming batch files
            output_dir: Directory for processed batch files
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_workers = max_workers
        
        # Create directories if they don't exist
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create upload handler for individual file processing
        self.upload_handler = UploadHandler(
            upload_dir=input_dir,
            processed_dir=output_dir
        )
        
        # Create batch-specific directories
        self.results_dir = os.path.join(output_dir, 'results')
        self.logs_dir = os.path.join(output_dir, 'logs')
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def process_batch(self, batch_dir: Optional[str] = None, 
                     file_filter: Optional[Callable[[str], bool]] = None,
                     move_files: bool = True) -> BatchProcessingResult:
        """
        Process a batch of video files
        
        Args:
            batch_dir: Directory containing the batch of files (defaults to self.input_dir)
            file_filter: Optional function to filter files (takes filename, returns bool)
            move_files: Whether to move processed files to output directory
            
        Returns:
            BatchProcessingResult with processing statistics
        """
        if batch_dir is None:
            batch_dir = self.input_dir
        
        logger.info(f"Starting batch processing of {batch_dir}")
        
        # Create batch result object
        result = BatchProcessingResult(
            total_files=0,
            successful_files=0,
            failed_files=0,
            start_time=datetime.now(),
            output_directory=os.path.join(self.output_dir, f"batch_{int(time.time())}")
        )
        
        # Create batch-specific output directory
        os.makedirs(result.output_directory, exist_ok=True)
        
        # Find all video files in the batch directory
        video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
        video_files = []
        
        for root, _, files in os.walk(batch_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    # Apply filter if provided
                    if file_filter is None or file_filter(file):
                        file_path = os.path.join(root, file)
                        video_files.append(file_path)
        
        result.total_files = len(video_files)
        logger.info(f"Found {result.total_files} video files to process")
        
        if result.total_files == 0:
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            return result
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all processing tasks
            future_to_file = {
                executor.submit(self._process_file, file_path, result.output_directory, move_files): file_path
                for file_path in video_files
            }
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                              total=len(future_to_file), 
                              desc="Processing videos"):
                file_path = future_to_file[future]
                try:
                    video_file = future.result()
                    if video_file.processed and not video_file.error:
                        result.successful_files += 1
                        result.processed_files.append(video_file)
                    else:
                        result.failed_files += 1
                        result.error_files.append(video_file)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    result.failed_files += 1
        
        # Calculate duration
        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        
        # Generate summary
        summary_file = self._generate_summary(result)
        result.summary_file = summary_file
        
        logger.info(f"Batch processing completed: {result.successful_files} successful, "
                   f"{result.failed_files} failed, took {result.duration_seconds:.2f} seconds")
        
        return result
    
    def _process_file(self, file_path: str, output_dir: str, move_file: bool) -> VideoFile:
        """
        Process a single file in the batch
        
        Args:
            file_path: Path to the video file
            output_dir: Directory to save processed output
            move_file: Whether to move the file after processing
            
        Returns:
            VideoFile object with processing results
        """
        try:
            # Create file-specific output directory
            file_name = os.path.basename(file_path)
            file_output_dir = os.path.join(output_dir, os.path.splitext(file_name)[0])
            os.makedirs(file_output_dir, exist_ok=True)
            
            # Process the file using the upload handler
            video_file = self.upload_handler.process_upload(file_path, move_file=False)
            
            # If successful and move_file is True, move to output directory
            if video_file.processed and move_file:
                new_path = os.path.join(file_output_dir, file_name)
                shutil.copy2(file_path, new_path)
                
                # If we have a thumbnail, copy it too
                if video_file.thumbnail_path and os.path.exists(video_file.thumbnail_path):
                    thumbnail_name = os.path.basename(video_file.thumbnail_path)
                    new_thumbnail_path = os.path.join(file_output_dir, thumbnail_name)
                    shutil.copy2(video_file.thumbnail_path, new_thumbnail_path)
                    video_file.thumbnail_path = new_thumbnail_path
                
                # Update paths
                video_file.processed_path = new_path
                
                # Save metadata to the file-specific output directory
                metadata_path = os.path.join(file_output_dir, f"{os.path.splitext(file_name)[0]}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump({
                        'file_name': video_file.file_name,
                        'file_size': video_file.file_size,
                        'file_hash': video_file.file_hash,
                        'width': video_file.width,
                        'height': video_file.height,
                        'fps': video_file.fps,
                        'duration': video_file.duration,
                        'format': video_file.format,
                        'codec': video_file.codec,
                        'processed_time': datetime.now().isoformat(),
                        'thumbnail_path': video_file.thumbnail_path
                    }, f, indent=2)
            
            return video_file
            
        except Exception as e:
            logger.error(f"Error in batch processing {file_path}: {e}")
            
            # Create error record
            return VideoFile(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                file_hash="",
                upload_time=datetime.now(),
                processed=False,
                error=str(e)
            )
    
    def _generate_summary(self, result: BatchProcessingResult) -> str:
        """
        Generate a summary of the batch processing
        
        Args:
            result: BatchProcessingResult to summarize
            
        Returns:
            Path to the summary file
        """
        # Create summary filename with timestamp
        timestamp = result.start_time.strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.results_dir, f"batch_summary_{timestamp}.json")
        
        # Prepare summary data
        summary = {
            'timestamp': timestamp,
            'total_files': result.total_files,
            'successful_files': result.successful_files,
            'failed_files': result.failed_files,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat(),
            'duration_seconds': result.duration_seconds,
            'output_directory': result.output_directory,
            'successful_files_list': [
                {
                    'file_name': vf.file_name,
                    'file_size': vf.file_size,
                    'width': vf.width,
                    'height': vf.height,
                    'duration': vf.duration,
                    'processed_path': vf.processed_path
                } for vf in result.processed_files
            ],
            'failed_files_list': [
                {
                    'file_name': vf.file_name,
                    'file_size': vf.file_size,
                    'error': vf.error
                } for vf in result.error_files
            ]
        }
        
        # Save summary to JSON file
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also generate a CSV summary
        try:
            import pandas as pd
            
            # Create DataFrames for successful and failed files
            if result.processed_files:
                success_df = pd.DataFrame([
                    {
                        'file_name': vf.file_name,
                        'file_size': vf.file_size,
                        'width': vf.width,
                        'height': vf.height,
                        'fps': vf.fps,
                        'duration': vf.duration,
                        'format': vf.format,
                        'codec': vf.codec
                    } for vf in result.processed_files
                ])
                success_csv = os.path.join(self.results_dir, f"successful_files_{timestamp}.csv")
                success_df.to_csv(success_csv, index=False)
            
            if result.error_files:
                error_df = pd.DataFrame([
                    {
                        'file_name': vf.file_name,
                        'file_size': vf.file_size,
                        'error': vf.error
                    } for vf in result.error_files
                ])
                error_csv = os.path.join(self.results_dir, f"failed_files_{timestamp}.csv")
                error_df.to_csv(error_csv, index=False)
                
        except Exception as e:
            logger.warning(f"Could not generate CSV summary: {e}")
        
        logger.info(f"Batch summary saved to {summary_path}")
        return summary_path
    
    def generate_report(self, result: BatchProcessingResult, output_path: Optional[str] = None) -> str:
        """
        Generate a detailed HTML report of the batch processing
        
        Args:
            result: BatchProcessingResult to report on
            output_path: Path to save the HTML report
            
        Returns:
            Path to the HTML report
        """
        if not output_path:
            timestamp = result.start_time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.results_dir, f"batch_report_{timestamp}.html")
        
        try:
            # Generate HTML report
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Batch Processing Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                    .success {{ color: green; }}
                    .error {{ color: red; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                <h1>Batch Processing Report</h1>
                <div class="summary">
                    <h2>Summary</h2>
                    <p>Start Time: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>End Time: {result.end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Duration: {result.duration_seconds:.2f} seconds</p>
                    <p>Total Files: {result.total_files}</p>
                    <p>Successful Files: <span class="success">{result.successful_files}</span></p>
                    <p>Failed Files: <span class="error">{result.failed_files}</span></p>
                    <p>Success Rate: {(result.successful_files / result.total_files * 100) if result.total_files > 0 else 0:.2f}%</p>
                </div>
            """
            
            # Add successful files table
            if result.processed_files:
                html += """
                <h2>Successful Files</h2>
                <table>
                    <tr>
                        <th>File Name</th>
                        <th>Size (MB)</th>
                        <th>Resolution</th>
                        <th>Duration</th>
                        <th>Format</th>
                        <th>Codec</th>
                    </tr>
                """
                
                for vf in result.processed_files:
                    html += f"""
                    <tr>
                        <td>{vf.file_name}</td>
                        <td>{vf.file_size / (1024 * 1024):.2f}</td>
                        <td>{vf.width}x{vf.height if vf.width and vf.height else 'N/A'}</td>
                        <td>{vf.duration:.2f if vf.duration else 'N/A'}</td>
                        <td>{vf.format if vf.format else 'N/A'}</td>
                        <td>{vf.codec if vf.codec else 'N/A'}</td>
                    </tr>
                    """
                
                html += "</table>"
            
            # Add failed files table
            if result.error_files:
                html += """
                <h2>Failed Files</h2>
                <table>
                    <tr>
                        <th>File Name</th>
                        <th>Size (MB)</th>
                        <th>Error</th>
                    </tr>
                """
                
                for vf in result.error_files:
                    html += f"""
                    <tr>
                        <td>{vf.file_name}</td>
                        <td>{vf.file_size / (1024 * 1024):.2f}</td>
                        <td class="error">{vf.error}</td>
                    </tr>
                    """
                
                html += "</table>"
            
            html += """
            </body>
            </html>
            """
            
            # Write HTML to file
            with open(output_path, 'w') as f:
                f.write(html)
            
            logger.info(f"HTML report generated at {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return ""

def main():
    """Example usage of the BatchProcessor"""
    # Create a batch processor
    processor = BatchProcessor(
        input_dir='data/batch_input',
        output_dir='data/batch_output',
        max_workers=4
    )
    
    # Create directories if they don't exist
    os.makedirs('data/batch_input', exist_ok=True)
    
    # Example: Process a batch of files
    result = processor.process_batch(move_files=True)
    
    if result.total_files > 0:
        print(f"Batch processing completed:")
        print(f"- Total files: {result.total_files}")
        print(f"- Successful: {result.successful_files}")
        print(f"- Failed: {result.failed_files}")
        print(f"- Duration: {result.duration_seconds:.2f} seconds")
        
        # Generate HTML report
        report_path = processor.generate_report(result)
        print(f"HTML report generated at {report_path}")
    else:
        print("No video files found in the input directory.")

if __name__ == "__main__":
    main()
