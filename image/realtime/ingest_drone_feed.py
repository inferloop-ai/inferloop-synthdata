import cv2
import numpy as np
import time
import threading
import queue
from typing import Optional, Generator, Dict, List, Callable
import logging
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class DroneFeedIngester:
    """Capture images from drone video feeds for real-time profiling."""
    
    def __init__(self, stream_url: str, auth_token: Optional[str] = None, 
                 resolution: tuple = (1280, 720), buffer_size: int = 30):
        self.stream_url = stream_url
        self.auth_token = auth_token
        self.resolution = resolution
        self.buffer_size = buffer_size
        self.cap = None
        self.is_running = False
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.thread = None
        self.last_frame_time = 0
        self.connection_retries = 3
        self.retry_delay = 5  # seconds
        
    def _validate_stream_url(self) -> bool:
        """Validate the stream URL format."""
        try:
            parsed = urlparse(self.stream_url)
            valid_schemes = ['rtsp', 'rtmp', 'http', 'https']
            
            if parsed.scheme not in valid_schemes:
                logger.error(f"Invalid stream URL scheme: {parsed.scheme}. Must be one of {valid_schemes}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Invalid stream URL: {e}")
            return False
    
    def initialize_stream(self) -> bool:
        """Initialize the drone video stream connection."""
        if not self._validate_stream_url():
            return False
            
        try:
            # Set up connection with auth if provided
            if self.auth_token:
                # For RTSP with token auth
                if self.stream_url.startswith('rtsp'):
                    self.stream_url = f"{self.stream_url}?auth={self.auth_token}"
                # For HTTP(S) with token auth in header
                else:
                    self.cap = cv2.VideoCapture(self.stream_url)
                    self.cap.setExtraParam(cv2.CAP_PROP_HEADERS, f"Authorization: Bearer {self.auth_token}")
            else:
                self.cap = cv2.VideoCapture(self.stream_url)
            
            if not self.cap.isOpened():
                logger.error(f"Cannot open drone stream: {self.stream_url}")
                return False
            
            # Set resolution if possible
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Test capture
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Cannot read from drone stream")
                return False
            
            logger.info(f"Drone stream initialized: {frame.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize drone stream: {e}")
            return False
    
    def _buffer_frames(self):
        """Background thread to continuously buffer frames from the stream."""
        retry_count = 0
        
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                if retry_count < self.connection_retries:
                    logger.warning(f"Stream connection lost. Retrying ({retry_count+1}/{self.connection_retries})...")
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    if self.initialize_stream():
                        retry_count = 0
                    continue
                else:
                    logger.error("Max retries reached. Stopping frame buffer thread.")
                    break
            
            try:
                ret, frame = self.cap.read()
                if ret:
                    # If buffer is full, remove oldest frame
                    if self.frame_buffer.full():
                        try:
                            self.frame_buffer.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.frame_buffer.put(frame.copy())
                    self.last_frame_time = time.time()
                    retry_count = 0  # Reset retry counter on successful frame
                else:
                    logger.warning("Failed to capture frame from drone stream")
                    time.sleep(0.1)  # Small delay to prevent CPU spinning
                    
            except Exception as e:
                logger.error(f"Error in frame buffer thread: {e}")
                time.sleep(0.5)  # Delay before retry
    
    def start_ingestion(self, on_frame: Optional[Callable] = None, 
                       on_error: Optional[Callable] = None,
                       buffer_size: Optional[int] = None) -> bool:
        """Start continuous ingestion from the drone feed."""
        if buffer_size is not None:
            self.buffer_size = buffer_size
            self.frame_buffer = queue.Queue(maxsize=buffer_size)
        
        if not self.initialize_stream():
            if on_error:
                on_error("Failed to initialize drone stream")
            return False
        
        self.is_running = True
        self.thread = threading.Thread(target=self._buffer_frames)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"Started drone feed ingestion from {self.stream_url}")
        
        # If callback provided, process frames as they arrive
        if on_frame:
            try:
                while self.is_running:
                    try:
                        frame = self.frame_buffer.get(timeout=1.0)
                        on_frame(frame)
                    except queue.Empty:
                        continue
            except Exception as e:
                if on_error:
                    on_error(f"Error processing frames: {e}")
                logger.error(f"Error in frame processing: {e}")
                return False
        
        return True
    
    def stop_ingestion(self):
        """Stop the ingestion process."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.release()
        logger.info("Drone feed ingestion stopped")
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get the latest frame from the buffer."""
        try:
            return self.frame_buffer.get(timeout=timeout)
        except queue.Empty:
            logger.warning("No frames available in buffer")
            return None
    
    def get_batch(self, count: int = 10, timeout: float = 5.0) -> List[np.ndarray]:
        """Get a batch of frames from the buffer."""
        frames = []
        end_time = time.time() + timeout
        
        while len(frames) < count and time.time() < end_time:
            try:
                remaining = end_time - time.time()
                if remaining <= 0:
                    break
                    
                frame = self.frame_buffer.get(timeout=min(1.0, remaining))
                frames.append(frame.copy())
                
            except queue.Empty:
                continue
        
        logger.info(f"Retrieved {len(frames)} frames from buffer")
        return frames
    
    def release(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def get_stream_info(self) -> Dict:
        """Get information about the drone stream."""
        if self.cap is None or not self.cap.isOpened():
            if not self.initialize_stream():
                return {}
        
        try:
            info = {
                'stream_url': self.stream_url.split('?')[0],  # Remove auth token from URL
                'resolution': (
                    int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'codec': int(self.cap.get(cv2.CAP_PROP_FOURCC)),
                'buffer_size': self.buffer_size,
                'buffer_usage': self.frame_buffer.qsize(),
                'last_frame_time': self.last_frame_time,
                'is_running': self.is_running
            }
            return info
            
        except Exception as e:
            logger.error(f"Failed to get stream info: {e}")
            return {}

if __name__ == "__main__":
    # Test the drone feed ingester
    import argparse
    
    parser = argparse.ArgumentParser(description='Test drone feed ingestion')
    parser.add_argument('--url', required=True, help='RTSP or HTTP URL of the drone stream')
    parser.add_argument('--token', help='Optional auth token')
    parser.add_argument('--frames', type=int, default=5, help='Number of frames to capture')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize ingester
        ingester = DroneFeedIngester(stream_url=args.url, auth_token=args.token)
        
        # Get stream info
        info = ingester.get_stream_info()
        print("Stream info:", info)
        
        # Define frame callback
        def process_frame(frame):
            print(f"Received frame: {frame.shape}")
        
        # Start ingestion
        print(f"Starting ingestion, will capture {args.frames} frames...")
        ingester.start_ingestion()
        
        # Get a batch of frames
        frames = ingester.get_batch(count=args.frames, timeout=10.0)
        print(f"Captured {len(frames)} frames")
        
        # Save the first frame if available
        if frames:
            cv2.imwrite("test_drone_frame.jpg", frames[0])
            print("Saved test_drone_frame.jpg")
        
        # Stop ingestion
        ingester.stop_ingestion()
        
    except Exception as e:
        print(f"Test failed: {e}")
