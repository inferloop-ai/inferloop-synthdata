import cv2
import numpy as np
import time
import json
import requests
import threading
import queue
from typing import Optional, Generator, Dict, List, Tuple, Callable
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class EdgeCameraIngester:
    """Ingest images from edge IoT cameras for real-time profiling."""
    
    def __init__(self, camera_ip: str, port: int = 80, username: Optional[str] = None, 
                 password: Optional[str] = None, api_key: Optional[str] = None,
                 camera_type: str = 'generic', buffer_size: int = 30):
        self.camera_ip = camera_ip
        self.port = port
        self.username = username
        self.password = password
        self.api_key = api_key
        self.camera_type = camera_type.lower()
        self.buffer_size = buffer_size
        self.base_url = f"http://{camera_ip}:{port}"
        self.session = None
        self.is_running = False
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.thread = None
        self.last_frame_time = 0
        self.connection_retries = 3
        self.retry_delay = 5  # seconds
        
        # Camera-specific endpoints
        self.endpoints = self._get_camera_endpoints()
    
    def _get_camera_endpoints(self) -> Dict:
        """Get API endpoints based on camera type."""
        endpoints = {
            'generic': {
                'snapshot': '/snapshot',
                'stream': '/video',
                'info': '/info',
                'settings': '/settings'
            },
            'axis': {
                'snapshot': '/axis-cgi/jpg/image.cgi',
                'stream': '/axis-cgi/mjpg/video.cgi',
                'info': '/axis-cgi/param.cgi?action=list',
                'settings': '/axis-cgi/param.cgi'
            },
            'hikvision': {
                'snapshot': '/ISAPI/Streaming/channels/101/picture',
                'stream': '/ISAPI/Streaming/channels/101/httpPreview',
                'info': '/ISAPI/System/deviceInfo',
                'settings': '/ISAPI/Image/channels/1/imageParam'
            },
            'dahua': {
                'snapshot': '/cgi-bin/snapshot.cgi',
                'stream': '/cgi-bin/mjpg/video.cgi',
                'info': '/cgi-bin/magicBox.cgi?action=getSystemInfo',
                'settings': '/cgi-bin/configManager.cgi?action=getConfig&name=VideoInOptions'
            },
            'amcrest': {
                'snapshot': '/cgi-bin/snapshot.cgi',
                'stream': '/cgi-bin/mjpg/video.cgi',
                'info': '/cgi-bin/magicBox.cgi?action=getSystemInfo',
                'settings': '/cgi-bin/configManager.cgi?action=getConfig&name=VideoInOptions'
            }
        }
        
        # Default to generic if camera type not found
        return endpoints.get(self.camera_type, endpoints['generic'])
    
    def initialize_session(self) -> bool:
        """Initialize HTTP session with authentication."""
        try:
            self.session = requests.Session()
            
            # Set authentication based on what's provided
            if self.username and self.password:
                self.session.auth = (self.username, self.password)
            
            if self.api_key:
                self.session.headers.update({'Authorization': f'Bearer {self.api_key}'})
            
            # Test connection with info endpoint
            info_url = f"{self.base_url}{self.endpoints['info']}"
            response = self.session.get(info_url, timeout=5)
            
            if response.status_code == 200:
                logger.info(f"Successfully connected to edge camera at {self.camera_ip}")
                return True
            else:
                logger.error(f"Failed to connect to camera: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize camera session: {e}")
            return False
    
    def capture_snapshot(self) -> Optional[np.ndarray]:
        """Capture a single snapshot from the edge camera."""
        if self.session is None:
            if not self.initialize_session():
                return None
        
        try:
            snapshot_url = f"{self.base_url}{self.endpoints['snapshot']}"
            response = self.session.get(snapshot_url, timeout=5)
            
            if response.status_code == 200:
                # Convert to numpy array
                img_array = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if image is not None:
                    logger.debug(f"Captured snapshot: {image.shape}")
                    return image
                else:
                    logger.warning("Failed to decode image")
            else:
                logger.warning(f"Failed to capture snapshot: HTTP {response.status_code}")
            
            return None
                
        except Exception as e:
            logger.error(f"Error capturing snapshot: {e}")
            return None
    
    def _buffer_frames(self, interval: float = 1.0):
        """Background thread to continuously buffer frames."""
        retry_count = 0
        
        while self.is_running:
            try:
                frame = self.capture_snapshot()
                
                if frame is not None:
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
                    retry_count += 1
                    if retry_count >= self.connection_retries:
                        logger.error(f"Failed to capture {self.connection_retries} consecutive frames. Reinitializing session...")
                        self.initialize_session()
                        retry_count = 0
            
            except Exception as e:
                logger.error(f"Error in frame buffer thread: {e}")
                retry_count += 1
            
            # Sleep for the specified interval
            time.sleep(interval)
    
    def start_ingestion(self, interval: float = 1.0, 
                       on_frame: Optional[Callable] = None,
                       on_error: Optional[Callable] = None) -> bool:
        """Start continuous ingestion from the edge camera."""
        if not self.initialize_session():
            if on_error:
                on_error("Failed to initialize camera session")
            return False
        
        self.is_running = True
        self.thread = threading.Thread(target=self._buffer_frames, args=(interval,))
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"Started edge camera ingestion from {self.camera_ip}:{self.port}")
        
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
        logger.info("Edge camera ingestion stopped")
    
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
    
    def get_camera_settings(self) -> Dict:
        """Get camera settings."""
        if self.session is None:
            if not self.initialize_session():
                return {}
        
        try:
            settings_url = f"{self.base_url}{self.endpoints['settings']}"
            response = self.session.get(settings_url, timeout=5)
            
            if response.status_code == 200:
                # Try to parse as JSON first
                try:
                    return response.json()
                except:
                    # Return as text if not JSON
                    return {'raw_settings': response.text}
            else:
                logger.warning(f"Failed to get camera settings: HTTP {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting camera settings: {e}")
            return {}
    
    def get_camera_info(self) -> Dict:
        """Get camera information."""
        if self.session is None:
            if not self.initialize_session():
                return {}
        
        try:
            info = {
                'camera_ip': self.camera_ip,
                'port': self.port,
                'camera_type': self.camera_type,
                'buffer_size': self.buffer_size,
                'buffer_usage': self.frame_buffer.qsize(),
                'last_frame_time': self.last_frame_time,
                'is_running': self.is_running
            }
            
            # Try to get additional info from camera
            info_url = f"{self.base_url}{self.endpoints['info']}"
            response = self.session.get(info_url, timeout=5)
            
            if response.status_code == 200:
                # Try to parse as JSON first
                try:
                    info['camera_details'] = response.json()
                except:
                    # Store as text if not JSON
                    info['camera_details'] = {'raw_info': response.text}
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get camera info: {e}")
            return {}

if __name__ == "__main__":
    # Test the edge camera ingester
    import argparse
    
    parser = argparse.ArgumentParser(description='Test edge camera ingestion')
    parser.add_argument('--ip', required=True, help='IP address of the camera')
    parser.add_argument('--port', type=int, default=80, help='Port number')
    parser.add_argument('--username', help='Username for authentication')
    parser.add_argument('--password', help='Password for authentication')
    parser.add_argument('--type', default='generic', help='Camera type (generic, axis, hikvision, dahua, amcrest)')
    parser.add_argument('--frames', type=int, default=5, help='Number of frames to capture')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize ingester
        ingester = EdgeCameraIngester(
            camera_ip=args.ip,
            port=args.port,
            username=args.username,
            password=args.password,
            camera_type=args.type
        )
        
        # Get camera info
        info = ingester.get_camera_info()
        print("Camera info:", info)
        
        # Capture a single snapshot
        print("Capturing single snapshot...")
        frame = ingester.capture_snapshot()
        if frame is not None:
            print(f"Snapshot captured: {frame.shape}")
            cv2.imwrite("test_edge_camera.jpg", frame)
            print("Saved test_edge_camera.jpg")
        
        # Start ingestion and get a batch
        print(f"Starting ingestion, will capture {args.frames} frames...")
        ingester.start_ingestion(interval=0.5)
        
        # Get a batch of frames
        frames = ingester.get_batch(count=args.frames, timeout=10.0)
        print(f"Captured {len(frames)} frames")
        
        # Stop ingestion
        ingester.stop_ingestion()
        
    except Exception as e:
        print(f"Test failed: {e}")
