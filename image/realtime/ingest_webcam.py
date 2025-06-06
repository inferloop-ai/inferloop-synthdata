import cv2
import numpy as np
import time
from typing import Optional, Generator, Dict, List
import logging

logger = logging.getLogger(__name__)

class WebcamIngester:
    """Capture images from webcam for real-time profiling."""
    
    def __init__(self, device_id: int = 0, resolution: tuple = (640, 480)):
        self.device_id = device_id
        self.resolution = resolution
        self.cap = None
        
    def initialize_camera(self) -> bool:
        """Initialize the camera capture."""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                logger.error(f"Cannot open camera {self.device_id}")
                return False
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Test capture
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Cannot read from camera")
                return False
            
            logger.info(f"Camera {self.device_id} initialized: {frame.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def capture_single_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the webcam."""
        if self.cap is None or not self.cap.isOpened():
            if not self.initialize_camera():
                return None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                logger.warning("Failed to capture frame")
                return None
                
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    def capture_batch(self, count: int = 10, interval: float = 1.0) -> List[np.ndarray]:
        """Capture a batch of frames with specified interval."""
        frames = []
        
        for i in range(count):
            frame = self.capture_single_frame()
            if frame is not None:
                frames.append(frame.copy())
                logger.debug(f"Captured frame {i+1}/{count}")
            else:
                logger.warning(f"Skipped frame {i+1}/{count}")
            
            if i < count - 1:  # Don't sleep after last frame
                time.sleep(interval)
        
        logger.info(f"Captured {len(frames)} frames from webcam")
        return frames
    
    def stream_frames(self, 
                     capture_interval: float = 1.0,
                     batch_size: int = 10,
                     batch_interval: float = 60.0) -> Generator[Dict, None, None]:
        """Stream frames in batches for real-time profiling."""
        logger.info(f"Starting webcam stream: {batch_size} frames every {batch_interval}s")
        
        if not self.initialize_camera():
            return
        
        try:
            while True:
                batch_frames = []
                start_time = time.time()
                
                # Capture batch
                for i in range(batch_size):
                    frame = self.capture_single_frame()
                    if frame is not None:
                        batch_frames.append(frame.copy())
                    
                    if i < batch_size - 1:
                        time.sleep(capture_interval)
                
                if batch_frames:
                    yield {
                        'images': batch_frames,
                        'source': f'webcam_{self.device_id}',
                        'timestamp': time.time(),
                        'count': len(batch_frames),
                        'capture_duration': time.time() - start_time
                    }
                
                # Wait for next batch
                elapsed = time.time() - start_time
                remaining = batch_interval - elapsed
                if remaining > 0:
                    logger.info(f"Batch complete, sleeping for {remaining:.1f} seconds")
                    time.sleep(remaining)
                
        except KeyboardInterrupt:
            logger.info("Webcam stream interrupted by user")
        except Exception as e:
            logger.error(f"Error in webcam stream: {e}")
        finally:
            self.release()
    
    def release(self):
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released")
    
    def get_camera_info(self) -> Dict:
        """Get camera information and capabilities."""
        if self.cap is None:
            if not self.initialize_camera():
                return {}
        
        try:
            info = {
                'device_id': self.device_id,
                'resolution': (
                    int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'backend': self.cap.getBackendName(),
                'fourcc': int(self.cap.get(cv2.CAP_PROP_FOURCC)),
                'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
                'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
                'hue': self.cap.get(cv2.CAP_PROP_HUE)
            }
            return info
            
        except Exception as e:
            logger.error(f"Failed to get camera info: {e}")
            return {}

if __name__ == "__main__":
    # Test the webcam ingester
    try:
        ingester = WebcamIngester(device_id=0)
        
        # Get camera info
        info = ingester.get_camera_info()
        print("Camera info:", info)
        
        # Capture a single frame
        frame = ingester.capture_single_frame()
        if frame is not None:
            print(f"Captured frame: {frame.shape}")
            cv2.imwrite("test_webcam_frame.jpg", frame)
        
        # Capture a small batch
        batch = ingester.capture_batch(count=3, interval=0.5)
        print(f"Captured batch: {len(batch)} frames")
        
        ingester.release()
        
    except Exception as e:
        print(f"Test failed: {e}")
