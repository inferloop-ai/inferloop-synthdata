import requests
import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class UnsplashIngester:
    """Ingest images from Unsplash API for real-time profiling."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('UNSPLASH_API_KEY')
        if not self.api_key:
            raise ValueError("Unsplash API key required. Set UNSPLASH_API_KEY environment variable.")
        
        self.base_url = "https://api.unsplash.com"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Client-ID {self.api_key}',
            'Accept-Version': 'v1'
        })
        
    def fetch_random_images(self, 
                          query: str = "urban", 
                          count: int = 10,
                          orientation: str = "landscape") -> List[np.ndarray]:
        """Fetch random images from Unsplash API."""
        logger.info(f"Fetching {count} images with query: '{query}'")
        
        try:
            params = {
                'query': query,
                'count': min(count, 30),  # API limit
                'orientation': orientation,
                'content_filter': 'high'
            }
            
            response = self.session.get(f"{self.base_url}/photos/random", params=params)
            response.raise_for_status()
            
            data = response.json()
            if not isinstance(data, list):
                data = [data]  # Single image response
            
            images = []
            for i, img_data in enumerate(data):
                try:
                    # Download image
                    img_url = img_data['urls']['regular']  # ~1080px wide
                    img_response = requests.get(img_url, timeout=30)
                    img_response.raise_for_status()
                    
                    # Convert to numpy array
                    img_array = np.frombuffer(img_response.content, np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        images.append(image)
                        logger.debug(f"Downloaded image {i+1}/{len(data)}: {image.shape}")
                    else:
                        logger.warning(f"Failed to decode image {i+1}")
                        
                except Exception as e:
                    logger.error(f"Failed to download image {i+1}: {e}")
                    continue
            
            logger.info(f"Successfully downloaded {len(images)} images")
            return images
            
        except Exception as e:
            logger.error(f"Failed to fetch images from Unsplash: {e}")
            return []
    
    def fetch_by_collections(self, collection_ids: List[str], per_collection: int = 5) -> List[np.ndarray]:
        """Fetch images from specific Unsplash collections."""
        all_images = []
        
        for collection_id in collection_ids:
            try:
                params = {
                    'per_page': per_collection,
                    'page': 1
                }
                
                response = self.session.get(
                    f"{self.base_url}/collections/{collection_id}/photos", 
                    params=params
                )
                response.raise_for_status()
                
                data = response.json()
                
                for img_data in data:
                    try:
                        img_url = img_data['urls']['regular']
                        img_response = requests.get(img_url, timeout=30)
                        img_response.raise_for_status()
                        
                        img_array = np.frombuffer(img_response.content, np.uint8)
                        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        
                        if image is not None:
                            all_images.append(image)
                            
                    except Exception as e:
                        logger.error(f"Failed to download from collection {collection_id}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to fetch collection {collection_id}: {e}")
                continue
        
        return all_images
    
    def stream_images(self, 
                     queries: List[str], 
                     interval: int = 300,
                     images_per_batch: int = 10) -> None:
        """Continuously stream images for real-time profiling."""
        logger.info(f"Starting image stream with {len(queries)} queries, {interval}s interval")
        
        query_index = 0
        
        while True:
            try:
                current_query = queries[query_index % len(queries)]
                
                # Fetch batch of images
                images = self.fetch_random_images(
                    query=current_query,
                    count=images_per_batch
                )
                
                if images:
                    # Process images with profiler (this would be called by the main pipeline)
                    yield {
                        'images': images,
                        'source': f'unsplash_{current_query}',
                        'query': current_query,
                        'timestamp': time.time(),
                        'count': len(images)
                    }
                
                query_index += 1
                
                logger.info(f"Sleeping for {interval} seconds...")
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Stream interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in image stream: {e}")
                time.sleep(60)  # Wait before retrying
    
    def save_images_locally(self, images: List[np.ndarray], output_dir: str, prefix: str = "unsplash") -> List[str]:
        """Save images to local directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        timestamp = int(time.time())
        
        for i, image in enumerate(images):
            filename = f"{prefix}_{timestamp}_{i:04d}.jpg"
            filepath = output_path / filename
            
            try:
                cv2.imwrite(str(filepath), image)
                saved_paths.append(str(filepath))
            except Exception as e:
                logger.error(f"Failed to save image {filename}: {e}")
        
        return saved_paths

if __name__ == "__main__":
    # Test the Unsplash ingester
    try:
        ingester = UnsplashIngester()
        
        # Fetch some test images
        images = ingester.fetch_random_images(query="urban traffic", count=5)
        print(f"Fetched {len(images)} images")
        
        if images:
            print(f"First image shape: {images[0].shape}")
            
            # Save locally for inspection
            saved = ingester.save_images_locally(images, "./data/realtime", "test_unsplash")
            print(f"Saved {len(saved)} images")
            
    except Exception as e:
        print(f"Test failed: {e}")
