import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
import json
import numpy as np

logger = logging.getLogger(__name__)

class ParquetExporter:
    """Export synthetic image datasets to Parquet format for efficient data processing."""
    
    def __init__(self):
        self.exported_count = 0
    
    def export_dataset(self, 
                      dataset_dir: str,
                      output_file: str,
                      include_metadata: bool = True,
                      include_features: bool = True,
                      compression: str = 'snappy',
                      row_group_size: int = 10000) -> Dict:
        """Export a complete dataset to Parquet format."""
        
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
        
        # Get all images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff']:
            image_files.extend(dataset_path.glob(ext))
        
        if not image_files:
            raise ValueError(f"No images found in {dataset_dir}")
        
        logger.info(f"Exporting {len(image_files)} images to Parquet...")
        
        # Collect all data
        records = []
        for i, image_file in enumerate(image_files):
            try:
                record = self._create_image_record(
                    image_file, 
                    dataset_path,
                    include_metadata,
                    include_features
                )
                records.append(record)
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_files)} images")
                    
            except Exception as e:
                logger.error(f"Failed to process {image_file}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Optimize data types
        df = self._optimize_datatypes(df)
        
        # Write to Parquet
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(
            output_file,
            compression=compression,
            index=False,
            row_group_size=row_group_size
        )
        
        self.exported_count = len(records)
        
        # Create summary
        summary = {
            'export_timestamp': datetime.now().isoformat(),
            'total_images': len(image_files),
            'exported_images': self.exported_count,
            'output_file': str(output_path),
            'dataset_directory': str(dataset_path),
            'format': 'parquet',
            'compression': compression,
            'file_size_mb': output_path.stat().st_size / (1024 * 1024),
            'schema': self._get_schema_info(df),
            'includes_metadata': include_metadata,
            'includes_features': include_features
        }
        
        # Save summary
        summary_file = output_path.parent / f"{output_path.stem}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Export complete: {self.exported_count} records written to {output_file}")
        
        return summary
    
    def _create_image_record(self, 
                           image_file: Path,
                           dataset_path: Path,
                           include_metadata: bool,
                           include_features: bool) -> Dict:
        """Create a record for a single image."""
        
        relative_path = image_file.relative_to(dataset_path)
        
        record = {
            'id': str(image_file.stem),
            'file_path': str(relative_path),
            'file_name': image_file.name,
            'created_at': datetime.fromtimestamp(image_file.stat().st_mtime)
        }
        
        # Add basic image metadata
        if include_metadata:
            metadata = self._extract_basic_metadata(image_file)
            record.update(metadata)
        
        # Add computed features
        if include_features:
            features = self._extract_features(image_file)
            record.update(features)
        
        # Add annotations as structured data
        annotations = self._load_structured_annotations(image_file, dataset_path)
        if annotations:
            record.update(annotations)
        
        return record
    
    def _extract_basic_metadata(self, image_file: Path) -> Dict:
        """Extract basic metadata optimized for Parquet storage."""
        
        from PIL import Image
        import os
        
        try:
            with Image.open(image_file) as img:
                return {
                    'width': img.width,
                    'height': img.height,
                    'channels': len(img.getbands()) if img.getbands() else 3,
                    'mode': img.mode,
                    'format': img.format,
                    'file_size_bytes': os.path.getsize(image_file),
                    'aspect_ratio': round(img.width / img.height, 3),
                    'resolution_category': self._categorize_resolution(img.width, img.height)
                }
                
        except Exception as e:
            logger.warning(f"Could not extract metadata from {image_file}: {e}")
            return {
                'width': None,
                'height': None,
                'channels': None,
                'mode': None,
                'format': None,
                'file_size_bytes': None,
                'aspect_ratio': None,
                'resolution_category': None
            }
    
    def _extract_features(self, image_file: Path) -> Dict:
        """Extract numerical features suitable for ML analysis."""
        
        try:
            import cv2
            
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                return {}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Basic statistics
            features = {
                'brightness_mean': float(np.mean(gray)),
                'brightness_std': float(np.std(gray)),
                'contrast_rms': float(np.sqrt(np.mean((gray - np.mean(gray)) ** 2))),
                'sharpness_laplacian': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                'entropy': self._calculate_entropy(gray),
            }
            
            # Color features (if color image)
            if len(image.shape) == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                features.update({
                    'saturation_mean': float(np.mean(hsv[:,:,1])),
                    'saturation_std': float(np.std(hsv[:,:,1])),
                    'hue_dominant': float(np.argmax(np.bincount(hsv[:,:,0].flatten()))),
                    'color_diversity': self._calculate_color_diversity(image)
                })
            
            # Texture features
            features.update(self._extract_texture_features(gray))
            
            return features
            
        except Exception as e:
            logger.warning(f"Could not extract features from {image_file}: {e}")
            return {}
    
    def _calculate_entropy(self, gray_image: np.ndarray) -> float:
        """Calculate Shannon entropy."""
        hist, _ = np.histogram(gray_image, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return float(entropy)
    
    def _calculate_color_diversity(self, image: np.ndarray) -> float:
        """Calculate color diversity score."""
        # Reshape and sample pixels for efficiency
        pixels = image.reshape(-1, 3)
        if len(pixels) > 10000:
            sample_indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[sample_indices]
        
        unique_colors = len(np.unique(pixels, axis=0))
        total_pixels = len(pixels)
        return float(unique_colors / total_pixels)
    
    def _extract_texture_features(self, gray_image: np.ndarray) -> Dict:
        """Extract basic texture features."""
        
        import cv2
        from skimage import feature
        
        try:
            # Edge density
            edges = cv2.Canny(gray_image, 100, 200)
            edge_density = float(np.sum(edges > 0) / edges.size)
            
            # Local Binary Pattern
            lbp = feature.local_binary_pattern(gray_image, 8, 1.5, method='uniform')
            lbp_variance = float(np.var(lbp))
            
            return {
                'edge_density': edge_density,
                'texture_variance': lbp_variance,
                'smoothness_score': float(1 - edge_density)
            }
            
        except Exception as e:
            logger.warning(f"Texture feature extraction failed: {e}")
            return {
                'edge_density': None,
                'texture_variance': None,
                'smoothness_score': None
            }
    
    def _load_structured_annotations(self, image_file: Path, dataset_path: Path) -> Dict:
        """Load annotations as structured columns."""
        
        annotations = {}
        
        # Load object detection annotations
        detection_file = dataset_path / "annotations" / f"{image_file.stem}.json"
        if detection_file.exists():
            try:
                with open(detection_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to structured format
                annotations.update({
                    'num_objects': len(data.get('objects', [])),
                    'object_classes': [obj.get('class', '') for obj in data.get('objects', [])],
                    'has_person': any(obj.get('class') == 'person' for obj in data.get('objects', [])),
                    'has_vehicle': any(obj.get('class') in ['car', 'truck', 'bus'] for obj in data.get('objects', [])),
                    'bbox_count': len([obj for obj in data.get('objects', []) if 'bbox' in obj])
                })
                
            except Exception as e:
                logger.warning(f"Could not load annotations from {detection_file}: {e}")
        
        return annotations
    
    def _categorize_resolution(self, width: int, height: int) -> str:
        """Categorize image resolution."""
        
        total_pixels = width * height
        
        if total_pixels >= 8000000:  # 8MP+
            return 'high'
        elif total_pixels >= 2000000:  # 2MP+
            return 'medium'
        elif total_pixels >= 500000:   # 0.5MP+
            return 'low'
        else:
            return 'very_low'
    
    def _optimize_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for efficient storage."""
        
        # Convert object columns with limited unique values to category
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')
        
        # Optimize integer columns
        for col in df.select_dtypes(include=['int64']):
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:  # Unsigned integers
                if col_max < 255:
                    df[col] = df[col].astype('uint8')
                elif col_max < 65535:
                    df[col] = df[col].astype('uint16')
                elif col_max < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:  # Signed integers
                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df[col] = df[col].astype('int32')
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float64']):
            if df[col].min() >= np.finfo(np.float32).min and df[col].max() <= np.finfo(np.float32).max:
                df[col] = df[col].astype('float32')
        
        return df
    
    def _get_schema_info(self, df: pd.DataFrame) -> Dict:
        """Get schema information for the DataFrame."""
        
        schema_info = {
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        return schema_info

def export_to_parquet(dataset_dir: str, 
                     output_file: str, 
                     **kwargs) -> Dict:
    """Convenience function for Parquet export."""
    
    exporter = ParquetExporter()
    return exporter.export_dataset(dataset_dir, output_file, **kwargs)

if __name__ == "__main__":
    # Example usage
    try:
        result = export_to_parquet(
            dataset_dir="./data/generated/test_dataset",
            output_file="./exports/test_dataset.parquet",
            include_metadata=True,
            include_features=True,
            compression='snappy'
        )
        print(f"Export completed: {result}")
    except Exception as e:
        print(f"Export failed: {e}")
