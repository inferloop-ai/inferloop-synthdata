import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class JSONLExporter:
    """Export synthetic image datasets to JSONL format for ML training."""
    
    def __init__(self):
        self.exported_count = 0
    
    def export_dataset(self, 
                      dataset_dir: str,
                      output_file: str,
                      include_metadata: bool = True,
                      include_annotations: bool = True,
                      batch_size: int = 1000) -> Dict:
        """Export a complete dataset to JSONL format."""
        
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
        
        # Get all images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff']:
            image_files.extend(dataset_path.glob(ext))
        
        if not image_files:
            raise ValueError(f"No images found in {dataset_dir}")
        
        logger.info(f"Exporting {len(image_files)} images to JSONL...")
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        exported_records = []
        
        with jsonlines.open(output_file, mode='w') as writer:
            for i, image_file in enumerate(image_files):
                try:
                    record = self._create_image_record(
                        image_file, 
                        dataset_path,
                        include_metadata,
                        include_annotations
                    )
                    
                    writer.write(record)
                    exported_records.append(record)
                    
                    # Progress logging
                    if (i + 1) % batch_size == 0:
                        logger.info(f"Exported {i + 1}/{len(image_files)} images")
                        
                except Exception as e:
                    logger.error(f"Failed to export {image_file}: {e}")
                    continue
        
        self.exported_count = len(exported_records)
        
        # Create summary
        summary = {
            'export_timestamp': datetime.now().isoformat(),
            'total_images': len(image_files),
            'exported_images': self.exported_count,
            'output_file': str(output_path),
            'dataset_directory': str(dataset_path),
            'format': 'jsonl',
            'includes_metadata': include_metadata,
            'includes_annotations': include_annotations
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
                           include_annotations: bool) -> Dict:
        """Create a JSONL record for a single image."""
        
        relative_path = image_file.relative_to(dataset_path)
        
        record = {
            'id': str(image_file.stem),
            'file_path': str(relative_path),
            'file_name': image_file.name,
            'created_at': datetime.fromtimestamp(image_file.stat().st_mtime).isoformat()
        }
        
        # Add image metadata
        if include_metadata:
            metadata = self._extract_image_metadata(image_file)
            record['metadata'] = metadata
        
        # Add annotations if available
        if include_annotations:
            annotations = self._load_annotations(image_file, dataset_path)
            if annotations:
                record['annotations'] = annotations
        
        # Add generation info if available
        generation_info = self._load_generation_info(image_file, dataset_path)
        if generation_info:
            record['generation'] = generation_info
        
        return record
    
    def _extract_image_metadata(self, image_file: Path) -> Dict:
        """Extract metadata from image file."""
        
        from PIL import Image
        import os
        
        try:
            with Image.open(image_file) as img:
                metadata = {
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'file_size_bytes': os.path.getsize(image_file),
                    'aspect_ratio': round(img.width / img.height, 3)
                }
                
                # Extract EXIF data if available
                if hasattr(img, '_getexif') and img._getexif():
                    exif_data = img._getexif()
                    if exif_data:
                        metadata['exif'] = {str(k): str(v) for k, v in exif_data.items()}
                
                return metadata
                
        except Exception as e:
            logger.warning(f"Could not extract metadata from {image_file}: {e}")
            return {}
    
    def _load_annotations(self, image_file: Path, dataset_path: Path) -> Optional[Dict]:
        """Load annotations for the image if available."""
        
        # Check for COCO-style annotations
        annotation_file = dataset_path / "annotations" / f"{image_file.stem}.json"
        if annotation_file.exists():
            try:
                with open(annotation_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load annotations from {annotation_file}: {e}")
        
        # Check for YOLO-style annotations
        yolo_file = dataset_path / "labels" / f"{image_file.stem}.txt"
        if yolo_file.exists():
            try:
                annotations = {'format': 'yolo', 'labels': []}
                with open(yolo_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            annotations['labels'].append({
                                'class_id': int(parts[0]),
                                'x_center': float(parts[1]),
                                'y_center': float(parts[2]),
                                'width': float(parts[3]),
                                'height': float(parts[4])
                            })
                return annotations
            except Exception as e:
                logger.warning(f"Could not load YOLO annotations from {yolo_file}: {e}")
        
        return None
    
    def _load_generation_info(self, image_file: Path, dataset_path: Path) -> Optional[Dict]:
        """Load generation information if available."""
        
        # Check for generation metadata
        gen_file = dataset_path / "generation_info" / f"{image_file.stem}.json"
        if gen_file.exists():
            try:
                with open(gen_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load generation info from {gen_file}: {e}")
        
        # Try to extract from filename patterns
        filename = image_file.stem
        if '_' in filename:
            parts = filename.split('_')
            generation_info = {}
            
            # Look for common patterns
            for part in parts:
                if part.startswith('diff'):
                    generation_info['method'] = 'diffusion'
                elif part.startswith('gan'):
                    generation_info['method'] = 'gan'
                elif part.startswith('sim'):
                    generation_info['method'] = 'simulation'
                elif part.isdigit():
                    generation_info['sequence_number'] = int(part)
            
            return generation_info if generation_info else None
        
        return None

def export_to_jsonl(dataset_dir: str, 
                   output_file: str, 
                   **kwargs) -> Dict:
    """Convenience function for JSONL export."""
    
    exporter = JSONLExporter()
    return exporter.export_dataset(dataset_dir, output_file, **kwargs)

if __name__ == "__main__":
    # Example usage
    try:
        result = export_to_jsonl(
            dataset_dir="./data/generated/test_dataset",
            output_file="./exports/test_dataset.jsonl",
            include_metadata=True,
            include_annotations=True
        )
        print(f"Export completed: {result}")
    except Exception as e:
        print(f"Export failed: {e}")
