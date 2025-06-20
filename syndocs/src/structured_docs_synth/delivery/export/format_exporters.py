#!/usr/bin/env python3
"""
Format exporters for converting documents to various output formats.

Supports multiple export formats including COCO, YOLO, Pascal VOC, JSON,
CSV, and custom formats with annotation preservation and privacy protection.

Features:
- COCO format export with bounding boxes and annotations
- YOLO format export for object detection
- Pascal VOC XML format export
- JSON/JSONL export with structured data
- CSV export for tabular data
- Custom format support
- Privacy-preserving export options
- Batch processing capabilities
"""

import csv
import json
import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
import numpy as np

from ...core import get_logger, get_config
from ...core.exceptions import ValidationError, ProcessingError
from ...privacy import create_privacy_engine


logger = get_logger(__name__)
config = get_config()


class BaseExporter(ABC):
    """Base class for all format exporters"""
    
    def __init__(self, privacy_protection: bool = True):
        self.privacy_protection = privacy_protection
        self.privacy_engine = create_privacy_engine() if privacy_protection else None
        self.supported_formats = []
        
    @abstractmethod
    async def export(self, documents: List[Dict[str, Any]], output_path: str, 
                    options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Export documents to specified format"""
        pass
    
    @abstractmethod
    def validate_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Validate documents for export compatibility"""
        pass
    
    def apply_privacy_protection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply privacy protection to export data"""
        if not self.privacy_protection or not self.privacy_engine:
            return data
            
        try:
            return self.privacy_engine.protect_export_data(data)
        except Exception as e:
            logger.warning(f"Privacy protection failed: {e}")
            return data


class COCOExporter(BaseExporter):
    """COCO format exporter for object detection datasets"""
    
    def __init__(self, privacy_protection: bool = True):
        super().__init__(privacy_protection)
        self.supported_formats = ['coco']
        
    def validate_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Validate documents have required COCO fields"""
        required_fields = ['id', 'image_path', 'annotations']
        
        for doc in documents:
            for field in required_fields:
                if field not in doc:
                    return False
                    
            # Validate annotations structure
            if not isinstance(doc.get('annotations'), list):
                return False
                
            for ann in doc['annotations']:
                if not all(k in ann for k in ['bbox', 'category_id']):
                    return False
                    
        return True
    
    async def export(self, documents: List[Dict[str, Any]], output_path: str, 
                    options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Export documents to COCO format"""
        if not self.validate_documents(documents):
            raise ValidationError("Documents not compatible with COCO format")
            
        options = options or {}
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Build COCO structure
        coco_data = {
            "info": {
                "description": options.get('description', 'Synthetic Documents Dataset'),
                "url": options.get('url', ''),
                "version": options.get('version', '1.0'),
                "year": datetime.now().year,
                "contributor": options.get('contributor', 'Structured Docs Synth'),
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "MIT License",
                    "url": "https://opensource.org/licenses/MIT"
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Build categories from documents
        categories = set()
        for doc in documents:
            for ann in doc.get('annotations', []):
                if 'category_name' in ann:
                    categories.add(ann['category_name'])
        
        # Add categories to COCO data
        for i, category in enumerate(sorted(categories), 1):
            coco_data['categories'].append({
                'id': i,
                'name': category,
                'supercategory': options.get('supercategory', 'document')
            })
        
        # Create category name to ID mapping
        cat_name_to_id = {cat['name']: cat['id'] for cat in coco_data['categories']}
        
        annotation_id = 1
        
        # Process each document
        for doc in documents:
            # Get image info
            image_path = doc['image_path']
            
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                logger.warning(f"Could not get image dimensions for {image_path}: {e}")
                width, height = 1024, 768  # Default dimensions
            
            # Add image info
            image_info = {
                'id': doc['id'],
                'file_name': os.path.basename(image_path),
                'width': width,
                'height': height,
                'date_captured': doc.get('created_at', datetime.now().isoformat())
            }
            
            # Apply privacy protection
            if self.privacy_protection:
                image_info = self.apply_privacy_protection(image_info)
            
            coco_data['images'].append(image_info)
            
            # Add annotations
            for ann in doc.get('annotations', []):
                bbox = ann['bbox']  # [x, y, width, height]
                area = bbox[2] * bbox[3]
                
                annotation = {
                    'id': annotation_id,
                    'image_id': doc['id'],
                    'category_id': cat_name_to_id.get(ann.get('category_name'), 1),
                    'bbox': bbox,
                    'area': area,
                    'iscrowd': ann.get('iscrowd', 0),
                    'segmentation': ann.get('segmentation', [])
                }
                
                # Apply privacy protection
                if self.privacy_protection:
                    annotation = self.apply_privacy_protection(annotation)
                
                coco_data['annotations'].append(annotation)
                annotation_id += 1
        
        # Save COCO JSON file
        coco_file = output_path / 'annotations.json'
        with open(coco_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        # Copy images if specified
        if options.get('copy_images', True):
            images_dir = output_path / 'images'
            images_dir.mkdir(exist_ok=True)
            
            for doc in documents:
                image_path = Path(doc['image_path'])
                if image_path.exists():
                    dest_path = images_dir / image_path.name
                    if not dest_path.exists():
                        import shutil
                        shutil.copy2(image_path, dest_path)
        
        return {
            'format': 'coco',
            'output_path': str(output_path),
            'annotations_file': str(coco_file),
            'images_count': len(coco_data['images']),
            'annotations_count': len(coco_data['annotations']),
            'categories_count': len(coco_data['categories'])
        }


class YOLOExporter(BaseExporter):
    """YOLO format exporter for object detection"""
    
    def __init__(self, privacy_protection: bool = True):
        super().__init__(privacy_protection)
        self.supported_formats = ['yolo']
        
    def validate_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Validate documents have required YOLO fields"""
        required_fields = ['id', 'image_path', 'annotations']
        
        for doc in documents:
            for field in required_fields:
                if field not in doc:
                    return False
                    
            # Validate annotations structure
            if not isinstance(doc.get('annotations'), list):
                return False
                
            for ann in doc['annotations']:
                if not all(k in ann for k in ['bbox', 'category_id']):
                    return False
                    
        return True
    
    def convert_bbox_to_yolo(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height] normalized"""
        x, y, w, h = bbox
        
        # Convert to center coordinates and normalize
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width_norm = w / img_width
        height_norm = h / img_height
        
        return [x_center, y_center, width_norm, height_norm]
    
    async def export(self, documents: List[Dict[str, Any]], output_path: str, 
                    options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Export documents to YOLO format"""
        if not self.validate_documents(documents):
            raise ValidationError("Documents not compatible with YOLO format")
            
        options = options or {}
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        labels_dir = output_path / 'labels'
        images_dir = output_path / 'images'
        labels_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        
        # Build class names file
        classes = set()
        for doc in documents:
            for ann in doc.get('annotations', []):
                if 'category_name' in ann:
                    classes.add(ann['category_name'])
        
        class_list = sorted(classes)
        class_to_id = {name: i for i, name in enumerate(class_list)}
        
        # Save classes file
        with open(output_path / 'classes.txt', 'w') as f:
            for class_name in class_list:
                f.write(f"{class_name}\n")
        
        exported_count = 0
        
        # Process each document
        for doc in documents:
            image_path = Path(doc['image_path'])
            
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                logger.warning(f"Could not process image {image_path}: {e}")
                continue
            
            # Create label file
            label_filename = f"{doc['id']}.txt"
            label_path = labels_dir / label_filename
            
            with open(label_path, 'w') as f:
                for ann in doc.get('annotations', []):
                    category_name = ann.get('category_name', 'unknown')
                    class_id = class_to_id.get(category_name, 0)
                    
                    bbox = ann['bbox']
                    yolo_bbox = self.convert_bbox_to_yolo(bbox, img_width, img_height)
                    
                    # Apply privacy protection to coordinates if needed
                    if self.privacy_protection:
                        protected_data = self.apply_privacy_protection({
                            'bbox': yolo_bbox,
                            'class_id': class_id
                        })
                        yolo_bbox = protected_data.get('bbox', yolo_bbox)
                        class_id = protected_data.get('class_id', class_id)
                    
                    # Write YOLO format: class_id x_center y_center width height
                    f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
            
            # Copy image if specified
            if options.get('copy_images', True):
                dest_image_path = images_dir / f"{doc['id']}{image_path.suffix}"
                if image_path.exists() and not dest_image_path.exists():
                    import shutil
                    shutil.copy2(image_path, dest_image_path)
            
            exported_count += 1
        
        # Create dataset YAML file
        yaml_content = f"""# YOLO dataset configuration
train: {output_path / 'images'}
val: {output_path / 'images'}

nc: {len(class_list)}  # number of classes
names: {class_list}  # class names
"""
        
        with open(output_path / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)
        
        return {
            'format': 'yolo',
            'output_path': str(output_path),
            'labels_dir': str(labels_dir),
            'images_dir': str(images_dir),
            'classes_file': str(output_path / 'classes.txt'),
            'dataset_yaml': str(output_path / 'dataset.yaml'),
            'exported_count': exported_count,
            'classes_count': len(class_list)
        }


class PascalVOCExporter(BaseExporter):
    """Pascal VOC format exporter"""
    
    def __init__(self, privacy_protection: bool = True):
        super().__init__(privacy_protection)
        self.supported_formats = ['pascal_voc', 'voc']
        
    def validate_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Validate documents have required Pascal VOC fields"""
        required_fields = ['id', 'image_path', 'annotations']
        
        for doc in documents:
            for field in required_fields:
                if field not in doc:
                    return False
                    
            if not isinstance(doc.get('annotations'), list):
                return False
                
        return True
    
    def create_voc_xml(self, doc: Dict[str, Any], img_width: int, img_height: int) -> ET.Element:
        """Create Pascal VOC XML annotation"""
        annotation = ET.Element('annotation')
        
        # Folder
        folder = ET.SubElement(annotation, 'folder')
        folder.text = 'images'
        
        # Filename
        filename = ET.SubElement(annotation, 'filename')
        filename.text = os.path.basename(doc['image_path'])
        
        # Path
        path = ET.SubElement(annotation, 'path')
        path.text = doc['image_path']
        
        # Source
        source = ET.SubElement(annotation, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Synthetic Documents'
        
        # Size
        size = ET.SubElement(annotation, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(img_width)
        height = ET.SubElement(size, 'height')
        height.text = str(img_height)
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'
        
        # Segmented
        segmented = ET.SubElement(annotation, 'segmented')
        segmented.text = '0'
        
        # Objects
        for ann in doc.get('annotations', []):
            obj = ET.SubElement(annotation, 'object')
            
            name = ET.SubElement(obj, 'name')
            name.text = ann.get('category_name', 'unknown')
            
            pose = ET.SubElement(obj, 'pose')
            pose.text = 'Unspecified'
            
            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = str(ann.get('truncated', 0))
            
            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = str(ann.get('difficult', 0))
            
            # Bounding box
            bbox = ann['bbox']  # [x, y, width, height]
            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[0] + bbox[2])
            y_max = int(bbox[1] + bbox[3])
            
            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(x_min)
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(y_min)
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(x_max)
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(y_max)
        
        return annotation
    
    async def export(self, documents: List[Dict[str, Any]], output_path: str, 
                    options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Export documents to Pascal VOC format"""
        if not self.validate_documents(documents):
            raise ValidationError("Documents not compatible with Pascal VOC format")
            
        options = options or {}
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        annotations_dir = output_path / 'Annotations'
        images_dir = output_path / 'JPEGImages'
        annotations_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        
        exported_count = 0
        
        # Process each document
        for doc in documents:
            image_path = Path(doc['image_path'])
            
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                logger.warning(f"Could not process image {image_path}: {e}")
                continue
            
            # Create XML annotation
            xml_annotation = self.create_voc_xml(doc, img_width, img_height)
            
            # Apply privacy protection
            if self.privacy_protection:
                # Note: Privacy protection for XML would need custom implementation
                pass
            
            # Save XML file
            xml_filename = f"{doc['id']}.xml"
            xml_path = annotations_dir / xml_filename
            
            tree = ET.ElementTree(xml_annotation)
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            
            # Copy image if specified
            if options.get('copy_images', True):
                dest_image_path = images_dir / f"{doc['id']}.jpg"
                if image_path.exists() and not dest_image_path.exists():
                    import shutil
                    shutil.copy2(image_path, dest_image_path)
            
            exported_count += 1
        
        return {
            'format': 'pascal_voc',
            'output_path': str(output_path),
            'annotations_dir': str(annotations_dir),
            'images_dir': str(images_dir),
            'exported_count': exported_count
        }


class JSONExporter(BaseExporter):
    """JSON/JSONL format exporter for structured data"""
    
    def __init__(self, privacy_protection: bool = True):
        super().__init__(privacy_protection)
        self.supported_formats = ['json', 'jsonl']
        
    def validate_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Validate documents are JSON serializable"""
        try:
            json.dumps(documents)
            return True
        except (TypeError, ValueError):
            return False
    
    async def export(self, documents: List[Dict[str, Any]], output_path: str, 
                    options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Export documents to JSON/JSONL format"""
        if not self.validate_documents(documents):
            raise ValidationError("Documents not JSON serializable")
            
        options = options or {}
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        format_type = options.get('format', 'json')
        
        # Apply privacy protection
        if self.privacy_protection:
            documents = [self.apply_privacy_protection(doc) for doc in documents]
        
        if format_type == 'jsonl':
            # Export as JSONL (one JSON object per line)
            output_file = output_path / 'documents.jsonl'
            with open(output_file, 'w', encoding='utf-8') as f:
                for doc in documents:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        else:
            # Export as JSON array
            output_file = output_path / 'documents.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
        
        return {
            'format': format_type,
            'output_path': str(output_path),
            'output_file': str(output_file),
            'exported_count': len(documents)
        }


class CSVExporter(BaseExporter):
    """CSV format exporter for tabular data"""
    
    def __init__(self, privacy_protection: bool = True):
        super().__init__(privacy_protection)
        self.supported_formats = ['csv']
        
    def validate_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Validate documents can be flattened to CSV"""
        if not documents:
            return False
            
        # Check if all documents have consistent structure
        first_keys = set(self.flatten_dict(documents[0]).keys())
        for doc in documents[1:]:
            if set(self.flatten_dict(doc).keys()) != first_keys:
                return False
                
        return True
    
    def flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to string representation
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    async def export(self, documents: List[Dict[str, Any]], output_path: str, 
                    options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Export documents to CSV format"""
        if not self.validate_documents(documents):
            raise ValidationError("Documents not compatible with CSV format")
            
        options = options or {}
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Flatten all documents
        flattened_docs = []
        for doc in documents:
            flattened = self.flatten_dict(doc)
            if self.privacy_protection:
                flattened = self.apply_privacy_protection(flattened)
            flattened_docs.append(flattened)
        
        # Get all column names
        all_columns = set()
        for doc in flattened_docs:
            all_columns.update(doc.keys())
        
        columns = sorted(all_columns)
        
        # Export to CSV
        output_file = output_path / 'documents.csv'
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(flattened_docs)
        
        return {
            'format': 'csv',
            'output_path': str(output_path),
            'output_file': str(output_file),
            'exported_count': len(flattened_docs),
            'columns_count': len(columns)
        }


# Format exporter factory
class FormatExporterFactory:
    """Factory for creating format exporters"""
    
    _exporters = {
        'coco': COCOExporter,
        'yolo': YOLOExporter,
        'pascal_voc': PascalVOCExporter,
        'voc': PascalVOCExporter,
        'json': JSONExporter,
        'jsonl': JSONExporter,
        'csv': CSVExporter
    }
    
    @classmethod
    def create_exporter(cls, format_type: str, privacy_protection: bool = True) -> BaseExporter:
        """Create exporter for specified format"""
        if format_type not in cls._exporters:
            raise ValueError(f"Unsupported export format: {format_type}")
            
        exporter_class = cls._exporters[format_type]
        return exporter_class(privacy_protection=privacy_protection)
    
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """Get list of supported export formats"""
        return list(cls._exporters.keys())
    
    @classmethod
    def register_exporter(cls, format_type: str, exporter_class: type):
        """Register custom exporter"""
        if not issubclass(exporter_class, BaseExporter):
            raise ValueError("Exporter must inherit from BaseExporter")
            
        cls._exporters[format_type] = exporter_class


# Factory functions
def create_format_exporter(format_type: str, privacy_protection: bool = True) -> BaseExporter:
    """Create format exporter"""
    return FormatExporterFactory.create_exporter(format_type, privacy_protection)


def get_supported_export_formats() -> List[str]:
    """Get list of supported export formats"""
    return FormatExporterFactory.get_supported_formats()


__all__ = [
    'BaseExporter',
    'COCOExporter',
    'YOLOExporter', 
    'PascalVOCExporter',
    'JSONExporter',
    'CSVExporter',
    'FormatExporterFactory',
    'create_format_exporter',
    'get_supported_export_formats'
]