#!/usr/bin/env python3
"""
Ground Truth Generator for creating training data annotations
"""

import logging
import time
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError, ValidationError


class AnnotationType(Enum):
    """Types of annotations"""
    ENTITY = "entity"
    BOUNDING_BOX = "bounding_box"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    RELATIONSHIP = "relationship"
    LAYOUT = "layout"


class AnnotationFormat(Enum):
    """Annotation output formats"""
    COCO = "coco"
    YOLO = "yolo"
    PASCAL_VOC = "pascal_voc"
    LABELME = "labelme"
    CUSTOM_JSON = "custom_json"


@dataclass
class GroundTruthAnnotation:
    """Ground truth annotation"""
    id: str
    annotation_type: AnnotationType
    bbox: Optional[Tuple[int, int, int, int]] = None
    label: str = ""
    text: str = ""
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundTruthDocument:
    """Document with ground truth annotations"""
    document_id: str
    file_path: str
    width: int
    height: int
    annotations: List[GroundTruthAnnotation]
    page_num: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundTruthDataset:
    """Complete ground truth dataset"""
    dataset_name: str
    documents: List[GroundTruthDocument]
    classes: List[str]
    statistics: Dict[str, int] = field(default_factory=dict)
    creation_time: float = 0.0


class GroundTruthConfig(BaseModel):
    """Ground truth generator configuration"""
    
    # Generation settings
    output_format: AnnotationFormat = Field(AnnotationFormat.CUSTOM_JSON, description="Output annotation format")
    include_confidence: bool = Field(True, description="Include confidence scores")
    auto_validate: bool = Field(True, description="Auto-validate generated annotations")
    
    # Quality settings
    min_bbox_area: int = Field(100, description="Minimum bounding box area")
    max_overlap_ratio: float = Field(0.5, description="Maximum allowed overlap ratio")
    
    # Output settings
    output_directory: str = Field("ground_truth", description="Output directory")
    split_ratios: Dict[str, float] = Field(
        default_factory=lambda: {"train": 0.7, "val": 0.2, "test": 0.1},
        description="Dataset split ratios"
    )


class GroundTruthGenerator:
    """
    Ground Truth Generator for creating high-quality training annotations
    """
    
    def __init__(self, config: Optional[GroundTruthConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or GroundTruthConfig()
        
        # Ensure output directory exists
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Ground Truth Generator initialized")
    
    def generate_from_ocr(self, ocr_results, ner_results=None) -> GroundTruthDataset:
        """Generate ground truth from OCR and NER results"""
        start_time = time.time()
        
        try:
            documents = []
            all_classes = set()
            
            # Process OCR results
            if not isinstance(ocr_results, list):
                ocr_results = [ocr_results]
            
            for i, ocr_result in enumerate(ocr_results):
                # Generate document annotations
                doc = self._create_document_from_ocr(ocr_result, i)
                
                # Add NER annotations if available
                if ner_results and i < len(ner_results):
                    self._add_ner_annotations(doc, ner_results[i])
                
                # Validate annotations
                if self.config.auto_validate:
                    self._validate_document(doc)
                
                documents.append(doc)
                
                # Collect classes
                for ann in doc.annotations:
                    if ann.label:
                        all_classes.add(ann.label)
            
            # Create dataset
            dataset = GroundTruthDataset(
                dataset_name=f"generated_{int(time.time())}",
                documents=documents,
                classes=sorted(list(all_classes)),
                statistics=self._calculate_statistics(documents),
                creation_time=time.time() - start_time
            )
            
            self.logger.info(f"Generated ground truth dataset with {len(documents)} documents")
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"Ground truth generation failed: {e}")
            raise ProcessingError(f"Ground truth generation error: {e}")
    
    def _create_document_from_ocr(self, ocr_result, doc_index: int) -> GroundTruthDocument:
        """Create document annotations from OCR result"""
        annotations = []
        
        # Process each page
        for page in ocr_result.pages:
            # Create word-level annotations
            for word in page.words:
                if len(word.text.strip()) > 0:  # Skip empty words
                    annotation = GroundTruthAnnotation(
                        id=f"word_{page.page_num}_{word.word_num}",
                        annotation_type=AnnotationType.BOUNDING_BOX,
                        bbox=(word.bbox.x, word.bbox.y, 
                              word.bbox.x + word.bbox.width, 
                              word.bbox.y + word.bbox.height),
                        label="text",
                        text=word.text,
                        confidence=word.confidence / 100.0,
                        attributes={
                            'page_num': page.page_num,
                            'line_num': word.line_num,
                            'word_num': word.word_num
                        }
                    )
                    annotations.append(annotation)
            
            # Create line-level annotations
            for line in page.lines:
                if len(line.text.strip()) > 0:
                    annotation = GroundTruthAnnotation(
                        id=f"line_{page.page_num}_{line.line_num}",
                        annotation_type=AnnotationType.LAYOUT,
                        bbox=(line.bbox.x, line.bbox.y,
                              line.bbox.x + line.bbox.width,
                              line.bbox.y + line.bbox.height),
                        label="text_line",
                        text=line.text,
                        confidence=line.confidence / 100.0,
                        attributes={
                            'page_num': page.page_num,
                            'line_num': line.line_num,
                            'word_count': len(line.words)
                        }
                    )
                    annotations.append(annotation)
        
        # Get page dimensions (use first page)
        first_page = ocr_result.pages[0] if ocr_result.pages else None
        width = first_page.width if first_page else 0
        height = first_page.height if first_page else 0
        
        document = GroundTruthDocument(
            document_id=f"doc_{doc_index}",
            file_path=ocr_result.file_path,
            width=width,
            height=height,
            annotations=annotations,
            metadata={
                'total_pages': ocr_result.total_pages,
                'engine_used': ocr_result.engine_used,
                'processing_time': ocr_result.processing_time
            }
        )
        
        return document
    
    def _add_ner_annotations(self, document: GroundTruthDocument, ner_result):
        """Add NER annotations to document"""
        for entity in ner_result.entities:
            # Create entity annotation
            annotation = GroundTruthAnnotation(
                id=f"entity_{len(document.annotations)}",
                annotation_type=AnnotationType.ENTITY,
                label=entity.label.value,
                text=entity.text,
                confidence=entity.confidence,
                attributes={
                    'start': entity.start,
                    'end': entity.end,
                    'source': entity.source
                },
                metadata=entity.metadata
            )
            document.annotations.append(annotation)
    
    def _validate_document(self, document: GroundTruthDocument):
        """Validate document annotations"""
        valid_annotations = []
        
        for annotation in document.annotations:
            # Check bounding box validity
            if annotation.bbox:
                x1, y1, x2, y2 = annotation.bbox
                
                # Check bounds
                if (x1 >= 0 and y1 >= 0 and x2 <= document.width and y2 <= document.height
                    and x2 > x1 and y2 > y1):
                    
                    # Check minimum area
                    area = (x2 - x1) * (y2 - y1)
                    if area >= self.config.min_bbox_area:
                        valid_annotations.append(annotation)
                    else:
                        self.logger.debug(f"Removed annotation with small area: {area}")
                else:
                    self.logger.debug(f"Removed annotation with invalid bounds: {annotation.bbox}")
            else:
                # Non-bbox annotations are valid
                valid_annotations.append(annotation)
        
        document.annotations = valid_annotations
        self.logger.debug(f"Validated document: kept {len(valid_annotations)} annotations")
    
    def _calculate_statistics(self, documents: List[GroundTruthDocument]) -> Dict[str, int]:
        """Calculate dataset statistics"""
        stats = {
            'total_documents': len(documents),
            'total_annotations': 0,
            'annotation_types': {},
            'label_distribution': {}
        }
        
        for doc in documents:
            stats['total_annotations'] += len(doc.annotations)
            
            for ann in doc.annotations:
                # Count by type
                ann_type = ann.annotation_type.value
                if ann_type not in stats['annotation_types']:
                    stats['annotation_types'][ann_type] = 0
                stats['annotation_types'][ann_type] += 1
                
                # Count by label
                if ann.label:
                    if ann.label not in stats['label_distribution']:
                        stats['label_distribution'][ann.label] = 0
                    stats['label_distribution'][ann.label] += 1
        
        return stats
    
    def export_dataset(self, dataset: GroundTruthDataset, output_path: Optional[str] = None) -> str:
        """Export dataset to specified format"""
        try:
            if output_path is None:
                output_path = Path(self.config.output_directory) / f"{dataset.dataset_name}.json"
            else:
                output_path = Path(output_path)
            
            # Export based on format
            if self.config.output_format == AnnotationFormat.CUSTOM_JSON:
                self._export_custom_json(dataset, output_path)
            elif self.config.output_format == AnnotationFormat.COCO:
                self._export_coco_format(dataset, output_path)
            elif self.config.output_format == AnnotationFormat.YOLO:
                self._export_yolo_format(dataset, output_path)
            else:
                raise ProcessingError(f"Unsupported export format: {self.config.output_format}")
            
            self.logger.info(f"Dataset exported to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Dataset export failed: {e}")
            raise ProcessingError(f"Export error: {e}")
    
    def _export_custom_json(self, dataset: GroundTruthDataset, output_path: Path):
        """Export in custom JSON format"""
        export_data = {
            'dataset_name': dataset.dataset_name,
            'creation_time': dataset.creation_time,
            'statistics': dataset.statistics,
            'classes': dataset.classes,
            'documents': []
        }
        
        for doc in dataset.documents:
            doc_data = {
                'document_id': doc.document_id,
                'file_path': doc.file_path,
                'width': doc.width,
                'height': doc.height,
                'annotations': []
            }
            
            for ann in doc.annotations:
                ann_data = {
                    'id': ann.id,
                    'type': ann.annotation_type.value,
                    'label': ann.label,
                    'text': ann.text,
                    'bbox': ann.bbox,
                    'attributes': ann.attributes
                }
                
                if self.config.include_confidence:
                    ann_data['confidence'] = ann.confidence
                
                doc_data['annotations'].append(ann_data)
            
            export_data['documents'].append(doc_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def _export_coco_format(self, dataset: GroundTruthDataset, output_path: Path):
        """Export in COCO format"""
        coco_data = {
            'info': {
                'description': dataset.dataset_name,
                'version': '1.0',
                'year': 2024
            },
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Create categories
        for i, class_name in enumerate(dataset.classes):
            coco_data['categories'].append({
                'id': i + 1,
                'name': class_name,
                'supercategory': 'object'
            })
        
        class_to_id = {name: i + 1 for i, name in enumerate(dataset.classes)}
        annotation_id = 1
        
        # Process documents
        for doc_id, doc in enumerate(dataset.documents):
            # Add image info
            coco_data['images'].append({
                'id': doc_id + 1,
                'file_name': Path(doc.file_path).name,
                'width': doc.width,
                'height': doc.height
            })
            
            # Add annotations
            for ann in doc.annotations:
                if ann.bbox and ann.label in class_to_id:
                    x1, y1, x2, y2 = ann.bbox
                    width = x2 - x1
                    height = y2 - y1
                    
                    coco_data['annotations'].append({
                        'id': annotation_id,
                        'image_id': doc_id + 1,
                        'category_id': class_to_id[ann.label],
                        'bbox': [x1, y1, width, height],
                        'area': width * height,
                        'iscrowd': 0
                    })
                    annotation_id += 1
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)
    
    def _export_yolo_format(self, dataset: GroundTruthDataset, output_path: Path):
        """Export in YOLO format"""
        # Create output directory
        yolo_dir = output_path.parent / output_path.stem
        yolo_dir.mkdir(exist_ok=True)
        
        # Create classes file
        classes_file = yolo_dir / "classes.txt"
        with open(classes_file, 'w') as f:
            for class_name in dataset.classes:
                f.write(f"{class_name}\n")
        
        class_to_id = {name: i for i, name in enumerate(dataset.classes)}
        
        # Process each document
        for doc in dataset.documents:
            # Create annotation file
            ann_file = yolo_dir / f"{doc.document_id}.txt"
            
            with open(ann_file, 'w') as f:
                for ann in doc.annotations:
                    if ann.bbox and ann.label in class_to_id:
                        x1, y1, x2, y2 = ann.bbox
                        
                        # Convert to YOLO format (normalized center coordinates)
                        center_x = (x1 + x2) / 2 / doc.width
                        center_y = (y1 + y2) / 2 / doc.height
                        width = (x2 - x1) / doc.width
                        height = (y2 - y1) / doc.height
                        
                        class_id = class_to_id[ann.label]
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    def split_dataset(self, dataset: GroundTruthDataset) -> Dict[str, GroundTruthDataset]:
        """Split dataset into train/val/test sets"""
        total_docs = len(dataset.documents)
        train_count = int(total_docs * self.config.split_ratios['train'])
        val_count = int(total_docs * self.config.split_ratios['val'])
        
        # Shuffle documents for random split
        import random
        docs_copy = dataset.documents.copy()
        random.shuffle(docs_copy)
        
        # Create splits
        splits = {
            'train': GroundTruthDataset(
                dataset_name=f"{dataset.dataset_name}_train",
                documents=docs_copy[:train_count],
                classes=dataset.classes.copy(),
                statistics=self._calculate_statistics(docs_copy[:train_count])
            ),
            'val': GroundTruthDataset(
                dataset_name=f"{dataset.dataset_name}_val",
                documents=docs_copy[train_count:train_count + val_count],
                classes=dataset.classes.copy(),
                statistics=self._calculate_statistics(docs_copy[train_count:train_count + val_count])
            ),
            'test': GroundTruthDataset(
                dataset_name=f"{dataset.dataset_name}_test",
                documents=docs_copy[train_count + val_count:],
                classes=dataset.classes.copy(),
                statistics=self._calculate_statistics(docs_copy[train_count + val_count:])
            )
        }
        
        self.logger.info(f"Dataset split: train={len(splits['train'].documents)}, val={len(splits['val'].documents)}, test={len(splits['test'].documents)}")
        
        return splits


# Factory function
def create_ground_truth_generator(**config_kwargs) -> GroundTruthGenerator:
    """Factory function to create ground truth generator"""
    config = GroundTruthConfig(**config_kwargs)
    return GroundTruthGenerator(config)