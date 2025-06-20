#!/usr/bin/env python3
"""
Dataset Loader for external datasets (DocBank, FUNSD, CORD, etc.)
"""

import json
import logging
import time
import zipfile
import requests
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError, ValidationError


class DatasetFormat(Enum):
    """Supported dataset formats"""
    DOCBANK = "docbank"
    FUNSD = "funsd"
    CORD = "cord"
    PUBLAYNET = "publaynet"
    KLEISTER = "kleister"
    XFUND = "xfund"
    WILDRECEIPT = "wildreceipt"
    COCO = "coco"
    PASCAL_VOC = "pascal_voc"
    CUSTOM_JSON = "custom_json"


class DatasetSource(Enum):
    """Dataset source types"""
    HUGGINGFACE = "huggingface"
    GITHUB = "github" 
    HTTP_URL = "http_url"
    LOCAL_PATH = "local_path"
    S3_BUCKET = "s3_bucket"
    GCS_BUCKET = "gcs_bucket"


@dataclass
class DatasetMetadata:
    """Dataset metadata information"""
    name: str
    format: DatasetFormat
    source: DatasetSource
    version: str = "latest"
    description: str = ""
    size_mb: float = 0.0
    num_documents: int = 0
    num_annotations: int = 0
    domains: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: ["en"])
    license: str = ""
    citation: str = ""


@dataclass
class DocumentRecord:
    """Individual document record from dataset"""
    id: str
    file_path: str
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    text_content: str = ""
    image_path: Optional[str] = None
    ground_truth: Optional[Dict[str, Any]] = None


@dataclass
class LoadedDataset:
    """Complete loaded dataset"""
    metadata: DatasetMetadata
    documents: List[DocumentRecord]
    train_split: List[str] = field(default_factory=list)  # Document IDs
    val_split: List[str] = field(default_factory=list)
    test_split: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)


class DatasetLoaderConfig(BaseModel):
    """Dataset loader configuration"""
    
    # Download settings
    cache_dir: str = Field("./data/cache", description="Cache directory for downloads")
    max_download_size_mb: float = Field(1000.0, description="Maximum download size in MB")
    download_timeout: int = Field(300, description="Download timeout in seconds")
    
    # Processing settings
    max_documents: Optional[int] = Field(None, description="Maximum documents to load")
    skip_images: bool = Field(False, description="Skip loading image files")
    extract_text: bool = Field(True, description="Extract text from documents")
    
    # Validation settings
    validate_annotations: bool = Field(True, description="Validate annotation format")
    strict_mode: bool = Field(False, description="Strict validation mode")
    
    # Filtering
    allowed_formats: List[str] = Field(
        default=["pdf", "png", "jpg", "jpeg", "tiff"],
        description="Allowed file formats"
    )
    min_file_size_kb: float = Field(1.0, description="Minimum file size in KB")


class DatasetLoader:
    """
    Dataset Loader for external structured document datasets
    
    Supports popular datasets like DocBank, FUNSD, CORD, PubLayNet, etc.
    Handles downloading, caching, parsing, and validation.
    """
    
    def __init__(self, config: Optional[DatasetLoaderConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or DatasetLoaderConfig()
        
        # Create cache directory
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset registry
        self.dataset_registry = self._initialize_dataset_registry()
        
        self.logger.info(f"Dataset Loader initialized with cache: {self.cache_dir}")
    
    def _initialize_dataset_registry(self) -> Dict[str, Dict[str, Any]]:
        """Initialize registry of known datasets"""
        return {
            "docbank": {
                "format": DatasetFormat.DOCBANK,
                "source": DatasetSource.GITHUB,
                "url": "https://github.com/doc-analysis/DocBank",
                "description": "500K document pages with layout annotations",
                "domains": ["academic", "general"],
                "size_mb": 2500.0
            },
            "funsd": {
                "format": DatasetFormat.FUNSD,
                "source": DatasetSource.GITHUB,
                "url": "https://guillaumejaume.github.io/FUNSD/",
                "description": "199 real, fully annotated, scanned forms",
                "domains": ["forms"],
                "size_mb": 15.0
            },
            "cord": {
                "format": DatasetFormat.CORD,
                "source": DatasetSource.GITHUB,
                "url": "https://github.com/clovaai/cord",
                "description": "11K receipt images with annotations",
                "domains": ["receipts", "retail"],
                "size_mb": 150.0
            },
            "publaynet": {
                "format": DatasetFormat.PUBLAYNET,
                "source": DatasetSource.GITHUB,
                "url": "https://github.com/ibm-aur-nlp/PubLayNet",
                "description": "360K+ document images for layout analysis",
                "domains": ["academic", "publications"],
                "size_mb": 10000.0
            },
            "kleister": {
                "format": DatasetFormat.KLEISTER,
                "source": DatasetSource.GITHUB,
                "url": "https://github.com/applicaai/kleister-nda",
                "description": "Legal document understanding dataset",
                "domains": ["legal"],
                "size_mb": 200.0
            }
        }
    
    def list_available_datasets(self) -> List[DatasetMetadata]:
        """List all available datasets"""
        datasets = []
        for name, info in self.dataset_registry.items():
            metadata = DatasetMetadata(
                name=name,
                format=info["format"],
                source=info["source"],
                description=info["description"],
                domains=info["domains"],
                size_mb=info["size_mb"]
            )
            datasets.append(metadata)
        return datasets
    
    def get_available_datasets(self) -> Dict[str, Any]:
        """Get available datasets as dictionary"""
        return {name: info for name, info in self.dataset_registry.items()}
    
    def load_dataset(self, dataset_name: str, source_path: Optional[str] = None,
                    split: Optional[str] = None) -> LoadedDataset:
        """Load a dataset by name or from custom path"""
        start_time = time.time()
        
        try:
            if dataset_name in self.dataset_registry:
                # Load known dataset
                dataset_info = self.dataset_registry[dataset_name]
                dataset = self._load_known_dataset(dataset_name, dataset_info, split)
            elif source_path:
                # Load custom dataset
                dataset = self._load_custom_dataset(dataset_name, source_path, split)
            else:
                raise ValueError(f"Unknown dataset '{dataset_name}' and no source_path provided")
            
            # Calculate statistics
            dataset.statistics = self._calculate_dataset_statistics(dataset)
            
            loading_time = time.time() - start_time
            self.logger.info(f"Loaded dataset '{dataset_name}' with {len(dataset.documents)} documents in {loading_time:.2f}s")
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset '{dataset_name}': {e}")
            raise ProcessingError(f"Dataset loading error: {e}")
    
    def _load_known_dataset(self, name: str, info: Dict[str, Any], 
                           split: Optional[str] = None) -> LoadedDataset:
        """Load a dataset from the known registry"""
        dataset_format = info["format"]
        
        # Download if not cached
        dataset_path = self._ensure_dataset_downloaded(name, info)
        
        # Parse dataset based on format
        if dataset_format == DatasetFormat.DOCBANK:
            return self._parse_docbank_dataset(name, dataset_path, split)
        elif dataset_format == DatasetFormat.FUNSD:
            return self._parse_funsd_dataset(name, dataset_path, split)
        elif dataset_format == DatasetFormat.CORD:
            return self._parse_cord_dataset(name, dataset_path, split)
        elif dataset_format == DatasetFormat.PUBLAYNET:
            return self._parse_publaynet_dataset(name, dataset_path, split)
        elif dataset_format == DatasetFormat.KLEISTER:
            return self._parse_kleister_dataset(name, dataset_path, split)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")
    
    def _load_custom_dataset(self, name: str, source_path: str, 
                           split: Optional[str] = None) -> LoadedDataset:
        """Load a custom dataset from provided path"""
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {source_path}")
        
        # Try to detect format from structure
        detected_format = self._detect_dataset_format(source_path)
        
        metadata = DatasetMetadata(
            name=name,
            format=detected_format,
            source=DatasetSource.LOCAL_PATH,
            description=f"Custom dataset from {source_path}"
        )
        
        # Parse based on detected format
        if detected_format == DatasetFormat.COCO:
            return self._parse_coco_dataset(metadata, source_path, split)
        elif detected_format == DatasetFormat.CUSTOM_JSON:
            return self._parse_custom_json_dataset(metadata, source_path, split)
        else:
            return self._parse_generic_dataset(metadata, source_path, split)
    
    def _ensure_dataset_downloaded(self, name: str, info: Dict[str, Any]) -> Path:
        """Ensure dataset is downloaded and cached"""
        cache_path = self.cache_dir / name
        
        if cache_path.exists():
            self.logger.debug(f"Using cached dataset: {cache_path}")
            return cache_path
        
        self.logger.info(f"Downloading dataset '{name}'...")
        
        # Create download directory
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # For this implementation, we'll create placeholder structure
        # In practice, you'd implement actual downloading from URLs
        self._create_placeholder_dataset(cache_path, name, info)
        
        return cache_path
    
    def _create_placeholder_dataset(self, path: Path, name: str, info: Dict[str, Any]):
        """Create placeholder dataset structure for testing"""
        # Create basic structure based on dataset type
        if info["format"] == DatasetFormat.FUNSD:
            # FUNSD structure
            (path / "training_data").mkdir(exist_ok=True)
            (path / "testing_data").mkdir(exist_ok=True)
            
            # Create sample annotation file
            sample_annotation = {
                "form": [
                    {
                        "id": 0,
                        "text": "Sample Form",
                        "box": [100, 100, 200, 120],
                        "label": "header",
                        "words": [{"text": "Sample", "box": [100, 100, 140, 120]}, 
                                {"text": "Form", "box": [150, 100, 200, 120]}]
                    }
                ]
            }
            
            with open(path / "training_data" / "sample.json", "w") as f:
                json.dump(sample_annotation, f)
        
        elif info["format"] == DatasetFormat.CORD:
            # CORD structure
            (path / "train").mkdir(exist_ok=True)
            (path / "test").mkdir(exist_ok=True)
            
            # Create sample receipt annotation
            sample_receipt = {
                "meta": {"version": "v1.0", "split": "train"},
                "valid_line": [
                    {"category": "menu.nm", "group_id": 0, "text": "Coffee", 
                     "words": [{"quad": {"x1": 100, "y1": 100, "x2": 150, "y2": 120}, "text": "Coffee"}]}
                ]
            }
            
            with open(path / "train" / "sample.json", "w") as f:
                json.dump(sample_receipt, f)
    
    def _parse_funsd_dataset(self, name: str, dataset_path: Path, 
                           split: Optional[str] = None) -> LoadedDataset:
        """Parse FUNSD format dataset"""
        metadata = DatasetMetadata(
            name=name,
            format=DatasetFormat.FUNSD,
            source=DatasetSource.LOCAL_PATH,
            description="FUNSD - Form Understanding in Noisy Scanned Documents"
        )
        
        documents = []
        
        # Load training data
        train_path = dataset_path / "training_data"
        if train_path.exists():
            train_docs = self._load_funsd_split(train_path, "train")
            documents.extend(train_docs)
        
        # Load testing data
        test_path = dataset_path / "testing_data"
        if test_path.exists():
            test_docs = self._load_funsd_split(test_path, "test")
            documents.extend(test_docs)
        
        # Create splits
        train_ids = [doc.id for doc in documents if doc.metadata.get("split") == "train"]
        test_ids = [doc.id for doc in documents if doc.metadata.get("split") == "test"]
        
        return LoadedDataset(
            metadata=metadata,
            documents=documents,
            train_split=train_ids,
            test_split=test_ids
        )
    
    def _load_funsd_split(self, split_path: Path, split_name: str) -> List[DocumentRecord]:
        """Load documents from a FUNSD split directory"""
        documents = []
        
        for json_file in split_path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Extract annotations
                annotations = []
                text_content = ""
                
                for form_item in data.get("form", []):
                    annotation = {
                        "id": form_item["id"],
                        "text": form_item["text"],
                        "bbox": form_item["box"],
                        "label": form_item["label"],
                        "words": form_item.get("words", [])
                    }
                    annotations.append(annotation)
                    text_content += form_item["text"] + " "
                
                # Create document record
                doc = DocumentRecord(
                    id=json_file.stem,
                    file_path=str(json_file),
                    annotations=annotations,
                    text_content=text_content.strip(),
                    metadata={
                        "split": split_name,
                        "format": "funsd",
                        "num_annotations": len(annotations)
                    }
                )
                
                documents.append(doc)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse {json_file}: {e}")
        
        return documents
    
    def _parse_cord_dataset(self, name: str, dataset_path: Path, 
                          split: Optional[str] = None) -> LoadedDataset:
        """Parse CORD format dataset"""
        metadata = DatasetMetadata(
            name=name,
            format=DatasetFormat.CORD,
            source=DatasetSource.LOCAL_PATH,
            description="CORD - Consolidated Receipt Dataset"
        )
        
        documents = []
        
        # Load train split
        train_path = dataset_path / "train"
        if train_path.exists():
            train_docs = self._load_cord_split(train_path, "train")
            documents.extend(train_docs)
        
        # Load test split
        test_path = dataset_path / "test"
        if test_path.exists():
            test_docs = self._load_cord_split(test_path, "test")
            documents.extend(test_docs)
        
        # Create splits
        train_ids = [doc.id for doc in documents if doc.metadata.get("split") == "train"]
        test_ids = [doc.id for doc in documents if doc.metadata.get("split") == "test"]
        
        return LoadedDataset(
            metadata=metadata,
            documents=documents,
            train_split=train_ids,
            test_split=test_ids
        )
    
    def _load_cord_split(self, split_path: Path, split_name: str) -> List[DocumentRecord]:
        """Load documents from a CORD split directory"""
        documents = []
        
        for json_file in split_path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Extract valid line annotations
                annotations = []
                text_content = ""
                
                for line in data.get("valid_line", []):
                    annotation = {
                        "category": line["category"],
                        "group_id": line["group_id"],
                        "text": line["text"],
                        "words": line.get("words", [])
                    }
                    annotations.append(annotation)
                    text_content += line["text"] + " "
                
                # Create document record
                doc = DocumentRecord(
                    id=json_file.stem,
                    file_path=str(json_file),
                    annotations=annotations,
                    text_content=text_content.strip(),
                    metadata={
                        "split": split_name,
                        "format": "cord",
                        "num_annotations": len(annotations),
                        "meta": data.get("meta", {})
                    }
                )
                
                documents.append(doc)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse {json_file}: {e}")
        
        return documents
    
    def _parse_docbank_dataset(self, name: str, dataset_path: Path, 
                             split: Optional[str] = None) -> LoadedDataset:
        """Parse DocBank format dataset"""
        metadata = DatasetMetadata(
            name=name,
            format=DatasetFormat.DOCBANK,
            source=DatasetSource.LOCAL_PATH,
            description="DocBank: Large-scale dataset for document layout analysis"
        )
        
        documents = []
        
        # DocBank structure: txt_files/ and json_files/ directories
        txt_dir = dataset_path / "txt_files"
        json_dir = dataset_path / "json_files"
        
        if not txt_dir.exists() or not json_dir.exists():
            self.logger.warning(f"DocBank dataset missing required directories at {dataset_path}")
            return LoadedDataset(metadata=metadata, documents=[])
        
        # Process each document
        for json_file in json_dir.glob("*.json"):
            try:
                doc_id = json_file.stem
                txt_file = txt_dir / f"{doc_id}.txt"
                
                if not txt_file.exists():
                    continue
                    
                # Load text content
                text_content = txt_file.read_text(encoding='utf-8')
                
                # Load annotations
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                annotations = []
                for token in json_data.get('tokens', []):
                    bbox = token.get('bbox', [])
                    if len(bbox) == 4:
                        annotations.append(Annotation(
                            id=f"{doc_id}_{len(annotations)}",
                            type=token.get('label', 'text'),
                            text=token.get('text', ''),
                            bbox=bbox,
                            metadata={
                                'font': token.get('font', {}),
                                'color': token.get('color', ''),
                                'page': token.get('page', 0)
                            }
                        ))
                
                doc = DocumentRecord(
                    id=doc_id,
                    name=f"{doc_id}.pdf",
                    path=str(json_file),
                    annotations=annotations,
                    metadata={
                        'text_content': text_content[:1000],  # Store sample
                        'token_count': len(json_data.get('tokens', [])),
                        'page_count': json_data.get('page_count', 1)
                    }
                )
                
                documents.append(doc)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse DocBank document {json_file}: {e}")
        
        dataset = LoadedDataset(metadata=metadata, documents=documents)
        
        # Apply splits if specified
        if split:
            dataset = self._apply_split(dataset, split)
            
        # Calculate statistics
        dataset.statistics = self._calculate_dataset_statistics(dataset)
        
        return dataset
    
    def _parse_publaynet_dataset(self, name: str, dataset_path: Path, 
                               split: Optional[str] = None) -> LoadedDataset:
        """Parse PubLayNet format dataset"""
        metadata = DatasetMetadata(
            name=name,
            format=DatasetFormat.PUBLAYNET,
            source=DatasetSource.LOCAL_PATH,
            description="PubLayNet: Large-scale dataset for document layout analysis in scientific articles"
        )
        
        documents = []
        
        # PubLayNet uses COCO format
        annotation_files = list(dataset_path.glob("*.json"))
        if not annotation_files:
            self.logger.warning(f"No annotation files found in {dataset_path}")
            return LoadedDataset(metadata=metadata, documents=[])
        
        for ann_file in annotation_files:
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                
                # Create lookup for images
                images_by_id = {img['id']: img for img in coco_data.get('images', [])}
                
                # Create lookup for categories
                categories_by_id = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
                
                # Group annotations by image
                annotations_by_image = {}
                for ann in coco_data.get('annotations', []):
                    img_id = ann['image_id']
                    if img_id not in annotations_by_image:
                        annotations_by_image[img_id] = []
                    annotations_by_image[img_id].append(ann)
                
                # Process each image
                for img_id, img_info in images_by_id.items():
                    annotations = []
                    
                    for ann in annotations_by_image.get(img_id, []):
                        # Convert COCO bbox format [x, y, width, height] to [x1, y1, x2, y2]
                        bbox = ann['bbox']
                        x1, y1, w, h = bbox
                        x2, y2 = x1 + w, y1 + h
                        
                        annotations.append(Annotation(
                            id=str(ann['id']),
                            type=categories_by_id.get(ann['category_id'], 'unknown'),
                            text='',  # PubLayNet doesn't include text
                            bbox=[x1, y1, x2, y2],
                            metadata={
                                'area': ann.get('area', 0),
                                'segmentation': ann.get('segmentation', []),
                                'iscrowd': ann.get('iscrowd', 0)
                            }
                        ))
                    
                    doc = DocumentRecord(
                        id=str(img_id),
                        name=img_info['file_name'],
                        path=str(dataset_path / img_info['file_name']),
                        annotations=annotations,
                        metadata={
                            'width': img_info['width'],
                            'height': img_info['height'],
                            'split': ann_file.stem  # train, val, or test
                        }
                    )
                    
                    documents.append(doc)
                    
            except Exception as e:
                self.logger.warning(f"Failed to parse PubLayNet file {ann_file}: {e}")
        
        dataset = LoadedDataset(metadata=metadata, documents=documents)
        
        # Apply splits
        for doc in documents:
            split_name = doc.metadata.get('split', 'train')
            if 'train' in split_name:
                dataset.train_split.append(doc.id)
            elif 'val' in split_name:
                dataset.val_split.append(doc.id)
            elif 'test' in split_name:
                dataset.test_split.append(doc.id)
        
        # Calculate statistics
        dataset.statistics = self._calculate_dataset_statistics(dataset)
        
        return dataset
    
    def _parse_kleister_dataset(self, name: str, dataset_path: Path, 
                              split: Optional[str] = None) -> LoadedDataset:
        """Parse Kleister format dataset"""
        metadata = DatasetMetadata(
            name=name,
            format=DatasetFormat.KLEISTER,
            source=DatasetSource.LOCAL_PATH,
            description="Kleister: Dataset for information extraction from long documents"
        )
        
        documents = []
        
        # Kleister structure: documents/, ground_truth/, and optionally OCR/
        docs_dir = dataset_path / "documents"
        gt_dir = dataset_path / "ground_truth"
        ocr_dir = dataset_path / "OCR"
        
        if not docs_dir.exists() or not gt_dir.exists():
            self.logger.warning(f"Kleister dataset missing required directories at {dataset_path}")
            return LoadedDataset(metadata=metadata, documents=[])
        
        # Process each document
        for pdf_file in docs_dir.glob("*.pdf"):
            try:
                doc_id = pdf_file.stem
                
                # Load ground truth annotations
                gt_file = gt_dir / f"{doc_id}.tsv"
                annotations = []
                
                if gt_file.exists():
                    with open(gt_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for idx, line in enumerate(lines):
                            if line.strip():
                                parts = line.strip().split('\t')
                                if len(parts) >= 2:
                                    field_name = parts[0]
                                    field_value = parts[1]
                                    
                                    annotations.append(Annotation(
                                        id=f"{doc_id}_field_{idx}",
                                        type=field_name,
                                        text=field_value,
                                        bbox=[0, 0, 0, 0],  # Kleister doesn't provide bbox
                                        metadata={
                                            'extraction_type': 'key-value',
                                            'confidence': 1.0
                                        }
                                    ))
                
                # Load OCR text if available
                ocr_text = ""
                ocr_file = ocr_dir / f"{doc_id}.txt"
                if ocr_file.exists():
                    ocr_text = ocr_file.read_text(encoding='utf-8')
                
                doc = DocumentRecord(
                    id=doc_id,
                    name=pdf_file.name,
                    path=str(pdf_file),
                    annotations=annotations,
                    metadata={
                        'has_ocr': ocr_file.exists(),
                        'ocr_text_sample': ocr_text[:500] if ocr_text else '',
                        'field_count': len(annotations)
                    }
                )
                
                documents.append(doc)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse Kleister document {pdf_file}: {e}")
        
        dataset = LoadedDataset(metadata=metadata, documents=documents)
        
        # Kleister typically uses predefined splits
        if (dataset_path / "train.txt").exists():
            train_ids = (dataset_path / "train.txt").read_text().strip().split('\n')
            dataset.train_split = [doc_id.strip() for doc_id in train_ids if doc_id.strip()]
        
        if (dataset_path / "dev.txt").exists():
            val_ids = (dataset_path / "dev.txt").read_text().strip().split('\n')
            dataset.val_split = [doc_id.strip() for doc_id in val_ids if doc_id.strip()]
        
        if (dataset_path / "test.txt").exists():
            test_ids = (dataset_path / "test.txt").read_text().strip().split('\n')
            dataset.test_split = [doc_id.strip() for doc_id in test_ids if doc_id.strip()]
        
        # Calculate statistics
        dataset.statistics = self._calculate_dataset_statistics(dataset)
        
        return dataset
    
    def _detect_dataset_format(self, dataset_path: Path) -> DatasetFormat:
        """Detect dataset format from directory structure"""
        if (dataset_path / "annotations.json").exists():
            return DatasetFormat.COCO
        elif any(dataset_path.glob("*.json")):
            return DatasetFormat.CUSTOM_JSON
        else:
            return DatasetFormat.CUSTOM_JSON
    
    def _parse_coco_dataset(self, metadata: DatasetMetadata, dataset_path: Path, 
                          split: Optional[str] = None) -> LoadedDataset:
        """Parse COCO format dataset"""
        documents = []
        
        # Look for annotations.json or train.json, val.json, test.json
        annotation_files = list(dataset_path.glob("*.json"))
        
        for ann_file in annotation_files:
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                
                # Process images and annotations
                images_by_id = {img['id']: img for img in coco_data.get('images', [])}
                categories_by_id = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
                
                # Group annotations by image
                annotations_by_image = {}
                for ann in coco_data.get('annotations', []):
                    img_id = ann['image_id']
                    if img_id not in annotations_by_image:
                        annotations_by_image[img_id] = []
                    annotations_by_image[img_id].append(ann)
                
                # Create document records
                for img_id, img_info in images_by_id.items():
                    annotations = []
                    
                    for ann in annotations_by_image.get(img_id, []):
                        bbox = ann.get('bbox', [0, 0, 0, 0])
                        if len(bbox) == 4:
                            x, y, w, h = bbox
                            bbox = [x, y, x + w, y + h]
                        
                        annotations.append(Annotation(
                            id=str(ann['id']),
                            type=categories_by_id.get(ann['category_id'], 'unknown'),
                            text=ann.get('caption', ''),
                            bbox=bbox,
                            metadata=ann
                        ))
                    
                    doc = DocumentRecord(
                        id=str(img_id),
                        name=img_info['file_name'],
                        path=str(dataset_path / 'images' / img_info['file_name']),
                        annotations=annotations,
                        metadata=img_info
                    )
                    
                    documents.append(doc)
                    
            except Exception as e:
                self.logger.warning(f"Failed to parse COCO file {ann_file}: {e}")
        
        dataset = LoadedDataset(metadata=metadata, documents=documents)
        dataset.statistics = self._calculate_dataset_statistics(dataset)
        
        return dataset
    
    def _parse_custom_json_dataset(self, metadata: DatasetMetadata, dataset_path: Path, 
                                 split: Optional[str] = None) -> LoadedDataset:
        """Parse custom JSON format dataset"""
        documents = []
        
        # Process all JSON files in the directory
        for json_file in dataset_path.glob("**/*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    # Array of documents
                    for idx, doc_data in enumerate(data):
                        doc = self._parse_custom_document(f"{json_file.stem}_{idx}", doc_data, json_file)
                        if doc:
                            documents.append(doc)
                elif isinstance(data, dict):
                    # Single document or collection
                    if 'documents' in data:
                        # Collection of documents
                        for idx, doc_data in enumerate(data['documents']):
                            doc = self._parse_custom_document(f"{json_file.stem}_{idx}", doc_data, json_file)
                            if doc:
                                documents.append(doc)
                    else:
                        # Single document
                        doc = self._parse_custom_document(json_file.stem, data, json_file)
                        if doc:
                            documents.append(doc)
                            
            except Exception as e:
                self.logger.warning(f"Failed to parse custom JSON {json_file}: {e}")
        
        dataset = LoadedDataset(metadata=metadata, documents=documents)
        dataset.statistics = self._calculate_dataset_statistics(dataset)
        
        return dataset
    
    def _parse_custom_document(self, doc_id: str, data: Dict[str, Any], source_file: Path) -> Optional[DocumentRecord]:
        """Parse a custom document from JSON data"""
        try:
            # Extract common fields
            annotations = []
            
            # Look for annotations in various possible locations
            ann_data = data.get('annotations', data.get('labels', data.get('entities', [])))
            
            if isinstance(ann_data, list):
                for idx, ann in enumerate(ann_data):
                    annotation = Annotation(
                        id=f"{doc_id}_ann_{idx}",
                        type=ann.get('type', ann.get('label', ann.get('category', 'unknown'))),
                        text=ann.get('text', ann.get('value', '')),
                        bbox=ann.get('bbox', ann.get('box', ann.get('coordinates', [0, 0, 0, 0]))),
                        metadata=ann
                    )
                    annotations.append(annotation)
            
            return DocumentRecord(
                id=doc_id,
                name=data.get('name', data.get('filename', f"{doc_id}.json")),
                path=str(source_file),
                annotations=annotations,
                metadata=data
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse custom document {doc_id}: {e}")
            return None
    
    def _parse_generic_dataset(self, metadata: DatasetMetadata, dataset_path: Path, 
                             split: Optional[str] = None) -> LoadedDataset:
        """Parse generic dataset directory"""
        documents = []
        
        # Look for common patterns
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        doc_extensions = {'.pdf', '.docx', '.doc'}
        
        # Process all files
        for file_path in dataset_path.rglob("*"):
            if file_path.is_file():
                try:
                    suffix = file_path.suffix.lower()
                    
                    if suffix in image_extensions or suffix in doc_extensions:
                        # Look for corresponding annotation file
                        ann_candidates = [
                            file_path.with_suffix('.json'),
                            file_path.with_suffix('.xml'),
                            file_path.with_suffix('.txt'),
                            file_path.parent / 'annotations' / f"{file_path.stem}.json"
                        ]
                        
                        annotations = []
                        for ann_file in ann_candidates:
                            if ann_file.exists():
                                if ann_file.suffix == '.json':
                                    with open(ann_file, 'r', encoding='utf-8') as f:
                                        ann_data = json.load(f)
                                        if isinstance(ann_data, list):
                                            for idx, ann in enumerate(ann_data):
                                                annotations.append(Annotation(
                                                    id=f"{file_path.stem}_ann_{idx}",
                                                    type=ann.get('type', 'unknown'),
                                                    text=ann.get('text', ''),
                                                    bbox=ann.get('bbox', [0, 0, 0, 0]),
                                                    metadata=ann
                                                ))
                                break
                        
                        doc = DocumentRecord(
                            id=file_path.stem,
                            name=file_path.name,
                            path=str(file_path),
                            annotations=annotations,
                            metadata={
                                'file_type': suffix,
                                'file_size': file_path.stat().st_size,
                                'relative_path': str(file_path.relative_to(dataset_path))
                            }
                        )
                        
                        documents.append(doc)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process file {file_path}: {e}")
        
        dataset = LoadedDataset(metadata=metadata, documents=documents)
        dataset.statistics = self._calculate_dataset_statistics(dataset)
        
        return dataset
    
    def _apply_split(self, dataset: LoadedDataset, split: str) -> LoadedDataset:
        """Apply dataset split filter"""
        if split == 'train' and dataset.train_split:
            split_ids = set(dataset.train_split)
        elif split == 'val' and dataset.val_split:
            split_ids = set(dataset.val_split)
        elif split == 'test' and dataset.test_split:
            split_ids = set(dataset.test_split)
        else:
            # If no predefined splits, create them
            doc_ids = [doc.id for doc in dataset.documents]
            random.shuffle(doc_ids)
            
            n_docs = len(doc_ids)
            train_end = int(n_docs * 0.8)
            val_end = int(n_docs * 0.9)
            
            if split == 'train':
                split_ids = set(doc_ids[:train_end])
            elif split == 'val':
                split_ids = set(doc_ids[train_end:val_end])
            elif split == 'test':
                split_ids = set(doc_ids[val_end:])
            else:
                return dataset
        
        # Filter documents
        dataset.documents = [doc for doc in dataset.documents if doc.id in split_ids]
        
        return dataset
    
    def _calculate_dataset_statistics(self, dataset: LoadedDataset) -> Dict[str, Any]:
        """Calculate dataset statistics"""
        stats = {
            "total_documents": len(dataset.documents),
            "total_annotations": sum(len(doc.annotations) for doc in dataset.documents),
            "average_annotations_per_doc": 0.0,
            "splits": {
                "train": len(dataset.train_split),
                "val": len(dataset.val_split),
                "test": len(dataset.test_split)
            }
        }
        
        if stats["total_documents"] > 0:
            stats["average_annotations_per_doc"] = stats["total_annotations"] / stats["total_documents"]
        
        return stats
    
    def export_dataset_info(self, dataset: LoadedDataset) -> Dict[str, Any]:
        """Export dataset information for inspection"""
        return {
            "metadata": {
                "name": dataset.metadata.name,
                "format": dataset.metadata.format.value,
                "description": dataset.metadata.description,
                "size_mb": dataset.metadata.size_mb
            },
            "statistics": dataset.statistics,
            "sample_documents": [
                {
                    "id": doc.id,
                    "num_annotations": len(doc.annotations),
                    "text_preview": doc.text_content[:100] + "..." if len(doc.text_content) > 100 else doc.text_content
                }
                for doc in dataset.documents[:5]  # Show first 5 documents
            ]
        }


# Factory function
def create_dataset_loader(**config_kwargs) -> DatasetLoader:
    """Factory function to create dataset loader"""
    config = DatasetLoaderConfig(**config_kwargs)
    return DatasetLoader(config)