"""
Database storage implementation for structured document data.
Provides database operations for document metadata, content, and relationships.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime, Text, Boolean,
    LargeBinary, JSON, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import SQLAlchemyError

from ...core.config import BaseConfig
from ...core.exceptions import ValidationError
from ...core.logging import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class Document(Base):
    """Document metadata table"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_type = Column(String(50), nullable=False)
    document_format = Column(String(20), nullable=False)
    title = Column(String(500))
    author = Column(String(200))
    subject = Column(String(500))
    language = Column(String(10), default="en")
    
    # File information
    file_path = Column(String(1000))
    file_size = Column(Integer)
    file_hash = Column(String(64))  # SHA-256 hash
    
    # Generation metadata
    generation_config = Column(JSON)
    generation_time = Column(Float)
    quality_score = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    content = relationship("DocumentContent", back_populates="document", uselist=False)
    elements = relationship("DocumentElement", back_populates="document")
    annotations = relationship("DocumentAnnotation", back_populates="document")
    validations = relationship("ValidationResult", back_populates="document")
    
    # Indexes
    __table_args__ = (
        Index('ix_documents_type_created', 'document_type', 'created_at'),
        Index('ix_documents_author_created', 'author', 'created_at'),
        Index('ix_documents_quality', 'quality_score'),
    )


class DocumentContent(Base):
    """Document content table"""
    __tablename__ = "document_content"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
    # Text content
    full_text = Column(Text)
    word_count = Column(Integer)
    character_count = Column(Integer)
    paragraph_count = Column(Integer)
    
    # Structured content
    sections = Column(JSON)  # List of sections with hierarchy
    metadata_fields = Column(JSON)  # Additional metadata
    
    # Content analysis
    language_detected = Column(String(10))
    readability_score = Column(Float)
    entity_count = Column(Integer)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="content")


class DocumentElement(Base):
    """Document layout elements table"""
    __tablename__ = "document_elements"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
    # Element properties
    element_type = Column(String(50), nullable=False)
    element_id = Column(String(100))  # External element ID
    page_number = Column(Integer, default=1)
    
    # Bounding box
    bbox_x = Column(Float, nullable=False)
    bbox_y = Column(Float, nullable=False)
    bbox_width = Column(Float, nullable=False)
    bbox_height = Column(Float, nullable=False)
    
    # Content
    text_content = Column(Text)
    confidence = Column(Float, default=1.0)
    
    # Hierarchy
    parent_id = Column(UUID(as_uuid=True), ForeignKey("document_elements.id"))
    level = Column(Integer, default=0)
    
    # Additional attributes
    style_attributes = Column(JSON)
    metadata = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="elements")
    parent = relationship("DocumentElement", remote_side=[id])
    children = relationship("DocumentElement")
    
    # Indexes
    __table_args__ = (
        Index('ix_elements_document_type', 'document_id', 'element_type'),
        Index('ix_elements_page', 'document_id', 'page_number'),
        Index('ix_elements_bbox', 'bbox_x', 'bbox_y'),
    )


class DocumentAnnotation(Base):
    """Document annotations table"""
    __tablename__ = "document_annotations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
    # Annotation properties
    annotation_type = Column(String(50), nullable=False)  # entity, bbox, relationship, etc.
    annotation_id = Column(String(100))
    
    # Position information
    start_pos = Column(Integer)  # For text annotations
    end_pos = Column(Integer)
    page_number = Column(Integer)
    
    # Bounding box (for visual annotations)
    bbox_x = Column(Float)
    bbox_y = Column(Float)
    bbox_width = Column(Float)
    bbox_height = Column(Float)
    
    # Annotation data
    label = Column(String(200))
    value = Column(Text)
    confidence = Column(Float, default=1.0)
    attributes = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="annotations")
    
    # Indexes
    __table_args__ = (
        Index('ix_annotations_document_type', 'document_id', 'annotation_type'),
        Index('ix_annotations_label', 'label'),
        Index('ix_annotations_position', 'start_pos', 'end_pos'),
    )


class ValidationResult(Base):
    """Document validation results table"""
    __tablename__ = "validation_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
    # Validation metadata
    validation_type = Column(String(50), nullable=False)
    validator_version = Column(String(20))
    validation_config = Column(JSON)
    
    # Results
    is_valid = Column(Boolean, nullable=False)
    validation_score = Column(Float)
    issues_count = Column(Integer, default=0)
    
    # Detailed results
    issues = Column(JSON)  # List of validation issues
    recommendations = Column(JSON)  # List of recommendations
    metrics = Column(JSON)  # Detailed metrics
    
    # Performance
    validation_time = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="validations")
    
    # Indexes
    __table_args__ = (
        Index('ix_validations_document_type', 'document_id', 'validation_type'),
        Index('ix_validations_score', 'validation_score'),
        Index('ix_validations_valid', 'is_valid'),
    )


class GenerationJob(Base):
    """Generation job tracking table"""
    __tablename__ = "generation_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_type = Column(String(50), nullable=False)  # single, batch
    status = Column(String(20), nullable=False, default="pending")
    
    # Job configuration
    generation_config = Column(JSON)
    document_count = Column(Integer, default=1)
    
    # Progress tracking
    completed_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    progress_percentage = Column(Float, default=0.0)
    
    # Results
    result_document_ids = Column(JSON)  # List of generated document IDs
    error_messages = Column(JSON)  # List of error messages
    
    # Performance metrics
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    total_time = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('ix_jobs_status_created', 'status', 'created_at'),
        Index('ix_jobs_type_status', 'job_type', 'status'),
    )


class DatabaseStorageConfig(BaseConfig):
    """Database storage configuration"""
    connection_url: str = "sqlite:///structured_docs.db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo_sql: bool = False
    
    # Performance settings
    batch_size: int = 1000
    connection_retries: int = 3
    query_timeout: int = 30


class DatabaseStorage:
    """
    Database storage manager for structured document data.
    Handles document metadata, content, elements, and validation results.
    """
    
    def __init__(self, config: Optional[DatabaseStorageConfig] = None):
        """Initialize database storage"""
        self.config = config or DatabaseStorageConfig()
        
        # Create engine
        self.engine = create_engine(
            self.config.connection_url,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            echo=self.config.echo_sql
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables
        self.create_tables()
        
        logger.info("Initialized DatabaseStorage")
    
    def create_tables(self):
        """Create database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create database tables: {str(e)}")
            raise
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def store_document(
        self,
        document_data: Dict[str, Any],
        content_data: Optional[Dict[str, Any]] = None,
        elements_data: Optional[List[Dict[str, Any]]] = None,
        annotations_data: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Store document with all associated data.
        
        Args:
            document_data: Document metadata
            content_data: Document content data
            elements_data: List of document elements
            annotations_data: List of annotations
        
        Returns:
            Document ID
        """
        session = self.get_session()
        
        try:
            # Create document
            document = Document(
                document_type=document_data.get("document_type"),
                document_format=document_data.get("document_format"),
                title=document_data.get("title"),
                author=document_data.get("author"),
                subject=document_data.get("subject"),
                language=document_data.get("language", "en"),
                file_path=document_data.get("file_path"),
                file_size=document_data.get("file_size"),
                file_hash=document_data.get("file_hash"),
                generation_config=document_data.get("generation_config"),
                generation_time=document_data.get("generation_time"),
                quality_score=document_data.get("quality_score")
            )
            
            session.add(document)
            session.flush()  # Get the document ID
            
            # Store content if provided
            if content_data:
                content = DocumentContent(
                    document_id=document.id,
                    full_text=content_data.get("full_text"),
                    word_count=content_data.get("word_count"),
                    character_count=content_data.get("character_count"),
                    paragraph_count=content_data.get("paragraph_count"),
                    sections=content_data.get("sections"),
                    metadata_fields=content_data.get("metadata_fields"),
                    language_detected=content_data.get("language_detected"),
                    readability_score=content_data.get("readability_score"),
                    entity_count=content_data.get("entity_count")
                )
                session.add(content)
            
            # Store elements if provided
            if elements_data:
                for elem_data in elements_data:
                    element = DocumentElement(
                        document_id=document.id,
                        element_type=elem_data.get("element_type"),
                        element_id=elem_data.get("element_id"),
                        page_number=elem_data.get("page_number", 1),
                        bbox_x=elem_data.get("bbox_x"),
                        bbox_y=elem_data.get("bbox_y"),
                        bbox_width=elem_data.get("bbox_width"),
                        bbox_height=elem_data.get("bbox_height"),
                        text_content=elem_data.get("text_content"),
                        confidence=elem_data.get("confidence", 1.0),
                        level=elem_data.get("level", 0),
                        style_attributes=elem_data.get("style_attributes"),
                        metadata=elem_data.get("metadata")
                    )
                    session.add(element)
            
            # Store annotations if provided
            if annotations_data:
                for ann_data in annotations_data:
                    annotation = DocumentAnnotation(
                        document_id=document.id,
                        annotation_type=ann_data.get("annotation_type"),
                        annotation_id=ann_data.get("annotation_id"),
                        start_pos=ann_data.get("start_pos"),
                        end_pos=ann_data.get("end_pos"),
                        page_number=ann_data.get("page_number"),
                        bbox_x=ann_data.get("bbox_x"),
                        bbox_y=ann_data.get("bbox_y"),
                        bbox_width=ann_data.get("bbox_width"),
                        bbox_height=ann_data.get("bbox_height"),
                        label=ann_data.get("label"),
                        value=ann_data.get("value"),
                        confidence=ann_data.get("confidence", 1.0),
                        attributes=ann_data.get("attributes")
                    )
                    session.add(annotation)
            
            session.commit()
            document_id = str(document.id)
            
            logger.info(f"Stored document: {document_id}")
            return document_id
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to store document: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_document(self, document_id: str, include_content: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get document by ID.
        
        Args:
            document_id: Document ID
            include_content: Include content and elements
        
        Returns:
            Document data or None
        """
        session = self.get_session()
        
        try:
            document = session.query(Document).filter(Document.id == document_id).first()
            
            if not document:
                return None
            
            result = {
                "id": str(document.id),
                "document_type": document.document_type,
                "document_format": document.document_format,
                "title": document.title,
                "author": document.author,
                "subject": document.subject,
                "language": document.language,
                "file_path": document.file_path,
                "file_size": document.file_size,
                "file_hash": document.file_hash,
                "generation_config": document.generation_config,
                "generation_time": document.generation_time,
                "quality_score": document.quality_score,
                "created_at": document.created_at.isoformat() if document.created_at else None,
                "updated_at": document.updated_at.isoformat() if document.updated_at else None
            }
            
            if include_content:
                # Include content
                if document.content:
                    result["content"] = {
                        "full_text": document.content.full_text,
                        "word_count": document.content.word_count,
                        "character_count": document.content.character_count,
                        "paragraph_count": document.content.paragraph_count,
                        "sections": document.content.sections,
                        "metadata_fields": document.content.metadata_fields,
                        "language_detected": document.content.language_detected,
                        "readability_score": document.content.readability_score,
                        "entity_count": document.content.entity_count
                    }
                
                # Include elements
                result["elements"] = []
                for element in document.elements:
                    result["elements"].append({
                        "id": str(element.id),
                        "element_type": element.element_type,
                        "element_id": element.element_id,
                        "page_number": element.page_number,
                        "bbox": {
                            "x": element.bbox_x,
                            "y": element.bbox_y,
                            "width": element.bbox_width,
                            "height": element.bbox_height
                        },
                        "text_content": element.text_content,
                        "confidence": element.confidence,
                        "level": element.level,
                        "style_attributes": element.style_attributes,
                        "metadata": element.metadata
                    })
                
                # Include annotations
                result["annotations"] = []
                for annotation in document.annotations:
                    result["annotations"].append({
                        "id": str(annotation.id),
                        "annotation_type": annotation.annotation_type,
                        "annotation_id": annotation.annotation_id,
                        "start_pos": annotation.start_pos,
                        "end_pos": annotation.end_pos,
                        "page_number": annotation.page_number,
                        "bbox": {
                            "x": annotation.bbox_x,
                            "y": annotation.bbox_y,
                            "width": annotation.bbox_width,
                            "height": annotation.bbox_height
                        } if annotation.bbox_x is not None else None,
                        "label": annotation.label,
                        "value": annotation.value,
                        "confidence": annotation.confidence,
                        "attributes": annotation.attributes
                    })
            
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            return None
        finally:
            session.close()
    
    def list_documents(
        self,
        document_type: Optional[str] = None,
        author: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        min_quality_score: Optional[float] = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        List documents with filtering and pagination.
        
        Args:
            document_type: Filter by document type
            author: Filter by author
            created_after: Filter by creation date
            created_before: Filter by creation date
            min_quality_score: Filter by minimum quality score
            page: Page number
            page_size: Items per page
            sort_by: Sort field
            sort_order: Sort order (asc/desc)
        
        Returns:
            Tuple of (documents, total_count)
        """
        session = self.get_session()
        
        try:
            query = session.query(Document)
            
            # Apply filters
            if document_type:
                query = query.filter(Document.document_type == document_type)
            if author:
                query = query.filter(Document.author.ilike(f"%{author}%"))
            if created_after:
                query = query.filter(Document.created_at >= created_after)
            if created_before:
                query = query.filter(Document.created_at <= created_before)
            if min_quality_score is not None:
                query = query.filter(Document.quality_score >= min_quality_score)
            
            # Get total count
            total_count = query.count()
            
            # Apply sorting
            if hasattr(Document, sort_by):
                sort_column = getattr(Document, sort_by)
                if sort_order.lower() == "desc":
                    query = query.order_by(sort_column.desc())
                else:
                    query = query.order_by(sort_column.asc())
            
            # Apply pagination
            offset = (page - 1) * page_size
            documents = query.offset(offset).limit(page_size).all()
            
            # Convert to dict format
            result = []
            for doc in documents:
                result.append({
                    "id": str(doc.id),
                    "document_type": doc.document_type,
                    "document_format": doc.document_format,
                    "title": doc.title,
                    "author": doc.author,
                    "subject": doc.subject,
                    "language": doc.language,
                    "file_size": doc.file_size,
                    "quality_score": doc.quality_score,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None
                })
            
            return result, total_count
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to list documents: {str(e)}")
            return [], 0
        finally:
            session.close()
    
    def store_validation_result(
        self,
        document_id: str,
        validation_data: Dict[str, Any]
    ) -> str:
        """
        Store validation result for a document.
        
        Args:
            document_id: Document ID
            validation_data: Validation result data
        
        Returns:
            Validation result ID
        """
        session = self.get_session()
        
        try:
            validation = ValidationResult(
                document_id=document_id,
                validation_type=validation_data.get("validation_type"),
                validator_version=validation_data.get("validator_version"),
                validation_config=validation_data.get("validation_config"),
                is_valid=validation_data.get("is_valid"),
                validation_score=validation_data.get("validation_score"),
                issues_count=validation_data.get("issues_count", 0),
                issues=validation_data.get("issues"),
                recommendations=validation_data.get("recommendations"),
                metrics=validation_data.get("metrics"),
                validation_time=validation_data.get("validation_time")
            )
            
            session.add(validation)
            session.commit()
            
            validation_id = str(validation.id)
            logger.info(f"Stored validation result: {validation_id}")
            return validation_id
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to store validation result: {str(e)}")
            raise
        finally:
            session.close()
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete document and all associated data.
        
        Args:
            document_id: Document ID to delete
        
        Returns:
            True if deleted successfully
        """
        session = self.get_session()
        
        try:
            document = session.query(Document).filter(Document.id == document_id).first()
            
            if document:
                session.delete(document)
                session.commit()
                logger.info(f"Deleted document: {document_id}")
                return True
            else:
                logger.warning(f"Document not found for deletion: {document_id}")
                return False
                
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            return False
        finally:
            session.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Statistics dictionary
        """
        session = self.get_session()
        
        try:
            stats = {}
            
            # Document counts
            stats["total_documents"] = session.query(Document).count()
            stats["documents_by_type"] = {}
            
            # Count by document type
            type_counts = session.query(
                Document.document_type,
                session.query(Document).filter(Document.document_type == Document.document_type).count()
            ).distinct().all()
            
            for doc_type, count in type_counts:
                stats["documents_by_type"][doc_type] = count
            
            # Quality statistics
            quality_stats = session.query(
                session.query(Document.quality_score).filter(Document.quality_score.isnot(None))
            ).first()
            
            if quality_stats:
                scores = [doc.quality_score for doc in session.query(Document).filter(Document.quality_score.isnot(None)).all()]
                if scores:
                    stats["quality_statistics"] = {
                        "mean_score": sum(scores) / len(scores),
                        "min_score": min(scores),
                        "max_score": max(scores),
                        "documents_with_scores": len(scores)
                    }
            
            # Recent activity
            week_ago = datetime.utcnow() - timedelta(days=7)
            stats["documents_last_week"] = session.query(Document).filter(
                Document.created_at >= week_ago
            ).count()
            
            return stats
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {}
        finally:
            session.close()


def create_database_storage(
    config: Optional[Union[Dict[str, Any], DatabaseStorageConfig]] = None
) -> DatabaseStorage:
    """Factory function to create database storage"""
    if isinstance(config, dict):
        config = DatabaseStorageConfig(**config)
    return DatabaseStorage(config)