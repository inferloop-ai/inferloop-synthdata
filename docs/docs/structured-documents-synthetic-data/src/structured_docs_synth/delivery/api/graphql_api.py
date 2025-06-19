#!/usr/bin/env python3
"""
GraphQL API for structured documents synthetic data system.

Provides GraphQL endpoints for document generation, validation, and management
with real-time subscriptions and efficient data fetching.

Features:
- Document generation and validation
- Real-time progress tracking via subscriptions
- Batch operations support
- Quality metrics and analytics
- Privacy-preserving data access
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

import strawberry
from strawberry.types import Info
from strawberry.asgi import GraphQL
from strawberry.subscriptions import GRAPHQL_WS_PROTOCOL
from strawberry.permission import BasePermission

from ...core import get_logger, get_config
from ...core.exceptions import ValidationError, ProcessingError
from ..storage import DatabaseStorage, CacheManager
from ...privacy import create_privacy_engine
from ...quality import create_quality_engine
from ...generation import create_layout_engine


logger = get_logger(__name__)
config = get_config()


# Permission classes
class IsAuthenticated(BasePermission):
    message = "User must be authenticated"
    
    def has_permission(self, source: Any, info: Info, **kwargs) -> bool:
        return info.context.get("user") is not None


class IsAdmin(BasePermission):
    message = "Admin access required"
    
    def has_permission(self, source: Any, info: Info, **kwargs) -> bool:
        user = info.context.get("user")
        return user and user.get("role") == "admin"


# GraphQL Types
@strawberry.type
class User:
    id: str
    username: str
    email: str
    role: str
    created_at: datetime


@strawberry.type
class DocumentInfo:
    id: str
    name: str
    format: str
    size: int
    created_at: datetime
    updated_at: datetime
    status: str
    metadata: str  # JSON string


@strawberry.type
class GenerationRequest:
    id: str
    user_id: str
    config: str  # JSON string
    status: str
    progress: float
    created_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]


@strawberry.type
class ValidationResult:
    document_id: str
    is_valid: bool
    score: float
    issues: List[str]
    recommendations: List[str]
    timestamp: datetime


@strawberry.type
class QualityMetrics:
    document_id: str
    ocr_accuracy: float
    layout_quality: float
    content_diversity: float
    privacy_score: float
    overall_score: float
    timestamp: datetime


@strawberry.type
class GenerationProgress:
    request_id: str
    stage: str
    progress: float
    message: str
    timestamp: datetime


# Input types
@strawberry.input
class DocumentGenerationInput:
    count: int = 10
    domain: str = "general"
    format: str = "docx"
    privacy_level: str = "medium"
    quality_threshold: float = 0.8
    custom_config: Optional[str] = None


@strawberry.input
class ValidationInput:
    document_id: str
    validation_rules: List[str]
    privacy_check: bool = True


@strawberry.input
class ExportInput:
    document_ids: List[str]
    format: str
    destination: str
    options: Optional[str] = None  # JSON string


# Main GraphQL API class
class GraphQLAPI:
    def __init__(self):
        self.storage = DatabaseStorage()
        self.cache = CacheManager()
        self.privacy_engine = create_privacy_engine()
        self.quality_engine = create_quality_engine()
        self.layout_engine = create_layout_engine()
        self.active_subscriptions = {}
        
    async def get_user_from_context(self, info: Info) -> Optional[Dict[str, Any]]:
        """Extract user from GraphQL context"""
        return info.context.get("user")
    
    async def authenticate_request(self, info: Info) -> bool:
        """Authenticate GraphQL request"""
        user = await self.get_user_from_context(info)
        return user is not None
    
    async def get_document_info(self, document_id: str) -> Optional[DocumentInfo]:
        """Get document information from storage"""
        try:
            doc = await self.storage.get_document(document_id)
            if not doc:
                return None
                
            return DocumentInfo(
                id=doc['id'],
                name=doc['name'],
                format=doc['format'],
                size=doc['size'],
                created_at=doc['created_at'],
                updated_at=doc['updated_at'],
                status=doc['status'],
                metadata=json.dumps(doc.get('metadata', {}))
            )
        except Exception as e:
            logger.error(f"Error getting document info: {e}")
            return None
    
    async def generate_documents_async(self, config: DocumentGenerationInput, user_id: str) -> str:
        """Start asynchronous document generation"""
        request_id = str(uuid4())
        
        try:
            # Store generation request
            await self.storage.store_generation_request({
                'id': request_id,
                'user_id': user_id,
                'config': config.__dict__,
                'status': 'pending',
                'progress': 0.0,
                'created_at': datetime.utcnow()
            })
            
            # Start background generation task
            asyncio.create_task(self._generate_documents_background(request_id, config))
            
            return request_id
            
        except Exception as e:
            logger.error(f"Error starting document generation: {e}")
            raise ProcessingError(f"Failed to start generation: {str(e)}")
    
    async def _generate_documents_background(self, request_id: str, config: DocumentGenerationInput):
        """Background task for document generation"""
        try:
            # Update progress
            await self._update_generation_progress(request_id, "initializing", 0.1, "Initializing generation...")
            
            # Generate documents
            documents = []
            for i in range(config.count):
                # Generate single document
                doc = await self.layout_engine.generate_document(
                    domain=config.domain,
                    format=config.format,
                    privacy_level=config.privacy_level
                )
                
                # Apply privacy protection
                if config.privacy_level != "none":
                    doc = await self.privacy_engine.apply_protection(doc, config.privacy_level)
                
                # Store document
                doc_id = await self.storage.store_document(doc)
                documents.append(doc_id)
                
                # Update progress
                progress = 0.1 + (0.8 * (i + 1) / config.count)
                await self._update_generation_progress(
                    request_id, "generating", progress, f"Generated {i+1}/{config.count} documents"
                )
            
            # Final validation
            await self._update_generation_progress(request_id, "validating", 0.95, "Validating generated documents...")
            
            # Complete generation
            await self.storage.update_generation_request(request_id, {
                'status': 'completed',
                'progress': 1.0,
                'completed_at': datetime.utcnow(),
                'result': {'document_ids': documents}
            })
            
            await self._update_generation_progress(request_id, "completed", 1.0, "Generation completed successfully")
            
        except Exception as e:
            logger.error(f"Error in background generation: {e}")
            await self.storage.update_generation_request(request_id, {
                'status': 'failed',
                'error_message': str(e),
                'completed_at': datetime.utcnow()
            })
            await self._update_generation_progress(request_id, "failed", 0.0, f"Generation failed: {str(e)}")
    
    async def _update_generation_progress(self, request_id: str, stage: str, progress: float, message: str):
        """Update generation progress and notify subscribers"""
        progress_update = GenerationProgress(
            request_id=request_id,
            stage=stage,
            progress=progress,
            message=message,
            timestamp=datetime.utcnow()
        )
        
        # Update in storage
        await self.storage.update_generation_request(request_id, {
            'progress': progress,
            'status': stage
        })
        
        # Notify subscribers
        if request_id in self.active_subscriptions:
            for subscriber in self.active_subscriptions[request_id]:
                await subscriber.send(progress_update)


# Initialize API instance
api_instance = GraphQLAPI()


# Query resolvers
@strawberry.type
class Query:
    @strawberry.field(permission_classes=[IsAuthenticated])
    async def me(self, info: Info) -> Optional[User]:
        """Get current user information"""
        user_data = await api_instance.get_user_from_context(info)
        if not user_data:
            return None
            
        return User(
            id=user_data['id'],
            username=user_data['username'],
            email=user_data['email'],
            role=user_data['role'],
            created_at=user_data['created_at']
        )
    
    @strawberry.field(permission_classes=[IsAuthenticated])
    async def document(self, info: Info, id: str) -> Optional[DocumentInfo]:
        """Get document by ID"""
        return await api_instance.get_document_info(id)
    
    @strawberry.field(permission_classes=[IsAuthenticated])
    async def documents(self, info: Info, limit: int = 10, offset: int = 0) -> List[DocumentInfo]:
        """List documents for current user"""
        user = await api_instance.get_user_from_context(info)
        if not user:
            return []
            
        documents = await api_instance.storage.list_user_documents(
            user['id'], limit=limit, offset=offset
        )
        
        return [
            DocumentInfo(
                id=doc['id'],
                name=doc['name'],
                format=doc['format'],
                size=doc['size'],
                created_at=doc['created_at'],
                updated_at=doc['updated_at'],
                status=doc['status'],
                metadata=json.dumps(doc.get('metadata', {}))
            )
            for doc in documents
        ]
    
    @strawberry.field(permission_classes=[IsAuthenticated])
    async def generation_request(self, info: Info, id: str) -> Optional[GenerationRequest]:
        """Get generation request status"""
        request = await api_instance.storage.get_generation_request(id)
        if not request:
            return None
            
        return GenerationRequest(
            id=request['id'],
            user_id=request['user_id'],
            config=json.dumps(request['config']),
            status=request['status'],
            progress=request['progress'],
            created_at=request['created_at'],
            completed_at=request.get('completed_at'),
            error_message=request.get('error_message')
        )
    
    @strawberry.field(permission_classes=[IsAuthenticated])
    async def quality_metrics(self, info: Info, document_id: str) -> Optional[QualityMetrics]:
        """Get quality metrics for a document"""
        metrics = await api_instance.quality_engine.get_metrics(document_id)
        if not metrics:
            return None
            
        return QualityMetrics(
            document_id=document_id,
            ocr_accuracy=metrics['ocr_accuracy'],
            layout_quality=metrics['layout_quality'],
            content_diversity=metrics['content_diversity'],
            privacy_score=metrics['privacy_score'],
            overall_score=metrics['overall_score'],
            timestamp=metrics['timestamp']
        )


# Mutation resolvers
@strawberry.type
class Mutation:
    @strawberry.field(permission_classes=[IsAuthenticated])
    async def generate_documents(self, info: Info, input: DocumentGenerationInput) -> str:
        """Start document generation"""
        user = await api_instance.get_user_from_context(info)
        if not user:
            raise ValidationError("User not authenticated")
            
        return await api_instance.generate_documents_async(input, user['id'])
    
    @strawberry.field(permission_classes=[IsAuthenticated])
    async def validate_document(self, info: Info, input: ValidationInput) -> ValidationResult:
        """Validate a document"""
        try:
            result = await api_instance.quality_engine.validate_document(
                input.document_id,
                rules=input.validation_rules,
                privacy_check=input.privacy_check
            )
            
            return ValidationResult(
                document_id=input.document_id,
                is_valid=result['is_valid'],
                score=result['score'],
                issues=result['issues'],
                recommendations=result['recommendations'],
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error validating document: {e}")
            raise ValidationError(f"Validation failed: {str(e)}")
    
    @strawberry.field(permission_classes=[IsAuthenticated])
    async def delete_document(self, info: Info, id: str) -> bool:
        """Delete a document"""
        try:
            user = await api_instance.get_user_from_context(info)
            if not user:
                return False
                
            return await api_instance.storage.delete_document(id, user['id'])
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    @strawberry.field(permission_classes=[IsAuthenticated])
    async def export_documents(self, info: Info, input: ExportInput) -> str:
        """Export documents"""
        try:
            user = await api_instance.get_user_from_context(info)
            if not user:
                raise ValidationError("User not authenticated")
                
            export_id = await api_instance.storage.create_export_job({
                'user_id': user['id'],
                'document_ids': input.document_ids,
                'format': input.format,
                'destination': input.destination,
                'options': json.loads(input.options) if input.options else {},
                'status': 'pending',
                'created_at': datetime.utcnow()
            })
            
            return export_id
            
        except Exception as e:
            logger.error(f"Error starting export: {e}")
            raise ProcessingError(f"Export failed: {str(e)}")


# Subscription resolvers
@strawberry.type
class Subscription:
    @strawberry.subscription(permission_classes=[IsAuthenticated])
    async def generation_progress(self, info: Info, request_id: str) -> GenerationProgress:
        """Subscribe to generation progress updates"""
        # Add subscriber to active subscriptions
        if request_id not in api_instance.active_subscriptions:
            api_instance.active_subscriptions[request_id] = []
        
        # Create async generator for progress updates
        async def progress_generator():
            while True:
                # Check if generation is still active
                request = await api_instance.storage.get_generation_request(request_id)
                if not request or request['status'] in ['completed', 'failed']:
                    break
                    
                # Wait for updates (this would be replaced with proper event handling)
                await asyncio.sleep(1)
                
                # Yield current progress
                yield GenerationProgress(
                    request_id=request_id,
                    stage=request['status'],
                    progress=request['progress'],
                    message=f"Status: {request['status']}",
                    timestamp=datetime.utcnow()
                )
        
        async for progress in progress_generator():
            yield progress


# Create GraphQL schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)


# Create GraphQL app
def create_graphql_app() -> GraphQL:
    """Create GraphQL ASGI application"""
    return GraphQL(
        schema,
        subscription_protocols=[GRAPHQL_WS_PROTOCOL],
        context_getter=lambda request: {
            "request": request,
            "user": getattr(request, "user", None)
        }
    )


# Factory function
def create_graphql_api() -> GraphQLAPI:
    """Create GraphQL API instance"""
    return GraphQLAPI()


__all__ = [
    'GraphQLAPI',
    'create_graphql_app',
    'create_graphql_api',
    'schema'
]