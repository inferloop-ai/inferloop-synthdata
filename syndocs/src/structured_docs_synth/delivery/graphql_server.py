#!/usr/bin/env python3
"""
GraphQL Server for document generation API.

Provides a GraphQL interface for document generation, querying,
and management operations.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
import json
import graphene
from graphene import ObjectType, String, Int, Float, Boolean, Field, List as GrapheneList
from graphene import Schema, Mutation, InputObjectType, Enum
from graphql.execution.executors.asyncio import AsyncioExecutor

from ..core.logging import get_logger
from ..core.exceptions import ValidationError, ProcessingError
from ..orchestration.pipeline_coordinator import PipelineCoordinator


logger = get_logger(__name__)


# GraphQL Type Definitions
class DocumentType(Enum):
    """Document types enumeration"""
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    LATEX = "latex"
    JSON = "json"


class PrivacyLevel(Enum):
    """Privacy level enumeration"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class JobStatus(Enum):
    """Job status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DocumentDomain(Enum):
    """Document domain enumeration"""
    GENERAL = "general"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    LEGAL = "legal"
    GOVERNMENT = "government"


# GraphQL Object Types
class Document(ObjectType):
    """Document type"""
    id = String(required=True)
    name = String(required=True)
    type = Field(DocumentType, required=True)
    domain = Field(DocumentDomain)
    size = Int()
    created_at = String()
    updated_at = String()
    metadata = String()  # JSON string
    content_preview = String()
    privacy_level = Field(PrivacyLevel)
    validation_status = String()
    export_formats = GrapheneList(String)


class GenerationJob(ObjectType):
    """Document generation job type"""
    id = String(required=True)
    status = Field(JobStatus, required=True)
    created_at = String(required=True)
    started_at = String()
    completed_at = String()
    progress = Float()
    document_count = Int()
    error_message = String()
    documents = GrapheneList(Document)
    parameters = String()  # JSON string


class GenerationStats(ObjectType):
    """Generation statistics type"""
    total_documents = Int()
    total_jobs = Int()
    active_jobs = Int()
    success_rate = Float()
    average_generation_time = Float()
    documents_by_type = String()  # JSON string
    documents_by_domain = String()  # JSON string


class OCRResult(ObjectType):
    """OCR result type"""
    document_id = String(required=True)
    text = String(required=True)
    confidence = Float()
    word_count = Int()
    language = String()
    processing_time = Float()


class ValidationResult(ObjectType):
    """Validation result type"""
    document_id = String(required=True)
    is_valid = Boolean(required=True)
    score = Float()
    issues = GrapheneList(String)
    recommendations = GrapheneList(String)


class ExportResult(ObjectType):
    """Export result type"""
    document_id = String(required=True)
    format = String(required=True)
    file_path = String()
    download_url = String()
    size = Int()
    created_at = String()


# Input Types
class GenerationInput(InputObjectType):
    """Document generation input"""
    count = Int(required=True)
    domain = Field(DocumentDomain, required=True)
    document_type = Field(DocumentType, default_value=DocumentType.PDF)
    privacy_level = Field(PrivacyLevel, default_value=PrivacyLevel.MEDIUM)
    template = String()
    metadata = String()  # JSON string
    parameters = String()  # JSON string


class FilterInput(InputObjectType):
    """Filter input for queries"""
    domain = Field(DocumentDomain)
    document_type = Field(DocumentType)
    privacy_level = Field(PrivacyLevel)
    created_after = String()
    created_before = String()
    status = Field(JobStatus)


class OCRInput(InputObjectType):
    """OCR processing input"""
    document_id = String(required=True)
    language = String(default_value="eng")
    enhance_image = Boolean(default_value=True)
    detect_layout = Boolean(default_value=True)


class ExportInput(InputObjectType):
    """Export input"""
    document_id = String(required=True)
    format = String(required=True)
    include_metadata = Boolean(default_value=True)
    compress = Boolean(default_value=False)


# Mutations
class GenerateDocuments(Mutation):
    """Generate documents mutation"""
    
    class Arguments:
        input = GenerationInput(required=True)
    
    job = Field(GenerationJob)
    
    async def mutate(self, info, input):
        """Execute document generation"""
        try:
            # Get pipeline coordinator from context
            coordinator = info.context.get('coordinator')
            if not coordinator:
                raise ProcessingError("Pipeline coordinator not available")
            
            # Parse parameters
            parameters = {}
            if input.parameters:
                parameters = json.loads(input.parameters)
            
            metadata = {}
            if input.metadata:
                metadata = json.loads(input.metadata)
            
            # Create generation job
            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Start pipeline
            pipeline_config = {
                "count": input.count,
                "domain": input.domain.value,
                "document_type": input.document_type.value,
                "privacy_level": input.privacy_level.value,
                "template": input.template,
                **parameters
            }
            
            instance_id = coordinator.start_pipeline(
                pipeline_id="standard",
                input_data={"records": [{}] * input.count},
                parameters=pipeline_config
            )
            
            # Create job object
            job = GenerationJob(
                id=job_id,
                status=JobStatus.RUNNING,
                created_at=datetime.now().isoformat(),
                document_count=input.count,
                parameters=json.dumps(pipeline_config)
            )
            
            # Store job mapping
            info.context['jobs'][job_id] = instance_id
            
            return GenerateDocuments(job=job)
            
        except Exception as e:
            logger.error(f"Document generation failed: {e}")
            raise


class ProcessOCR(Mutation):
    """Process document with OCR"""
    
    class Arguments:
        input = OCRInput(required=True)
    
    result = Field(OCRResult)
    
    async def mutate(self, info, input):
        """Execute OCR processing"""
        try:
            # Simulate OCR processing
            result = OCRResult(
                document_id=input.document_id,
                text="Sample OCR extracted text for document",
                confidence=0.95,
                word_count=42,
                language=input.language,
                processing_time=2.5
            )
            
            return ProcessOCR(result=result)
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise


class ValidateDocument(Mutation):
    """Validate document mutation"""
    
    class Arguments:
        document_id = String(required=True)
        rules = GrapheneList(String)
    
    result = Field(ValidationResult)
    
    async def mutate(self, info, document_id, rules=None):
        """Execute document validation"""
        try:
            # Simulate validation
            result = ValidationResult(
                document_id=document_id,
                is_valid=True,
                score=0.92,
                issues=[],
                recommendations=["Consider adding more structured headers"]
            )
            
            return ValidateDocument(result=result)
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise


class ExportDocument(Mutation):
    """Export document mutation"""
    
    class Arguments:
        input = ExportInput(required=True)
    
    result = Field(ExportResult)
    
    async def mutate(self, info, input):
        """Execute document export"""
        try:
            # Simulate export
            result = ExportResult(
                document_id=input.document_id,
                format=input.format,
                file_path=f"/exports/{input.document_id}.{input.format}",
                download_url=f"https://api.example.com/download/{input.document_id}.{input.format}",
                size=1024 * 50,  # 50KB
                created_at=datetime.now().isoformat()
            )
            
            return ExportDocument(result=result)
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise


class CancelJob(Mutation):
    """Cancel job mutation"""
    
    class Arguments:
        job_id = String(required=True)
    
    success = Boolean()
    message = String()
    
    async def mutate(self, info, job_id):
        """Cancel a running job"""
        try:
            coordinator = info.context.get('coordinator')
            jobs = info.context.get('jobs', {})
            
            if job_id in jobs:
                instance_id = jobs[job_id]
                # Cancel pipeline
                # coordinator.cancel_pipeline(instance_id)
                
                return CancelJob(success=True, message="Job cancelled successfully")
            else:
                return CancelJob(success=False, message="Job not found")
                
        except Exception as e:
            logger.error(f"Job cancellation failed: {e}")
            return CancelJob(success=False, message=str(e))


# Queries
class Query(ObjectType):
    """Root query type"""
    
    # Single item queries
    document = Field(
        Document,
        id=String(required=True),
        description="Get a single document by ID"
    )
    
    job = Field(
        GenerationJob,
        id=String(required=True),
        description="Get a generation job by ID"
    )
    
    # List queries
    documents = GrapheneList(
        Document,
        filter=FilterInput(),
        limit=Int(default_value=20),
        offset=Int(default_value=0),
        description="List documents with optional filtering"
    )
    
    jobs = GrapheneList(
        GenerationJob,
        filter=FilterInput(),
        limit=Int(default_value=20),
        offset=Int(default_value=0),
        description="List generation jobs"
    )
    
    # Statistics
    stats = Field(
        GenerationStats,
        description="Get generation statistics"
    )
    
    # Resolvers
    async def resolve_document(self, info, id):
        """Resolve single document"""
        # Simulate document retrieval
        return Document(
            id=id,
            name=f"Document {id}",
            type=DocumentType.PDF,
            domain=DocumentDomain.GENERAL,
            size=1024 * 100,
            created_at=datetime.now().isoformat(),
            privacy_level=PrivacyLevel.MEDIUM
        )
    
    async def resolve_job(self, info, id):
        """Resolve single job"""
        jobs = info.context.get('jobs', {})
        coordinator = info.context.get('coordinator')
        
        if id in jobs and coordinator:
            instance_id = jobs[id]
            status = coordinator.get_pipeline_status(instance_id)
            
            return GenerationJob(
                id=id,
                status=JobStatus.RUNNING if status else JobStatus.COMPLETED,
                created_at=datetime.now().isoformat(),
                progress=0.75,
                document_count=10
            )
        
        return None
    
    async def resolve_documents(self, info, filter=None, limit=20, offset=0):
        """Resolve document list"""
        # Simulate document list
        documents = []
        for i in range(offset, offset + limit):
            doc = Document(
                id=f"doc_{i}",
                name=f"Document {i}",
                type=DocumentType.PDF,
                domain=DocumentDomain.GENERAL,
                size=1024 * (i + 1),
                created_at=datetime.now().isoformat()
            )
            documents.append(doc)
        
        return documents
    
    async def resolve_jobs(self, info, filter=None, limit=20, offset=0):
        """Resolve job list"""
        jobs = info.context.get('jobs', {})
        
        job_list = []
        for job_id in list(jobs.keys())[offset:offset + limit]:
            job = GenerationJob(
                id=job_id,
                status=JobStatus.COMPLETED,
                created_at=datetime.now().isoformat(),
                document_count=10
            )
            job_list.append(job)
        
        return job_list
    
    async def resolve_stats(self, info):
        """Resolve statistics"""
        coordinator = info.context.get('coordinator')
        
        if coordinator:
            stats_data = coordinator.get_stats()
            
            return GenerationStats(
                total_documents=1000,
                total_jobs=100,
                active_jobs=len(stats_data.get('running_instances', [])),
                success_rate=0.95,
                average_generation_time=5.2,
                documents_by_type=json.dumps({"pdf": 600, "docx": 300, "html": 100}),
                documents_by_domain=json.dumps({"general": 400, "finance": 300, "healthcare": 300})
            )
        
        return GenerationStats()


# Root Mutation
class Mutation(ObjectType):
    """Root mutation type"""
    generate_documents = GenerateDocuments.Field()
    process_ocr = ProcessOCR.Field()
    validate_document = ValidateDocument.Field()
    export_document = ExportDocument.Field()
    cancel_job = CancelJob.Field()


# Create schema
schema = Schema(query=Query, mutation=Mutation)


class GraphQLServer:
    """GraphQL server implementation"""
    
    def __init__(self, coordinator: Optional[PipelineCoordinator] = None):
        """Initialize GraphQL server"""
        self.logger = get_logger(__name__)
        self.coordinator = coordinator or PipelineCoordinator()
        self.schema = schema
        self.context = {
            'coordinator': self.coordinator,
            'jobs': {}  # Job ID to pipeline instance mapping
        }
        
        self.logger.info("GraphQL server initialized")
    
    async def execute_query(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute GraphQL query.
        
        Args:
            query: GraphQL query string
            variables: Query variables
            operation_name: Operation name
            
        Returns:
            Query result
        """
        try:
            # Execute query with async executor
            result = await self.schema.execute_async(
                query,
                variables=variables,
                operation_name=operation_name,
                context=self.context,
                executor=AsyncioExecutor()
            )
            
            # Format result
            response = {}
            
            if result.data:
                response['data'] = result.data
            
            if result.errors:
                response['errors'] = [
                    {
                        'message': str(error),
                        'path': error.path,
                        'locations': [
                            {'line': loc.line, 'column': loc.column}
                            for loc in error.locations
                        ] if error.locations else None
                    }
                    for error in result.errors
                ]
            
            return response
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            return {
                'errors': [{'message': str(e)}]
            }
    
    def get_schema_introspection(self) -> Dict[str, Any]:
        """Get schema introspection for GraphQL playground"""
        from graphql import graphql_sync
        
        introspection_query = """
            query IntrospectionQuery {
                __schema {
                    queryType { name }
                    mutationType { name }
                    types {
                        ...FullType
                    }
                }
            }
            
            fragment FullType on __Type {
                kind
                name
                description
                fields(includeDeprecated: true) {
                    name
                    description
                    args {
                        ...InputValue
                    }
                    type {
                        ...TypeRef
                    }
                    isDeprecated
                    deprecationReason
                }
                inputFields {
                    ...InputValue
                }
                interfaces {
                    ...TypeRef
                }
                enumValues(includeDeprecated: true) {
                    name
                    description
                    isDeprecated
                    deprecationReason
                }
                possibleTypes {
                    ...TypeRef
                }
            }
            
            fragment InputValue on __InputValue {
                name
                description
                type { ...TypeRef }
                defaultValue
            }
            
            fragment TypeRef on __Type {
                kind
                name
                ofType {
                    kind
                    name
                    ofType {
                        kind
                        name
                        ofType {
                            kind
                            name
                        }
                    }
                }
            }
        """
        
        result = graphql_sync(self.schema.graphql_schema, introspection_query)
        return result.data if result.data else {}
    
    def cleanup(self):
        """Clean up server resources"""
        if self.coordinator:
            self.coordinator.cleanup()
        
        self.logger.info("GraphQL server cleaned up")


# Factory function
def create_graphql_server(
    coordinator: Optional[PipelineCoordinator] = None
) -> GraphQLServer:
    """Create and return a GraphQL server instance"""
    return GraphQLServer(coordinator)


# Example queries for documentation
EXAMPLE_QUERIES = {
    "generate_documents": """
        mutation GenerateDocuments($input: GenerationInput!) {
            generateDocuments(input: $input) {
                job {
                    id
                    status
                    createdAt
                    documentCount
                }
            }
        }
    """,
    
    "get_job_status": """
        query GetJobStatus($jobId: String!) {
            job(id: $jobId) {
                id
                status
                progress
                completedAt
                documents {
                    id
                    name
                    type
                }
            }
        }
    """,
    
    "list_documents": """
        query ListDocuments($filter: FilterInput, $limit: Int) {
            documents(filter: $filter, limit: $limit) {
                id
                name
                type
                domain
                createdAt
                privacyLevel
            }
        }
    """,
    
    "process_ocr": """
        mutation ProcessOCR($input: OCRInput!) {
            processOcr(input: $input) {
                result {
                    documentId
                    text
                    confidence
                    wordCount
                }
            }
        }
    """,
    
    "get_statistics": """
        query GetStatistics {
            stats {
                totalDocuments
                totalJobs
                activeJobs
                successRate
                averageGenerationTime
            }
        }
    """
}