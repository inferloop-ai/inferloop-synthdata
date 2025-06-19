"""
Storage module for structured document data management.
Provides database, cache, and vector storage implementations.
"""

from typing import Dict, List, Optional, Union, Any

from .database_storage import (
    DatabaseStorage,
    DatabaseStorageConfig,
    Document,
    DocumentContent,
    DocumentElement,
    DocumentAnnotation,
    ValidationResult,
    GenerationJob,
    create_database_storage
)

from .cache_manager import (
    CacheManager,
    CacheManagerConfig,
    create_cache_manager
)

from .vector_store import (
    VectorStore,
    VectorStoreConfig,
    VectorDocument,
    create_vector_store
)

from ...core.logging import get_logger

logger = get_logger(__name__)

# Module version
__version__ = "1.0.0"

# Public API
__all__ = [
    # Database Storage
    "DatabaseStorage",
    "DatabaseStorageConfig",
    "Document",
    "DocumentContent", 
    "DocumentElement",
    "DocumentAnnotation",
    "ValidationResult",
    "GenerationJob",
    "create_database_storage",
    
    # Cache Manager
    "CacheManager",
    "CacheManagerConfig",
    "create_cache_manager",
    
    # Vector Store
    "VectorStore",
    "VectorStoreConfig", 
    "VectorDocument",
    "create_vector_store",
    
    # Factory functions
    "create_storage_suite",
    "StorageManager"
]


class StorageManager:
    """
    Unified storage manager that coordinates database, cache, and vector storage.
    """
    
    def __init__(
        self,
        database_config: Optional[Union[Dict[str, Any], DatabaseStorageConfig]] = None,
        cache_config: Optional[Union[Dict[str, Any], CacheManagerConfig]] = None,
        vector_config: Optional[Union[Dict[str, Any], VectorStoreConfig]] = None
    ):
        """Initialize storage manager"""
        self.database = create_database_storage(database_config)
        self.cache = create_cache_manager(cache_config)
        self.vector_store = create_vector_store(vector_config)
        
        logger.info("Initialized StorageManager")
    
    def store_document_complete(
        self,
        document_data: Dict[str, Any],
        content_data: Optional[Dict[str, Any]] = None,
        elements_data: Optional[List[Dict[str, Any]]] = None,
        annotations_data: Optional[List[Dict[str, Any]]] = None,
        embedding: Optional[Union[List[float], Any]] = None
    ) -> str:
        """
        Store document with all data across all storage systems.
        
        Args:
            document_data: Document metadata
            content_data: Document content
            elements_data: Document elements
            annotations_data: Document annotations
            embedding: Document embedding for vector search
        
        Returns:
            Document ID
        """
        # Store in database
        document_id = self.database.store_document(
            document_data, content_data, elements_data, annotations_data
        )
        
        # Cache document data
        full_document = self.database.get_document(document_id, include_content=True)
        if full_document:
            self.cache.cache_document(document_id, full_document)
        
        # Store in vector store if embedding provided
        if embedding:
            self.vector_store.add_document(
                document_id=document_id,
                embedding=embedding,
                metadata={
                    "document_type": document_data.get("document_type"),
                    "title": document_data.get("title"),
                    "author": document_data.get("author")
                },
                text_content=content_data.get("full_text") if content_data else None
            )
        
        return document_id
    
    def get_document_complete(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document from cache or database"""
        # Try cache first
        cached = self.cache.get_cached_document(document_id)
        if cached:
            return cached
        
        # Get from database
        document = self.database.get_document(document_id, include_content=True)
        if document:
            # Cache for future use
            self.cache.cache_document(document_id, document)
        
        return document
    
    def search_similar_documents(
        self,
        query_embedding: Union[List[float], Any],
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity"""
        # Get similar document IDs
        similar_docs = self.vector_store.search(query_embedding, k)
        
        # Get full document data
        results = []
        for doc_id, similarity_score in similar_docs:
            document = self.get_document_complete(doc_id)
            if document:
                document["similarity_score"] = similarity_score
                results.append(document)
        
        return results
    
    def delete_document_complete(self, document_id: str) -> bool:
        """Delete document from all storage systems"""
        success = True
        
        # Delete from vector store
        try:
            self.vector_store.remove_document(document_id)
        except:
            success = False
        
        # Invalidate cache
        self.cache.invalidate_document(document_id)
        
        # Delete from database
        success &= self.database.delete_document(document_id)
        
        return success
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics from all storage systems"""
        return {
            "database": self.database.get_statistics(),
            "cache": self.cache.get_cache_stats(),
            "vector_store": self.vector_store.get_stats()
        }


def create_storage_suite(
    database_config: Optional[Union[Dict[str, Any], DatabaseStorageConfig]] = None,
    cache_config: Optional[Union[Dict[str, Any], CacheManagerConfig]] = None,
    vector_config: Optional[Union[Dict[str, Any], VectorStoreConfig]] = None
) -> Dict[str, Any]:
    """
    Create a complete storage suite with all components.
    
    Args:
        database_config: Database storage configuration
        cache_config: Cache manager configuration
        vector_config: Vector store configuration
    
    Returns:
        Dictionary with all storage components
    """
    return {
        "database": create_database_storage(database_config),
        "cache": create_cache_manager(cache_config),
        "vector_store": create_vector_store(vector_config),
        "manager": StorageManager(database_config, cache_config, vector_config)
    }


# Initialize module
logger.info(f"Initialized storage module v{__version__}")
logger.info("Available storage: Database, Cache, Vector Store")