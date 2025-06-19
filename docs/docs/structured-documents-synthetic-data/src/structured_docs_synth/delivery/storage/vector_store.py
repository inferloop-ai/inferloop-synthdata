"""
Vector store implementation for document embeddings and similarity search.
"""

from __future__ import annotations

import json
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from ...core.config import BaseConfig
from ...core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VectorDocument:
    """Vector document representation"""
    document_id: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    text_content: Optional[str] = None


class VectorStoreConfig(BaseConfig):
    """Vector store configuration"""
    dimension: int = 768
    index_type: str = "flat"  # flat, ivf, hnsw
    metric: str = "cosine"
    max_vectors: int = 1000000


class VectorStore:
    """Vector store for document embeddings and similarity search."""
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """Initialize vector store"""
        self.config = config or VectorStoreConfig()
        self.documents: Dict[str, VectorDocument] = {}
        self.faiss_index = None
        
        if FAISS_AVAILABLE:
            self._init_faiss_index()
        
        logger.info("Initialized VectorStore")
    
    def _init_faiss_index(self):
        """Initialize FAISS index"""
        if not FAISS_AVAILABLE:
            return
        
        dimension = self.config.dimension
        if self.config.index_type == "flat":
            self.faiss_index = faiss.IndexFlatIP(dimension)
        else:
            self.faiss_index = faiss.IndexFlatIP(dimension)
    
    def add_document(
        self,
        document_id: str,
        embedding: Union[np.ndarray, List[float]],
        metadata: Optional[Dict[str, Any]] = None,
        text_content: Optional[str] = None
    ) -> bool:
        """Add document embedding to the vector store"""
        try:
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            
            doc = VectorDocument(
                document_id=document_id,
                embedding=embedding,
                metadata=metadata or {},
                text_content=text_content
            )
            
            self.documents[document_id] = doc
            
            if FAISS_AVAILABLE and self.faiss_index is not None:
                self.faiss_index.add(embedding.reshape(1, -1))
            
            return True
        except Exception as e:
            logger.error(f"Failed to add document {document_id}: {str(e)}")
            return False
    
    def search(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        try:
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            
            if FAISS_AVAILABLE and self.faiss_index is not None:
                scores, indices = self.faiss_index.search(
                    query_embedding.reshape(1, -1), k
                )
                
                results = []
                doc_list = list(self.documents.keys())
                for score, idx in zip(scores[0], indices[0]):
                    if idx != -1 and idx < len(doc_list):
                        results.append((doc_list[idx], float(score)))
                return results
            
            return []
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """Get document by ID"""
        return self.documents.get(document_id)
    
    def count(self) -> int:
        """Get total number of documents"""
        return len(self.documents)


def create_vector_store(
    config: Optional[Union[Dict[str, Any], VectorStoreConfig]] = None
) -> VectorStore:
    """Factory function to create vector store"""
    if isinstance(config, dict):
        config = VectorStoreConfig(**config)
    return VectorStore(config)