"""
Vector store service using FAISS for efficient similarity search.
Handles document indexing, storage, and retrieval operations.
"""

import logging
import numpy as np
import faiss
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store for efficient similarity search.
    Supports document indexing, persistence, and retrieval.
    """
    
    def __init__(
        self,
        embedding_service,  # Changed from specific import to generic
        storage_path: str = "./vector_db",
        index_type: str = "flat"
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_service: Service for creating embeddings
            storage_path: Path to store the index and metadata
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        self.embedding_service = embedding_service
        self.storage_path = Path(storage_path)
        self.index_type = index_type
        
        # Initialize storage
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # FAISS index and metadata
        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict[str, Any]] = []
        self.document_metadata: Dict[int, Dict[str, Any]] = {}
        
        # Index configuration
        self.embedding_dim = embedding_service.embedding_dim
        self.is_trained = False
        
        logger.info(f"Initialized vector store at {storage_path}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        logger.info(f"Index type: {index_type}")
    
    def _create_index(self, num_documents: int = 0) -> faiss.Index:
        """
        Create a FAISS index based on the specified type.
        
        Args:
            num_documents: Estimated number of documents (for optimization)
            
        Returns:
            FAISS index object
        """
        try:
            if self.index_type == "flat":
                # Exact search using L2 distance
                index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
                logger.info("Created flat index for exact search")
                
            elif self.index_type == "ivf":
                # Inverted file index for faster approximate search
                nlist = min(100, max(10, num_documents // 100))  # Number of clusters
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                logger.info(f"Created IVF index with {nlist} clusters")
                
            elif self.index_type == "hnsw":
                # Hierarchical Navigable Small World for very fast approximate search
                M = 32  # Number of connections
                index = faiss.IndexHNSWFlat(self.embedding_dim, M)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 100
                logger.info(f"Created HNSW index with M={M}")
                
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            return index
            
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            raise
    
    def add_documents(
        self, 
        documents: List[Dict[str, Any]], 
        batch_size: int = 100
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'content' field
            batch_size: Batch size for processing
        """
        if not documents:
            logger.warning("No documents provided to add")
            return
        
        try:
            start_time = time.time()
            logger.info(f"Adding {len(documents)} documents to vector store")
            
            # Create embeddings for documents
            embedded_docs = self.embedding_service.embed_documents(documents)
            
            if not embedded_docs:
                raise ValueError("No valid documents could be embedded")
            
            # Initialize index if not exists
            if self.index is None:
                self.index = self._create_index(len(embedded_docs))
            
            # Prepare embeddings for FAISS
            embeddings = np.array([doc['embedding'] for doc in embedded_docs]).astype('float32')
            
            # Train index if needed (for IVF)
            if self.index_type == "ivf" and not self.is_trained:
                if len(embeddings) >= 100:  # Need sufficient data for training
                    logger.info("Training IVF index...")
                    self.index.train(embeddings)
                    self.is_trained = True
                else:
                    logger.warning("Insufficient data for IVF training, using flat index")
                    self.index = self._create_index()
            
            # Add to index
            start_idx = len(self.documents)
            self.index.add(embeddings)
            
            # Store documents and metadata
            for i, doc in enumerate(embedded_docs):
                doc_id = start_idx + i
                
                # Store document
                self.documents.append({
                    'id': doc_id,
                    'content': doc.get('content', ''),
                    'title': doc.get('title', f'Document {doc_id}'),
                    'source': doc.get('source', 'unknown'),
                    'added_at': datetime.utcnow().isoformat()
                })
                
                # Store metadata
                self.document_metadata[doc_id] = {
                    'embedding_model': doc.get('embedding_model', self.embedding_service.model_name),
                    'embedding_dim': doc.get('embedding_dim', self.embedding_dim),
                    'content_length': len(doc.get('content', '')),
                    'section': doc.get('section'),
                    'policy_type': doc.get('policy_type')
                }
            
            processing_time = time.time() - start_time
            logger.info(f"Successfully added {len(embedded_docs)} documents in {processing_time:.2f}s")
            logger.info(f"Total documents in store: {len(self.documents)}")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using text query.
        
        Args:
            query: Text query to search for
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar documents with scores
        """
        if not self.index or len(self.documents) == 0:
            logger.warning("Vector store is empty")
            return []
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            start_time = time.time()
            
            # Create query embedding
            query_embedding = self.embedding_service.embed_text(query.strip())
            query_vector = query_embedding.reshape(1, -1).astype('float32')
            
            # Perform search
            search_k = min(top_k * 2, len(self.documents))  # Get more results for filtering
            scores, indices = self.index.search(query_vector, search_k)
            
            # Process results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid results
                    continue
                
                # Convert FAISS inner product score to cosine similarity
                similarity_score = float(score)
                
                if similarity_score < score_threshold:
                    continue
                
                # Get document
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = similarity_score
                    doc['metadata'] = self.document_metadata.get(idx, {})
                    
                    # Apply metadata filters
                    if filter_metadata:
                        if not self._matches_filter(doc, filter_metadata):
                            continue
                    
                    results.append(doc)
            
            # Sort by similarity and limit results
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            results = results[:top_k]
            
            search_time = time.time() - start_time
            logger.debug(f"Search completed in {search_time:.3f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
    
    def _matches_filter(self, document: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Check if document matches the provided filters.
        
        Args:
            document: Document to check
            filters: Filter criteria
            
        Returns:
            True if document matches all filters
        """
        for key, value in filters.items():
            if key in document:
                if document[key] != value:
                    return False
            elif key in document.get('metadata', {}):
                if document['metadata'][key] != value:
                    return False
            else:
                return False  # Filter key not found
        
        return True
    
    def get_document_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document dictionary or None if not found
        """
        if 0 <= doc_id < len(self.documents):
            doc = self.documents[doc_id].copy()
            doc['metadata'] = self.document_metadata.get(doc_id, {})
            return doc
        return None
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents in the store.
        
        Returns:
            List of all documents
        """
        results = []
        for i, doc in enumerate(self.documents):
            doc_copy = doc.copy()
            doc_copy['metadata'] = self.document_metadata.get(i, {})
            results.append(doc_copy)
        return results
    
    def save_index(self, filename: Optional[str] = None) -> str:
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path where the index was saved
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if filename is None:
                filename = f"index_{timestamp}"
            
            # Save FAISS index
            index_path = self.storage_path / f"{filename}.faiss"
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata = {
                'documents': self.documents,
                'document_metadata': self.document_metadata,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'is_trained': self.is_trained,
                'created_at': datetime.utcnow().isoformat(),
                'embedding_model': self.embedding_service.model_name,
                'total_documents': len(self.documents)
            }
            
            metadata_path = self.storage_path / f"{filename}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Index saved to {index_path}")
            logger.info(f"Metadata saved to {metadata_path}")
            
            return str(index_path)
            
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise
    
    def load_index(self, filename: str) -> None:
        """
        Load a FAISS index and metadata from disk.
        
        Args:
            filename: Name of the index file (without extension)
        """
        try:
            # Load FAISS index
            index_path = self.storage_path / f"{filename}.faiss"
            if not index_path.exists():
                raise FileNotFoundError(f"Index file not found: {index_path}")
            
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            metadata_path = self.storage_path / f"{filename}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.documents = metadata.get('documents', [])
                self.document_metadata = {
                    int(k): v for k, v in metadata.get('document_metadata', {}).items()
                }
                self.is_trained = metadata.get('is_trained', False)
                
                # Verify compatibility
                saved_dim = metadata.get('embedding_dim')
                if saved_dim and saved_dim != self.embedding_dim:
                    logger.warning(f"Embedding dimension mismatch: saved={saved_dim}, current={self.embedding_dim}")
                
                saved_model = metadata.get('embedding_model')
                if saved_model and saved_model != self.embedding_service.model_name:
                    logger.warning(f"Embedding model mismatch: saved={saved_model}, current={self.embedding_service.model_name}")
            
            logger.info(f"Successfully loaded index from {index_path}")
            logger.info(f"Loaded {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary containing store statistics
        """
        stats = {
            'total_documents': len(self.documents),
            'embedding_dimension': self.embedding_dim,
            'index_type': self.index_type,
            'is_trained': self.is_trained,
            'storage_path': str(self.storage_path),
            'embedding_model': self.embedding_service.model_name
        }
        
        if self.index:
            stats['index_size'] = self.index.ntotal
            stats['index_is_trained'] = self.index.is_trained
        
        # Document source distribution
        source_counts = {}
        for doc in self.documents:
            source = doc.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        stats['source_distribution'] = source_counts
        
        return stats
    
    def clear(self) -> None:
        """Clear all documents and reset the index."""
        self.index = None
        self.documents.clear()
        self.document_metadata.clear()
        self.is_trained = False
        logger.info("Vector store cleared")
    
    def remove_document(self, doc_id: int) -> bool:
        """
        Remove a document from the vector store.
        Note: This is a simplified implementation. For production use,
        consider rebuilding the index for better performance.
        
        Args:
            doc_id: ID of the document to remove
            
        Returns:
            True if document was removed, False if not found
        """
        try:
            if doc_id < 0 or doc_id >= len(self.documents):
                logger.warning(f"Document ID {doc_id} not found")
                return False
            
            # Remove from documents list
            removed_doc = self.documents.pop(doc_id)
            self.document_metadata.pop(doc_id, None)
            
            # Update IDs for remaining documents
            for i in range(doc_id, len(self.documents)):
                self.documents[i]['id'] = i
                if i + 1 in self.document_metadata:
                    self.document_metadata[i] = self.document_metadata.pop(i + 1)
            
            logger.info(f"Removed document: {removed_doc.get('title', f'ID {doc_id}')}")
            logger.warning("Index rebuild recommended after document removal")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove document {doc_id}: {str(e)}")
            return False
    
    def rebuild_index(self) -> None:
        """
        Rebuild the FAISS index from scratch.
        Useful after removing documents or changing index parameters.
        """
        if not self.documents:
            logger.warning("No documents to rebuild index")
            return
        
        try:
            logger.info("Rebuilding index from existing documents...")
            
            # Extract content from stored documents
            texts = [doc['content'] for doc in self.documents]
            
            # Create new embeddings
            embeddings_list = self.embedding_service.embed_batch(texts, show_progress=True)
            embeddings = np.array(embeddings_list).astype('float32')
            
            # Create new index
            self.index = self._create_index(len(embeddings))
            
            # Train if needed
            if self.index_type == "ivf" and len(embeddings) >= 100:
                logger.info("Training new index...")
                self.index.train(embeddings)
                self.is_trained = True
            
            # Add all embeddings
            self.index.add(embeddings)
            
            logger.info(f"Successfully rebuilt index with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {str(e)}")
            raise
    
    def update_document(self, doc_id: int, updated_content: str) -> bool:
        """
        Update a document's content and re-embed it.
        Note: This requires rebuilding the index for proper functionality.
        
        Args:
            doc_id: ID of the document to update
            updated_content: New content for the document
            
        Returns:
            True if document was updated, False if not found
        """
        try:
            if doc_id < 0 or doc_id >= len(self.documents):
                logger.warning(f"Document ID {doc_id} not found")
                return False
            
            # Update document content
            old_content = self.documents[doc_id]['content']
            self.documents[doc_id]['content'] = updated_content
            self.documents[doc_id]['updated_at'] = datetime.utcnow().isoformat()
            
            # Update metadata
            if doc_id in self.document_metadata:
                self.document_metadata[doc_id]['content_length'] = len(updated_content)
            
            logger.info(f"Updated document {doc_id} content")
            logger.warning("Index rebuild recommended for updated document to take effect in search")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {str(e)}")
            return False
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search using a pre-computed embedding vector.
        
        Args:
            query_embedding: Pre-computed embedding vector
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of similar documents with scores
        """
        if not self.index or len(self.documents) == 0:
            logger.warning("Vector store is empty")
            return []
        
        try:
            start_time = time.time()
            
            # Prepare query vector
            query_vector = query_embedding.reshape(1, -1).astype('float32')
            
            # Perform search
            scores, indices = self.index.search(query_vector, top_k)
            
            # Process results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid results
                    continue
                
                similarity_score = float(score)
                
                if similarity_score < score_threshold:
                    continue
                
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = similarity_score
                    doc['metadata'] = self.document_metadata.get(idx, {})
                    results.append(doc)
            
            search_time = time.time() - start_time
            logger.debug(f"Embedding search completed in {search_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Embedding search failed: {str(e)}")
            raise


# Utility functions for integration
def create_vector_store_from_documents(
    documents: List[Dict[str, Any]],
    embedding_service,
    storage_path: str = "./vector_db",
    index_type: str = "flat"
) -> VectorStore:
    """
    Create and populate a vector store from a list of documents.
    
    Args:
        documents: List of document dictionaries
        embedding_service: Embedding service instance
        storage_path: Path to store the index
        index_type: Type of FAISS index to create
        
    Returns:
        Populated VectorStore instance
    """
    try:
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        # Create vector store
        vector_store = VectorStore(
            embedding_service=embedding_service,
            storage_path=storage_path,
            index_type=index_type
        )
        
        # Add documents
        vector_store.add_documents(documents)
        
        # Save the index
        index_path = vector_store.save_index("policy_documents")
        logger.info(f"Vector store created and saved to {index_path}")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Failed to create vector store: {str(e)}")
        raise


def load_vector_store(
    embedding_service,
    storage_path: str = "./vector_db",
    index_filename: str = "policy_documents"
) -> VectorStore:
    """
    Load an existing vector store from disk.
    
    Args:
        embedding_service: Embedding service instance
        storage_path: Path where the index is stored
        index_filename: Name of the index file to load
        
    Returns:
        Loaded VectorStore instance
    """
    try:
        vector_store = VectorStore(
            embedding_service=embedding_service,
            storage_path=storage_path
        )
        
        vector_store.load_index(index_filename)
        logger.info("Vector store loaded successfully")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Failed to load vector store: {str(e)}")
        raise