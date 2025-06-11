"""
Vector store service for hate speech detection project.
Provides high-level interface for document indexing and semantic search
using the existing embedding service with Qdrant integration.
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import time

# Assuming your embedding service is in the same project
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    High-level vector store service for policy document indexing and search.
    Uses the embedding service with Qdrant for persistent vector storage.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        collection_name: str = "policy_documents",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize the vector store service.

        Args:
            embedding_service: Pre-initialized embedding service (optional)
            collection_name: Name of the Qdrant collection for policy documents
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            model_name: Sentence transformer model name
        """
        self.collection_name = collection_name
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.model_name = model_name

        # Initialize embedding service if not provided
        if embedding_service is None:
            logger.info("Initializing new embedding service for vector store")
            self.embedding_service = EmbeddingService(
                model_name=model_name,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                collection_name=collection_name,
                use_qdrant=True,
                cache_dir="./cache/embeddings",
            )
        else:
            self.embedding_service = embedding_service
            logger.info("Using provided embedding service")

        # Track indexed documents
        self.indexed_documents: Dict[str, Dict[str, Any]] = {}
        self.is_ready = False

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the vector store and check connection."""
        try:
            # Test embedding service
            test_embedding = self.embedding_service.embed_text("test")
            logger.info(
                f"Embedding service initialized successfully. Dimension: {len(test_embedding)}"
            )

            # Get collection info
            collection_info = self.embedding_service.get_qdrant_collection_info()
            if "error" not in collection_info:
                logger.info(f"Connected to Qdrant collection: {collection_info}")
                self.is_ready = True
            else:
                logger.error(
                    f"Failed to connect to Qdrant: {collection_info.get('error')}"
                )

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    def load_policy_documents(self, documents_dir: str) -> List[Dict[str, Any]]:
        """
        Load policy documents from directory.

        Args:
            documents_dir: Directory containing policy document files

        Returns:
            List of document dictionaries
        """
        documents = []
        documents_path = Path(documents_dir)

        if not documents_path.exists():
            raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

        # Supported file extensions
        supported_extensions = {".txt", ".md", ".json"}

        logger.info(f"Loading documents from: {documents_dir}")

        for file_path in documents_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    document = self._load_single_document(file_path)
                    if document:
                        documents.append(document)
                        logger.info(f"Loaded document: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load document {file_path.name}: {str(e)}")

        logger.info(f"Successfully loaded {len(documents)} policy documents")
        return documents

    def _load_single_document(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load a single document file.

        Args:
            file_path: Path to the document file

        Returns:
            Document dictionary or None if failed
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                logger.warning(f"Empty document: {file_path.name}")
                return None

            # Create document metadata
            document = {
                "id": file_path.stem,  # Use filename without extension as ID
                "title": file_path.stem.replace("_", " ").title(),
                "content": content,
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_size": len(content),
                "loaded_at": datetime.now().isoformat(),
                "document_type": "policy_document",
            }

            # If it's a JSON file, try to extract structured data
            if file_path.suffix.lower() == ".json":
                try:
                    json_data = json.loads(content)
                    if isinstance(json_data, dict):
                        # Use JSON content as the text to embed
                        if "content" in json_data:
                            document["content"] = json_data["content"]
                        elif "text" in json_data:
                            document["content"] = json_data["text"]

                        # Add other JSON fields as metadata
                        for key, value in json_data.items():
                            if key not in ["content", "text"] and isinstance(
                                value, (str, int, float, bool)
                            ):
                                document[f"json_{key}"] = value
                except json.JSONDecodeError:
                    # If JSON parsing fails, use raw content
                    pass

            return document

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return None

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32,
        chunk_long_documents: bool = True,
        max_chunk_length: int = 1000,
    ) -> Dict[str, Any]:
        """
        Index documents in the vector store.

        Args:
            documents: List of document dictionaries
            batch_size: Batch size for embedding generation
            chunk_long_documents: Whether to chunk long documents
            max_chunk_length: Maximum length for document chunks

        Returns:
            Indexing results summary
        """
        if not self.is_ready:
            raise RuntimeError("Vector store not ready. Check Qdrant connection.")

        if not documents:
            raise ValueError("No documents provided for indexing")

        logger.info(f"Starting indexing of {len(documents)} documents")
        start_time = time.time()

        # Prepare documents for embedding
        processed_docs = []
        for doc in documents:
            if chunk_long_documents and len(doc.get("content", "")) > max_chunk_length:
                # Split long documents into chunks
                chunks = self._chunk_document(doc, max_chunk_length)
                processed_docs.extend(chunks)
            else:
                processed_docs.append(doc)

        logger.info(f"Processing {len(processed_docs)} document chunks")

        try:
            # Use embedding service to embed and store documents
            embedded_docs = self.embedding_service.embed_documents(
                documents=processed_docs, text_field="content", store_in_qdrant=True
            )

            # Track indexed documents
            for doc in embedded_docs:
                self.indexed_documents[doc["id"]] = {
                    "title": doc.get("title", ""),
                    "document_type": doc.get("document_type", ""),
                    "indexed_at": datetime.now().isoformat(),
                    "embedding_model": doc.get("embedding_model", ""),
                    "chunk_count": 1,
                }

            # Update chunk counts for chunked documents
            chunk_counts = {}
            for doc in embedded_docs:
                original_id = doc["id"].split("_chunk_")[0]
                chunk_counts[original_id] = chunk_counts.get(original_id, 0) + 1

            for original_id, count in chunk_counts.items():
                if original_id in self.indexed_documents:
                    self.indexed_documents[original_id]["chunk_count"] = count

            processing_time = time.time() - start_time

            # Prepare results summary
            results = {
                "success": True,
                "documents_processed": len(documents),
                "chunks_created": len(processed_docs),
                "embeddings_created": len(embedded_docs),
                "processing_time": processing_time,
                "documents_per_second": len(embedded_docs) / processing_time,
                "collection_name": self.collection_name,
                "embedding_model": self.model_name,
                "indexed_at": datetime.now().isoformat(),
            }

            logger.info(f"Indexing completed successfully in {processing_time:.2f}s")
            logger.info(f"Indexed {len(embedded_docs)} document chunks")

            return results

        except Exception as e:
            logger.error(f"Failed to index documents: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "documents_processed": 0,
                "processing_time": time.time() - start_time,
            }

    def _chunk_document(
        self, document: Dict[str, Any], max_length: int
    ) -> List[Dict[str, Any]]:
        """
        Split a document into smaller chunks.

        Args:
            document: Document dictionary
            max_length: Maximum length per chunk

        Returns:
            List of document chunks
        """
        content = document.get("content", "")
        if len(content) <= max_length:
            return [document]

        chunks = []
        words = content.split()
        current_chunk = []
        current_length = 0
        chunk_index = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space

            if current_length + word_length > max_length and current_chunk:
                # Create chunk
                chunk_content = " ".join(current_chunk)
                chunk_doc = document.copy()
                chunk_doc["id"] = f"{document['id']}_chunk_{chunk_index}"
                chunk_doc["content"] = chunk_content
                chunk_doc["chunk_index"] = chunk_index
                chunk_doc["is_chunk"] = True
                chunk_doc["original_document_id"] = document["id"]
                chunks.append(chunk_doc)

                # Reset for next chunk
                current_chunk = [word]
                current_length = word_length
                chunk_index += 1
            else:
                current_chunk.append(word)
                current_length += word_length

        # Add final chunk
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            chunk_doc = document.copy()
            chunk_doc["id"] = f"{document['id']}_chunk_{chunk_index}"
            chunk_doc["content"] = chunk_content
            chunk_doc["chunk_index"] = chunk_index
            chunk_doc["is_chunk"] = True
            chunk_doc["original_document_id"] = document["id"]
            chunks.append(chunk_doc)

        logger.debug(f"Split document '{document['id']}' into {len(chunks)} chunks")
        return chunks

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
        include_content: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform semantic search in the vector store.

        Args:
            query: Search query text
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            filter_conditions: Optional filter conditions
            include_content: Whether to include full content in results

        Returns:
            Search results with metadata
        """
        if not self.is_ready:
            raise RuntimeError("Vector store not ready. Check Qdrant connection.")

        if not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"Performing semantic search for: '{query[:100]}...'")
        start_time = time.time()

        try:
            # Use embedding service's semantic search
            search_results = self.embedding_service.semantic_search(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                filter_conditions=filter_conditions,
            )

            search_time = time.time() - start_time

            # Format results
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    "id": result["id"],
                    "score": result["score"],
                    "title": result.get("metadata", {}).get("title", "Unknown"),
                    "document_type": result.get("metadata", {}).get(
                        "document_type", "unknown"
                    ),
                    "filename": result.get("metadata", {}).get("filename", ""),
                    "is_chunk": result.get("metadata", {}).get("is_chunk", False),
                    "chunk_index": result.get("metadata", {}).get("chunk_index"),
                    "original_document_id": result.get("metadata", {}).get(
                        "original_document_id"
                    ),
                }

                if include_content:
                    formatted_result["content"] = result.get("text", "")
                    formatted_result["content_preview"] = (
                        result.get("text", "")[:200] + "..."
                        if len(result.get("text", "")) > 200
                        else result.get("text", "")
                    )

                # Add all other metadata
                for key, value in result.get("metadata", {}).items():
                    if key not in formatted_result:
                        formatted_result[key] = value

                formatted_results.append(formatted_result)

            # Prepare response
            response = {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "search_time": search_time,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "collection_name": self.collection_name,
                "searched_at": datetime.now().isoformat(),
            }

            logger.info(
                f"Search completed in {search_time:.3f}s, found {len(formatted_results)} results"
            )
            return response

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": str(e),
                "search_time": time.time() - start_time,
            }

    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.

        Args:
            document_id: Document ID to retrieve

        Returns:
            Document data or None if not found
        """
        try:
            # Search for the specific document
            results = self.search(
                query="",  # Empty query to match by ID filter
                top_k=1,
                filter_conditions={"id": document_id},
            )

            if results["results"]:
                return results["results"][0]
            else:
                logger.warning(f"Document not found: {document_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve document {document_id}: {str(e)}")
            return None

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.

        Returns:
            Collection statistics
        """
        try:
            # Get Qdrant collection info
            qdrant_info = self.embedding_service.get_qdrant_collection_info()

            # Get embedding service info
            model_info = self.embedding_service.get_model_info()

            stats = {
                "collection_name": self.collection_name,
                "total_vectors": qdrant_info.get("vectors_count", 0),
                "indexed_vectors": qdrant_info.get("indexed_vectors_count", 0),
                "total_points": qdrant_info.get("points_count", 0),
                "collection_status": qdrant_info.get("status", "unknown"),
                "embedding_model": model_info.get("model_name", ""),
                "embedding_dimension": model_info.get("embedding_dimension", 0),
                "cache_size": model_info.get("cache_size", 0),
                "indexed_documents_count": len(self.indexed_documents),
                "qdrant_enabled": model_info.get("qdrant_enabled", False),
                "last_updated": datetime.now().isoformat(),
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.

        Args:
            document_id: ID of the document to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            # Find all chunks of the document
            point_ids = [document_id]

            # If it's a chunked document, find all chunks
            if document_id in self.indexed_documents:
                chunk_count = self.indexed_documents[document_id].get("chunk_count", 1)
                if chunk_count > 1:
                    for i in range(chunk_count):
                        chunk_id = f"{document_id}_chunk_{i}"
                        point_ids.append(chunk_id)

            # Delete from Qdrant
            success = self.embedding_service.delete_from_qdrant(point_ids)

            if success:
                # Remove from local tracking
                if document_id in self.indexed_documents:
                    del self.indexed_documents[document_id]
                logger.info(f"Successfully deleted document: {document_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            return False

    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all point IDs
            stats = self.get_collection_stats()
            total_points = stats.get("total_points", 0)

            if total_points == 0:
                logger.info("Collection is already empty")
                return True

            # This would require getting all point IDs first
            # For now, we'll recreate the collection
            logger.warning(
                "Collection clearing not implemented. Consider recreating the collection."
            )
            return False

        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the vector store.

        Returns:
            Health status information
        """
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "checks": {},
        }

        try:
            # Check embedding service
            try:
                test_embedding = self.embedding_service.embed_text("health check")
                health_status["checks"]["embedding_service"] = {
                    "status": "healthy",
                    "embedding_dimension": len(test_embedding),
                }
            except Exception as e:
                health_status["checks"]["embedding_service"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }

            # Check Qdrant connection
            qdrant_info = self.embedding_service.get_qdrant_collection_info()
            if "error" in qdrant_info:
                health_status["checks"]["qdrant"] = {
                    "status": "unhealthy",
                    "error": qdrant_info["error"],
                }
            else:
                health_status["checks"]["qdrant"] = {
                    "status": "healthy",
                    "vectors_count": qdrant_info.get("vectors_count", 0),
                }

            # Overall status
            all_healthy = all(
                check["status"] == "healthy"
                for check in health_status["checks"].values()
            )
            health_status["overall_status"] = "healthy" if all_healthy else "unhealthy"

        except Exception as e:
            health_status["overall_status"] = "error"
            health_status["error"] = str(e)

        return health_status


# Utility function for easy initialization
def create_vector_store(
    documents_dir: str,
    collection_name: str = "policy_documents",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_documents: bool = True,
) -> VectorStoreService:
    """
    Create and initialize a vector store with policy documents.

    Args:
        documents_dir: Directory containing policy documents
        collection_name: Name for the Qdrant collection
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        model_name: Sentence transformer model name
        index_documents: Whether to automatically index documents

    Returns:
        Initialized VectorStoreService
    """
    logger.info(f"Creating vector store for documents in: {documents_dir}")

    # Initialize vector store
    vector_store = VectorStoreService(
        collection_name=collection_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        model_name=model_name,
    )

    if index_documents:
        # Load and index documents
        documents = vector_store.load_policy_documents(documents_dir)
        if documents:
            indexing_results = vector_store.index_documents(documents)
            logger.info(f"Indexing results: {indexing_results}")
        else:
            logger.warning("No documents found to index")

    return vector_store
