"""
Embedding service for converting text to vector representations.
Uses sentence-transformers for high-quality semantic embeddings.
Now includes Qdrant vector database integration for persistent storage.
"""

import logging
import numpy as np
from typing import List, Union, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
import pickle
import hashlib
import time
import re
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    SearchRequest,
    Filter,
    FieldCondition,
    MatchValue,
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for creating and managing text embeddings using sentence-transformers.
    Supports both single text and batch processing with caching capabilities.
    Now includes Qdrant vector database integration for persistent storage.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_seq_length: int = 512,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "embeddings",
        use_qdrant: bool = True,
    ):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformer model to use
            cache_dir: Directory to cache embeddings (optional)
            device: Device to run on ('cpu', 'cuda', 'auto')
            batch_size: Batch size for processing multiple texts
            max_seq_length: Maximum sequence length for the model
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Name of the Qdrant collection
            use_qdrant: Whether to use Qdrant for vector storage
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Qdrant configuration
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = self._sanitize_collection_name(collection_name)
        self.use_qdrant = use_qdrant
        self.qdrant_client = None

        # Setup device
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing embedding service with model: {model_name}")
        logger.info(f"Using device: {self.device}")

        try:
            # Load the sentence transformer model
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model.max_seq_length = max_seq_length

            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Model loaded successfully. Embedding dimension: {self.embedding_dim}"
            )

            # Initialize Qdrant client if enabled
            if self.use_qdrant:
                self._init_qdrant()

            # Load cache if it exists
            self._load_cache()

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise

    def _init_qdrant(self) -> None:
        """Initialize Qdrant client and create collection if it doesn't exist."""
        try:
            logger.info(
                f"Connecting to Qdrant at {self.qdrant_host}:{self.qdrant_port}"
            )
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host, port=self.qdrant_port
            )

            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim, distance=Distance.COSINE
                    ),
                )
                logger.info(f"Collection '{self.collection_name}' created successfully")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")

            # Get collection info
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            logger.info(
                f"Collection info - vectors count: {collection_info.vectors_count}, "
                f"indexed vectors: {collection_info.indexed_vectors_count}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {str(e)}")
            if "collection name cannot contain" in str(e):
                logger.error(f"Invalid collection name '{self.collection_name}'. Collection names cannot contain special characters like <, >, /, \\, |, ?, *, or spaces.")
                # Try with a sanitized name
                self.collection_name = self._sanitize_collection_name(self.collection_name)
                logger.info(f"Retrying with sanitized collection name: '{self.collection_name}'")
                # You could add a retry logic here if needed
            logger.warning("Continuing without Qdrant vector storage")
            self.use_qdrant = False
            self.qdrant_client = None

    def _sanitize_collection_name(self, name: str) -> str:
        """Sanitize collection name to remove invalid characters."""
        # Handle case where non-string is passed
        if not isinstance(name, str):
            logger.warning(f"Collection name should be string, got {type(name)}. Using default.")
            name = "embeddings"
        
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>"/\\|?*\s]', '_', name)
        # Ensure it doesn't start with underscore and is not empty
        sanitized = sanitized.strip('_')
        if not sanitized:
            sanitized = 'default_collection'
        return sanitized

    def _get_cache_key(self, text: str) -> str:
        """Generate a unique cache key for a text."""
        return hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()

    def _load_cache(self) -> None:
        """Load embedding cache from disk if it exists."""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / "embedding_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self._embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {str(e)}")
                self._embedding_cache = {}

    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        if not self.cache_dir or not self._embedding_cache:
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / "embedding_cache.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(self._embedding_cache, f)
            logger.debug(f"Saved {len(self._embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {str(e)}")

    def store_embedding_in_qdrant(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        point_id: Optional[str] = None,
    ) -> str:
        """
        Store an embedding in Qdrant vector database.

        Args:
            text: Original text
            embedding: Embedding vector
            metadata: Additional metadata to store
            point_id: Custom point ID (if None, generates UUID)

        Returns:
            Point ID of the stored embedding
        """
        if not self.use_qdrant or not self.qdrant_client:
            logger.warning("Qdrant not available, skipping vector storage")
            return ""

        try:
            if point_id is None:
                point_id = str(uuid.uuid4())

            # Prepare payload
            payload = {
                "text": text,
                "model_name": self.model_name,
                "embedding_dim": self.embedding_dim,
                "created_at": time.time(),
            }

            # Add custom metadata
            if metadata:
                payload.update(metadata)

            # Create point - FIXED: Use unnamed vector
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),  # <-- FIXED: Direct vector, not named
                payload=payload,
            )

            # Upsert to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=[point]
            )

            logger.debug(f"Stored embedding in Qdrant with ID: {point_id}")
            return point_id

        except Exception as e:
            logger.error(f"Failed to store embedding in Qdrant: {str(e)}")
            return ""

    def store_embeddings_batch_in_qdrant(
        self,
        texts: List[str],
        embeddings: List[np.ndarray],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        point_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Store multiple embeddings in Qdrant vector database.

        Args:
            texts: List of original texts
            embeddings: List of embedding vectors
            metadata_list: List of metadata dictionaries
            point_ids: List of custom point IDs

        Returns:
            List of point IDs of the stored embeddings
        """
        if not self.use_qdrant or not self.qdrant_client:
            logger.warning("Qdrant not available, skipping vector storage")
            return []

        try:
            if len(texts) != len(embeddings):
                raise ValueError("Number of texts and embeddings must match")

            points = []
            stored_ids = []

            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                # Generate or use provided point ID
                if point_ids and i < len(point_ids):
                    point_id = point_ids[i]
                else:
                    point_id = str(uuid.uuid4())

                # Prepare payload
                payload = {
                    "text": text,
                    "model_name": self.model_name,
                    "embedding_dim": self.embedding_dim,
                    "created_at": time.time(),
                }

                # Add custom metadata
                if metadata_list and i < len(metadata_list) and metadata_list[i]:
                    payload.update(metadata_list[i])

                # Create point - FIXED: Use unnamed vector
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),  # <-- FIXED: Direct vector, not named
                    payload=payload,
                )

                points.append(point)
                stored_ids.append(point_id)

            # Batch upsert to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=points
            )

            logger.info(f"Stored {len(points)} embeddings in Qdrant batch operation")
            return stored_ids

        except Exception as e:
            logger.error(f"Failed to store embeddings batch in Qdrant: {str(e)}")
            return []

    def search_similar_in_qdrant(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings in Qdrant.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            filter_conditions: Optional filter conditions

        Returns:
            List of similar documents with metadata and scores
        """
        if not self.use_qdrant or not self.qdrant_client:
            logger.warning("Qdrant not available, cannot perform search")
            return []

        try:
            # Prepare search filter
            search_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                search_filter = Filter(must=conditions)

            # Perform search - FIXED: Use unnamed vector
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),  # <-- FIXED: Direct vector, no tuple
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False,  # Don't return vectors to save bandwidth
            )

            # Format results
            results = []
            for result in search_results:
                results.append(
                    {
                        "id": result.id,
                        "score": result.score,
                        "text": result.payload.get("text", ""),
                        "metadata": {
                            k: v for k, v in result.payload.items() if k != "text"
                        },
                    }
                )

            logger.debug(f"Found {len(results)} similar embeddings in Qdrant")
            return results

        except Exception as e:
            logger.error(f"Failed to search similar embeddings in Qdrant: {str(e)}")
            return []

    def embed_text(
        self,
        text: str,
        use_cache: bool = True,
        store_in_qdrant: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Create embedding for a single text.

        Args:
            text: Input text to embed
            use_cache: Whether to use caching
            store_in_qdrant: Whether to store in Qdrant
            metadata: Additional metadata for Qdrant storage

        Returns:
            Numpy array representing the text embedding
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        text = text.strip()

        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                embedding = self._embedding_cache[cache_key]
                if store_in_qdrant:
                    self.store_embedding_in_qdrant(text, embedding, metadata)
                return embedding

        try:
            start_time = time.time()

            # Create embedding
            embedding = self.model.encode(
                text, convert_to_numpy=True, normalize_embeddings=True
            )

            processing_time = time.time() - start_time
            logger.debug(
                f"Embedded text in {processing_time:.3f}s (length: {len(text)})"
            )

            # Cache the result
            if use_cache:
                self._embedding_cache[cache_key] = embedding
                if len(self._embedding_cache) % 100 == 0:  # Periodic cache save
                    self._save_cache()

            # Store in Qdrant if requested
            if store_in_qdrant:
                point_id = self.store_embedding_in_qdrant(text, embedding, metadata)
                logger.debug(f"Stored embedding in Qdrant with ID: {point_id}")

            return embedding

        except Exception as e:
            logger.error(f"Failed to embed text: {str(e)}")
            raise

    def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        show_progress: bool = False,
        store_in_qdrant: bool = False,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[np.ndarray]:
        """
        Create embeddings for a batch of texts efficiently.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use caching
            show_progress: Whether to show progress bar
            store_in_qdrant: Whether to store in Qdrant
            metadata_list: List of metadata for Qdrant storage

        Returns:
            List of numpy arrays representing text embeddings
        """
        if not texts:
            return []

        # Clean texts
        cleaned_texts = [text.strip() for text in texts if text and text.strip()]
        if not cleaned_texts:
            raise ValueError("No valid texts provided")

        embeddings = []
        texts_to_embed = []
        cache_keys = []
        indices_to_compute = []

        # Check cache for each text
        for i, text in enumerate(cleaned_texts):
            if use_cache:
                cache_key = self._get_cache_key(text)
                if cache_key in self._embedding_cache:
                    embeddings.append(self._embedding_cache[cache_key])
                    cache_keys.append(None)
                else:
                    embeddings.append(None)  # Placeholder
                    texts_to_embed.append(text)
                    cache_keys.append(cache_key)
                    indices_to_compute.append(i)
            else:
                embeddings.append(None)  # Placeholder
                texts_to_embed.append(text)
                cache_keys.append(None)
                indices_to_compute.append(i)

        # Compute embeddings for uncached texts
        if texts_to_embed:
            try:
                start_time = time.time()

                logger.info(f"Computing embeddings for {len(texts_to_embed)} texts")

                # Batch encode
                new_embeddings = self.model.encode(
                    texts_to_embed,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=show_progress,
                )

                processing_time = time.time() - start_time
                logger.info(f"Batch embedding completed in {processing_time:.3f}s")

                # Update results and cache
                for i, (idx, embedding) in enumerate(
                    zip(indices_to_compute, new_embeddings)
                ):
                    embeddings[idx] = embedding

                    # Cache the result
                    if (
                        use_cache
                        and i < len(cache_keys)
                        and cache_keys[idx] is not None
                    ):
                        self._embedding_cache[cache_keys[idx]] = embedding

                # Save cache
                if use_cache:
                    self._save_cache()

            except Exception as e:
                logger.error(f"Failed to create batch embeddings: {str(e)}")
                raise

        # Store in Qdrant if requested
        if store_in_qdrant and embeddings:
            try:
                logger.info(f"Storing {len(embeddings)} embeddings in Qdrant")
                point_ids = self.store_embeddings_batch_in_qdrant(
                    cleaned_texts, embeddings, metadata_list
                )
                logger.info(
                    f"Successfully stored {len(point_ids)} embeddings in Qdrant"
                )
            except Exception as e:
                logger.error(f"Failed to store batch embeddings in Qdrant: {str(e)}")

        return embeddings

    def embed_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "content",
        store_in_qdrant: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Embed a list of document dictionaries.

        Args:
            documents: List of document dictionaries
            text_field: Field name containing the text to embed
            store_in_qdrant: Whether to store embeddings in Qdrant

        Returns:
            List of documents with added 'embedding' field
        """
        if not documents:
            return []

        try:
            # Extract texts
            texts = []
            valid_docs = []
            metadata_list = []

            for doc in documents:
                if text_field in doc and doc[text_field]:
                    texts.append(doc[text_field])
                    valid_docs.append(doc.copy())

                    # Prepare metadata for Qdrant (exclude the text field and embedding)
                    if store_in_qdrant:
                        metadata = {
                            k: v
                            for k, v in doc.items()
                            if k != text_field and k != "embedding"
                        }
                        metadata_list.append(metadata)
                else:
                    logger.warning(
                        f"Document missing '{text_field}' field or empty content"
                    )

            if not texts:
                raise ValueError(f"No valid documents with '{text_field}' field found")

            # Create embeddings
            embeddings = self.embed_batch(
                texts,
                show_progress=True,
                store_in_qdrant=store_in_qdrant,
                metadata_list=metadata_list if store_in_qdrant else None,
            )

            # Add embeddings to documents
            for doc, embedding in zip(valid_docs, embeddings):
                doc["embedding"] = embedding
                doc["embedding_model"] = self.model_name
                doc["embedding_dim"] = self.embedding_dim

            logger.info(f"Successfully embedded {len(valid_docs)} documents")
            if store_in_qdrant:
                logger.info(
                    f"Documents stored in Qdrant collection: {self.collection_name}"
                )

            return valid_docs

        except Exception as e:
            logger.error(f"Failed to embed documents: {str(e)}")
            raise

    def compute_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Failed to compute similarity: {str(e)}")
            return 0.0

    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        document_embeddings: List[np.ndarray],
        top_k: int = 5,
    ) -> List[tuple]:
        """
        Find most similar documents to a query embedding.

        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embedding vectors
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score) tuples sorted by similarity
        """
        if not document_embeddings:
            return []

        try:
            similarities = []

            for i, doc_embedding in enumerate(document_embeddings):
                similarity = self.compute_similarity(query_embedding, doc_embedding)
                similarities.append((i, similarity))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Return top-k results
            return similarities[:top_k]

        except Exception as e:
            logger.error(f"Failed to find similar documents: {str(e)}")
            return []

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using query text.

        Args:
            query: Search query text
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            filter_conditions: Optional filter conditions

        Returns:
            List of similar documents with metadata and scores
        """
        try:
            # Create embedding for query
            query_embedding = self.embed_text(
                query, use_cache=True, store_in_qdrant=False
            )

            # Search in Qdrant
            results = self.search_similar_in_qdrant(
                query_embedding,
                top_k=top_k,
                score_threshold=score_threshold,
                filter_conditions=filter_conditions,
            )

            logger.info(
                f"Semantic search for '{query[:50]}...' returned {len(results)} results"
            )
            return results

        except Exception as e:
            logger.error(f"Failed to perform semantic search: {str(e)}")
            return []

    def get_qdrant_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the Qdrant collection.

        Returns:
            Dictionary containing collection information
        """
        if not self.use_qdrant or not self.qdrant_client:
            return {"error": "Qdrant not available"}

        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "status": collection_info.status,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {"error": str(e)}

    def delete_from_qdrant(self, point_ids: List[str]) -> bool:
        """
        Delete points from Qdrant collection.

        Args:
            point_ids: List of point IDs to delete

        Returns:
            True if successful, False otherwise
        """
        if not self.use_qdrant or not self.qdrant_client:
            logger.warning("Qdrant not available, cannot delete points")
            return False

        try:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids),
            )
            logger.info(f"Deleted {len(point_ids)} points from Qdrant")
            return True

        except Exception as e:
            logger.error(f"Failed to delete points from Qdrant: {str(e)}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model and Qdrant connection.

        Returns:
            Dictionary containing model and service information
        """
        info = {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": self.max_seq_length,
            "device": self.device,
            "batch_size": self.batch_size,
            "cache_size": len(self._embedding_cache),
            "model_loaded": self.model is not None,
            "qdrant_enabled": self.use_qdrant,
            "qdrant_host": self.qdrant_host,
            "qdrant_port": self.qdrant_port,
            "qdrant_collection": self.collection_name,
        }

        # Add Qdrant collection info if available
        if self.use_qdrant:
            qdrant_info = self.get_qdrant_collection_info()
            info.update({"qdrant_info": qdrant_info})

        return info

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")

    def __del__(self):
        """Cleanup when the service is destroyed."""
        try:
            self._save_cache()
        except:
            pass  # Ignore errors during cleanup


# Utility functions
def preprocess_text_for_embedding(text: str) -> str:
    """
    Preprocess text before embedding.

    Args:
        text: Raw text to preprocess

    Returns:
        Cleaned text ready for embedding
    """
    if not text:
        return ""

    # Basic cleaning
    text = text.strip()

    # Remove excessive whitespace
    text = " ".join(text.split())

    # Remove very short texts (less meaningful for embedding)
    if len(text) < 10:
        logger.warning(f"Text too short for meaningful embedding: '{text[:50]}...'")

    return text


def chunk_text(text: str, max_length: int = 500, overlap: int = 50) -> List[str]:
    """
    Split long text into overlapping chunks for better embedding.

    Args:
        text: Text to chunk
        max_length: Maximum length per chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_length

        # Try to break at word boundary
        if end < len(text):
            # Find last space before the limit
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break

    return chunks
