import pytest
import pytest_asyncio
from app.services.vector_service import VectorStoreService
import logging
from qdrant_client.http.models import Distance, VectorParams

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TestVectorStore")

@pytest_asyncio.fixture(scope="function")
async def vector_store():
    """Create a test vector store instance"""
    store = VectorStoreService(
        collection_name="test_embeddings",
        qdrant_host="localhost",
        qdrant_port=6333,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    
    # Clear and recreate collection before each test
    try:
        if store.embedding_service.qdrant_client.get_collection("test_embeddings"):
            store.embedding_service.qdrant_client.delete_collection("test_embeddings")
        
        store.embedding_service.qdrant_client.create_collection(
            collection_name="test_embeddings",
            vectors_config=VectorParams(
                size=store.embedding_service.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Test collection initialized")
    except Exception as e:
        logger.error(f"Collection setup failed: {e}")
        raise
    
    yield store
    
    # Cleanup after test
    try:
        store.embedding_service.qdrant_client.delete_collection("test_embeddings")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

@pytest.mark.asyncio
async def test_vector_store_initialization(vector_store):
    """Test if vector store initializes properly"""
    assert vector_store is not None
    assert vector_store.collection_name == "test_embeddings"
    
    collection_info = vector_store.embedding_service.qdrant_client.get_collection("test_embeddings")
    assert collection_info is not None
    logger.info(f"Collection info: {collection_info}")

@pytest.mark.asyncio
async def test_document_indexing(vector_store):
    """Test document indexing functionality"""
    test_docs = [
        {
            "id": "test_doc_1",
            "content": "I don't hate china.",
            "title": "Test Document 1",
            "document_type": "test_document",
        }
    ]
    
    # Index documents
    result = vector_store.index_documents(test_docs)
    assert result is not None
    logger.info(f"Indexing result: {result}")
    
    # Check points count
    collection_info = vector_store.embedding_service.qdrant_client.get_collection("test_embeddings")
    assert collection_info.points_count > 0
    logger.info(f"Collection info after indexing: {collection_info}")

@pytest.mark.asyncio
async def test_semantic_search(vector_store):
    """Test semantic search functionality"""
    # First index a document
    test_docs = [
        {
            "id": "test_doc_1",
            "content": "I don't hate china.",
            "title": "Test Document 1",
            "document_type": "test_document",
        }
    ]
    vector_store.index_documents(test_docs)
    
    # Perform search using VectorStoreService's search method directly
    query = "Detecting offensive language"
    search_results = vector_store.search(query, top_k=3)
    
    assert search_results is not None
    assert "results" in search_results
    assert len(search_results["results"]) >= 0
    
    # Log results
    for res in search_results["results"]:
        logger.info(f"Search result - ID: {res['id']}, Score: {res['score']:.4f}, Content: {res['content']}")

@pytest.mark.asyncio
async def test_collection_stats(vector_store):
    """Test collection statistics retrieval"""
    collection_info = vector_store.embedding_service.qdrant_client.get_collection("test_embeddings")
    assert collection_info is not None
    assert hasattr(collection_info, "points_count")
    logger.info(f"Collection info: {collection_info}")

@pytest.mark.asyncio
async def test_empty_search(vector_store):
    """Test search behavior with empty collection"""
    query = "Test query"
    search_results = vector_store.search(query, top_k=3)
    
    assert search_results is not None
    assert "results" in search_results
    assert isinstance(search_results["results"], list)
    logger.info(f"Empty collection search results: {search_results}")