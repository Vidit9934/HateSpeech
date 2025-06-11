# It basically tests the embedding service ,Qdrant integration and semantic search functionality of the Qdrant.
# It initializes the agent, sends a test text for classification, and prints the results.

import logging
from app.services.vector_service import VectorStoreService


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


def main():
    setup_logging()
    logger = logging.getLogger("main1")

    # Initialize the vector store service
    vector_store = VectorStoreService(
        collection_name="test_embeddings",
        qdrant_host="localhost",
        qdrant_port=6333,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    logger.info("VectorStoreService initialized.")

    # OPTION 1: Clear the collection before adding new data
    logger.info("Clearing existing collection data...")
    try:
        # Delete the entire collection and recreate it
        vector_store.embedding_service.qdrant_client.delete_collection(
            "test_embeddings"
        )
        logger.info("Deleted existing collection")

        # Recreate the collection
        from qdrant_client.http.models import Distance, VectorParams

        vector_store.embedding_service.qdrant_client.create_collection(
            collection_name="test_embeddings",
            vectors_config=VectorParams(
                size=vector_store.embedding_service.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Recreated empty collection")

    except Exception as e:
        logger.warning(f"Failed to clear collection: {e}")

    # Prepare a single document to index
    documents = [
        {
            "id": "current_test_doc",
            "content": "I don't hate china.",
            "title": "Current Test Document",
            "document_type": "test_document",
        }
    ]

    # Index document in Qdrant
    logger.info("Indexing document in Qdrant...")
    indexing_result = vector_store.index_documents(documents)
    print("Indexing result:", indexing_result)

    # Perform semantic search
    query = "Detecting offensive language"
    logger.info(f"Performing semantic search for query: '{query}'")
    search_result = vector_store.search(query, top_k=3)
    print("Semantic search results from Qdrant:")
    for res in search_result["results"]:
        print(f"ID: {res['id']}, Score: {res['score']:.4f}, Content: {res['content']}")

    # Print collection stats
    stats = vector_store.get_collection_stats()
    print("Collection stats:", stats)


if __name__ == "__main__":
    main()
