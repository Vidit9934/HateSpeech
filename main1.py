import logging
from app.services.embedding_service import EmbeddingService
from qdrant_client import models

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

def main():
    setup_logging()
    logger = logging.getLogger("main1")

    # Initialize the embedding service with Qdrant enabled
    embedder = EmbeddingService(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir="./embedding_cache",
        device="auto",
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="test_embeddings",
        use_qdrant=True
    )

    logger.info("EmbeddingService initialized.")

    # Delete and recreate the collection in Qdrant
    embedder.qdrant_client.delete_collection("test_embeddings")
    embedder.qdrant_client.recreate_collection(
        collection_name="test_embeddings",
        vectors_config=models.VectorParams(size=embedder.embedding_dim, distance=models.Distance.COSINE)
    )

    # Test single text embedding and store in Qdrant
    text = "This is a test sentence for embedding and Qdrant storage."
    logger.info(f"Embedding single text: {text}")
    embedding = embedder.embed_text(text, use_cache=True, store_in_qdrant=True, metadata={"source": "main1_test"})
    print("Single text embedding shape:", embedding.shape)

    # Test batch embedding and store in Qdrant
    texts = [
        "I hate china.",
        "americans are dumb.",
        "maa ki shoot"
    ]
    logger.info(f"Embedding batch of {len(texts)} texts and storing in Qdrant.")
    batch_embeddings = embedder.embed_batch(texts, store_in_qdrant=True, metadata_list=[{"source": "main1_test"}]*len(texts))
    print("Batch embedding shapes:", [e.shape for e in batch_embeddings])

    # Test similarity
    sim = embedder.compute_similarity(batch_embeddings[0], batch_embeddings[1])
    print("Similarity between first two batch embeddings:", sim)

    # Test semantic search in Qdrant
    query = "Detecting offensive language"
    logger.info(f"Performing semantic search for query: '{query}'")
    results = embedder.semantic_search(query, top_k=3)
    print("Semantic search results from Qdrant:")
    for res in results:
        print(res)

    # Test model and Qdrant info
    info = embedder.get_model_info()
    print("Model and Qdrant info:", info)

if __name__ == "__main__":
    main()