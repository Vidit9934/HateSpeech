# app/agents/hybrid_retriever_agent.py
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List
from app.models.schemas import PolicyDocument

class HybridRetrieverAgent:
    def __init__(self, model_name: str, policy_docs_path: str, vector_db_path: str):
        self.model = SentenceTransformer(model_name)
        self.policy_docs_path = policy_docs_path
        self.vector_db_path = vector_db_path
        self.documents = []
        self.embeddings = None
        self.index = None
        
    def load_and_index_documents(self):
        # Load policy documents
        for filename in os.listdir(self.policy_docs_path):
            if filename.endswith('.txt'):
                with open(os.path.join(self.policy_docs_path, filename), 'r') as f:
                    content = f.read()
                    self.documents.append({'filename': filename, 'content': content})
        
        # Create embeddings
        texts = [doc['content'] for doc in self.documents]
        self.embeddings = self.model.encode(texts)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        self.index.add(self.embeddings.astype('float32'))
    
    async def retrieve_relevant_policies(self, query: str, top_k: int = 3) -> List[PolicyDocument]:
        if self.index is None:
            self.load_and_index_documents()
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            doc = self.documents[idx]
            results.append(PolicyDocument(
                filename=doc['filename'],
                content=doc['content'],
                relevance_score=float(score)
            ))
        
        return results