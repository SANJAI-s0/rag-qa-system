"""
Hybrid Search Module
Combines dense and sparse retrieval for better results
"""

from typing import List, Optional
from langchain.schema import Document
from rank_bm25 import BM25Okapi
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Combine dense (FAISS) and sparse (BM25) retrieval"""
    
    def __init__(self, vectorstore, documents: List[Document], 
                 dense_weight: float = 0.5):
        """
        Args:
            vectorstore: FAISS vector store
            documents: List of all documents
            dense_weight: Weight for dense scores (sparse_weight = 1-dense_weight)
        """
        self.vectorstore = vectorstore
        self.documents = documents
        self.dense_weight = dense_weight
        
        # Prepare BM25 index
        try:
            tokenized_corpus = [doc.page_content.split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"Created BM25 index with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error creating BM25 index: {e}")
            self.bm25 = None
        
    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform hybrid search combining dense and sparse results"""
        if self.bm25 is None:
            logger.warning("BM25 not available, falling back to dense retrieval")
            return self.vectorstore.similarity_search(query, k=k)
            
        try:
            # Dense retrieval
            dense_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
            dense_scores = {}
            for doc, score in dense_results:
                # Convert distance to similarity (FAISS returns L2 distance, smaller is better)
                similarity = 1.0 / (1.0 + score)
                dense_scores[doc.page_content] = similarity
            
            # Sparse retrieval (BM25)
            tokenized_query = query.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Normalize and combine scores
            combined_scores = []
            for i, doc in enumerate(self.documents):
                dense_score = dense_scores.get(doc.page_content, 0)
                
                # Normalize BM25 score
                max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1
                bm25_score = bm25_scores[i] / max_bm25 if max_bm25 > 0 else 0
                
                # Weighted combination
                combined = self.dense_weight * dense_score + (1 - self.dense_weight) * bm25_score
                combined_scores.append((doc, combined))
            
            # Sort and return top-k
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in combined_scores[:k]]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return self.vectorstore.similarity_search(query, k=k)