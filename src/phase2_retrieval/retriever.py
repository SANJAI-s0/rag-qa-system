"""
Retriever Module
Core retrieval functionality for finding relevant documents
"""

from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class Retriever:
    """Main retriever class for finding relevant document chunks"""
    
    def __init__(self, vectorstore, top_k: int = 5):
        """
        Args:
            vectorstore: FAISS vector store
            top_k: Number of documents to retrieve
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
        
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Retrieve top-k relevant documents for a query
        
        Args:
            query: User's question
            k: Number of documents to retrieve (overrides default)
            
        Returns:
            List of relevant documents with scores
        """
        if self.vectorstore is None:
            logger.error("Vector store not available")
            return []
            
        if k is None:
            k = self.top_k
            
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, k=k
            )
            
            # Add relevance scores to metadata
            for doc, score in docs_with_scores:
                doc.metadata["relevance_score"] = float(score)
                
            return [doc for doc, _ in docs_with_scores]
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def retrieve_with_scores(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """Retrieve documents with their similarity scores"""
        if self.vectorstore is None:
            logger.error("Vector store not available")
            return []
            
        if k is None:
            k = self.top_k
            
        try:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def batch_retrieve(self, queries: List[str], k: Optional[int] = None) -> List[List[Document]]:
        """Retrieve for multiple queries at once"""
        return [self.retrieve(q, k) for q in queries]
    
    def retrieve_with_context(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve and format context for generation
        Returns dictionary with documents and formatted context
        """
        docs = self.retrieve(query, k)
        
        if not docs:
            return {
                "documents": [],
                "context": "No relevant documents found.",
                "num_docs": 0,
                "query": query
            }
        
        # Format context with source attribution
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            context_parts.append(
                f"[Document {i+1}] From '{source}', Page {page}:\n{doc.page_content}"
            )
            
        context = "\n\n".join(context_parts)
        
        return {
            "documents": docs,
            "context": context,
            "num_docs": len(docs),
            "query": query
        }