"""
Embedding Generator Module
Creates vector embeddings for document chunks
"""

from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import logging
import os

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for document chunks using HuggingFace (free)"""
    
    def __init__(self, model_type: str = "huggingface", model_name: Optional[str] = None):
        """
        Args:
            model_type: Always "huggingface" for free embeddings
            model_name: Specific model name (optional)
        """
        self.model_type = "huggingface"
        model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        
        logger.info(f"Using HuggingFace embeddings with model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name
        )
    
    def generate_embeddings(self, chunks: List[Document]) -> List[List[float]]:
        """Generate embeddings for all chunks"""
        if not chunks:
            logger.warning("No chunks to generate embeddings for")
            return []
            
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def create_vector_store(self, chunks: List[Document], persist_directory: str) -> Optional[FAISS]:
        """Create and persist FAISS vector store"""
        if not chunks:
            logger.error("No chunks to create vector store from")
            return None
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)
            
            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            vectorstore.save_local(persist_directory)
            logger.info(f"Vector store saved to {persist_directory}")
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return None