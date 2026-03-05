"""
Vector Store Manager
Handles loading and querying the FAISS index
"""

from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
import os

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manage vector store operations"""
    
    def __init__(self, vector_store_path: str, embedding_model=None):
        """
        Args:
            vector_store_path: Path to saved FAISS index
            embedding_model: Embedding model to use
        """
        self.vector_store_path = vector_store_path
        
        if embedding_model is None:
            # Use HuggingFace embeddings (free)
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        self.embedding_model = embedding_model
        
        # Check if index exists
        index_file = os.path.join(vector_store_path, "index.faiss")
        if not os.path.exists(index_file):
            logger.info(f"Vector store not found at {vector_store_path}. Will create new one.")
            self.vectorstore = None
            return
            
        try:
            self.vectorstore = FAISS.load_local(
                vector_store_path, 
                embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded vector store from {vector_store_path}")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            logger.info("You may need to delete and recreate the index")
            self.vectorstore = None
        
    def get_retriever(self, search_kwargs: dict = None):
        """Get a retriever object"""
        if self.vectorstore is None:
            logger.error("Vector store not loaded")
            return None
            
        if search_kwargs is None:
            search_kwargs = {"k": 5}
            
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )