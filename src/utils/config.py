"""Configuration settings for the RAG system"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class"""
    
    # Paths
    RAW_DATA_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    VECTOR_STORE_PATH = "models/faiss_index"
    CHUNKS_OUTPUT_PATH = "data/processed/chunks.pkl"
    OUTPUT_DIR = "outputs/answers"
    LOG_DIR = "outputs/logs"
    
    # Preprocessing parameters
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    
    # Embedding model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "huggingface")
    EMBEDDING_MODEL_NAME = os.getenv("HUGGINGFACE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    
    # LLM parameters
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.0))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 500))
    
    # Retrieval parameters
    TOP_K = int(os.getenv("TOP_K", 5))
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.GOOGLE_API_KEY:
            print("⚠️  Warning: GOOGLE_API_KEY not found. Using mock responses.")
            return False
        return True