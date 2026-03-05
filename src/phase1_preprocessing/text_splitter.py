"""
Text Splitter Module
Handles intelligent chunking of documents with overlap
"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Split documents into overlapping chunks for better retrieval"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        chunked_docs = self.splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, doc in enumerate(chunked_docs):
            doc.metadata["chunk_id"] = i
            doc.metadata["chunk_size"] = len(doc.page_content)
            
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs
    
    def save_chunks(self, chunks: List[Document], output_path: str):
        """Save chunks to disk for later use"""
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(chunks, f)
        logger.info(f"Saved chunks to {output_path}")