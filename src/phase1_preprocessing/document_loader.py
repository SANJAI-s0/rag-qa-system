"""
Document Loader Module
Responsible for loading PDF files and extracting text with metadata
"""

import os
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFLoader:
    """Load PDF documents and extract text with metadata"""
    
    def __init__(self, pdf_dir: str):
        """
        Args:
            pdf_dir: Directory containing PDF files
        """
        self.pdf_dir = pdf_dir
        self.documents = []
        
    def load_all_pdfs(self) -> List[Document]:
        """Load all PDFs from the directory"""
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            docs = self.load_single_pdf(pdf_path, pdf_file)
            self.documents.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from {pdf_file}")
            
        return self.documents
    
    def load_single_pdf(self, pdf_path: str, filename: str) -> List[Document]:
        """Load a single PDF file and return list of page documents"""
        documents = []
        
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    metadata = {
                        "source": filename,
                        "page": page_num + 1,
                        "total_pages": len(reader.pages)
                    }
                    doc = Document(page_content=text, metadata=metadata)
                    documents.append(doc)
        except Exception as e:
            logger.error(f"Error loading {pdf_path}: {e}")
            
        return documents