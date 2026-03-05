"""Tests for preprocessing phase"""

import unittest
import os
import tempfile
from src.phase1_preprocessing.document_loader import PDFLoader
from src.phase1_preprocessing.text_splitter import DocumentChunker

class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def test_pdf_loader_no_files(self):
        loader = PDFLoader(self.temp_dir)
        docs = loader.load_all_pdfs()
        self.assertEqual(len(docs), 0)
        
    def test_chunker_empty_docs(self):
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents([])
        self.assertEqual(len(chunks), 0)

if __name__ == '__main__':
    unittest.main()