"""Tests for retrieval phase"""

import unittest
from unittest.mock import Mock, patch
from src.phase2_retrieval.retriever import Retriever

class TestRetrieval(unittest.TestCase):
    
    def setUp(self):
        self.mock_vectorstore = Mock()
        self.retriever = Retriever(self.mock_vectorstore, top_k=3)
        
    def test_retrieve_with_empty_query(self):
        self.mock_vectorstore.similarity_search_with_score.return_value = []
        docs = self.retriever.retrieve("")
        self.assertEqual(len(docs), 0)
        
    def test_retrieve_with_context_no_docs(self):
        result = self.retriever.retrieve_with_context("test query", k=0)
        self.assertEqual(result['num_docs'], 0)
        self.assertIn("No relevant documents", result['context'])

if __name__ == '__main__':
    unittest.main()