#!/usr/bin/env python3
"""Script to fix all __init__.py files"""

import os

# Define the content for each __init__.py file
init_files = {
    "src/phase1_preprocessing/__init__.py": '''"""Phase 1: Document Preprocessing"""

from .document_loader import PDFLoader
from .text_splitter import DocumentChunker
from .embedding_generator import EmbeddingGenerator

__all__ = ['PDFLoader', 'DocumentChunker', 'EmbeddingGenerator']
''',
    
    "src/phase2_retrieval/__init__.py": '''"""Phase 2: Retrieval System"""

from .vector_store import VectorStoreManager
from .retriever import Retriever
from .hybrid_search import HybridRetriever

__all__ = ['VectorStoreManager', 'Retriever', 'HybridRetriever']
''',
    
    "src/phase3_generation/__init__.py": '''"""Phase 3: Answer Generation"""

from .llm_integration import LLMGenerator
from .prompt_templates import PromptTemplates
from .answer_formatter import AnswerFormatter

__all__ = ['LLMGenerator', 'PromptTemplates', 'AnswerFormatter']
''',
    
    "src/phase4_evaluation/__init__.py": '''"""Phase 4: Evaluation"""

from .test_queries import SAMPLE_QUERIES, get_queries_by_category
from .metrics import RetrievalMetrics, GenerationMetrics
from .results_analyzer import ResultsAnalyzer

__all__ = ['SAMPLE_QUERIES', 'get_queries_by_category', 'RetrievalMetrics', 'GenerationMetrics', 'ResultsAnalyzer']
''',
    
    "src/utils/__init__.py": '''"""Utility modules"""

from .config import Config

__all__ = ['Config']
'''
}

# Write each file
for filepath, content in init_files.items():
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Write the file
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"✅ Fixed: {filepath}")

print("\n🎉 All __init__.py files fixed! Now try running:")
print("python run_pipeline.py")
