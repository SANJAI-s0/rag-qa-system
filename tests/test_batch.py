#!/usr/bin/env python3
"""
Batch testing script for RAG QA System
Tests multiple custom questions and saves results
"""

import sys
import os
import json
import time
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.config import Config
from src.phase2_retrieval import VectorStoreManager, Retriever
from src.phase3_generation import LLMGenerator, AnswerFormatter, PromptTemplates

# Custom test questions (add your own here)
CUSTOM_TEST_QUERIES = [
    {
        "question": "What is the Transformer architecture?",
        "category": "transformer",
        "difficulty": "basic"
    },
    {
        "question": "How does attention mechanism work?",
        "category": "transformer",
        "difficulty": "intermediate"
    },
    {
        "question": "What are the benefits of multi-head attention?",
        "category": "transformer",
        "difficulty": "advanced"
    },
    {
        "question": "Explain the training process of GPT-3",
        "category": "gpt3",
        "difficulty": "advanced"
    },
    {
        "question": "What is the difference between RAG-Sequence and RAG-Token?",
        "category": "rag",
        "difficulty": "intermediate"
    },
    {
        "question": "How does positional encoding help Transformers understand word order?",
        "category": "transformer",
        "difficulty": "intermediate"
    },
    {
        "question": "What are the limitations of GPT-3 according to the paper?",
        "category": "gpt3",
        "difficulty": "advanced"
    },
    {
        "question": "Compare the training costs of different models discussed in the papers",
        "category": "comparative",
        "difficulty": "advanced"
    }
]

def test_batch():
    """Run batch tests and save results"""
    
    print("="*60)
    print("📊 RAG QA System - Batch Testing Mode")
    print("="*60)
    
    # Initialize system
    print("\n🔧 Initializing system...")
    vs_manager = VectorStoreManager(Config.VECTOR_STORE_PATH)
    if vs_manager.vectorstore is None:
        print("❌ Failed to load vector store")
        return
    
    retriever = Retriever(vs_manager.vectorstore, top_k=Config.TOP_K)
    generator = LLMGenerator(
        model_name=Config.LLM_MODEL,
        temperature=Config.TEMPERATURE,
        max_tokens=Config.MAX_TOKENS,
        api_key=Config.GOOGLE_API_KEY
    )
    formatter = AnswerFormatter()
    
    results = []
    
    # Run tests
    for i, test in enumerate(CUSTOM_TEST_QUERIES, 1):
        question = test["question"]
        print(f"\n[{i}/{len(CUSTOM_TEST_QUERIES)}] Testing: {question[:50]}...")
        
        try:
            # Time the retrieval
            start_time = time.time()
            retrieval_result = retriever.retrieve_with_context(question)
            retrieval_time = time.time() - start_time
            
            docs = retrieval_result["documents"]
            context = retrieval_result["context"]
            
            # Generate answer
            prompt = PromptTemplates.get_template(question, context)
            
            start_time = time.time()
            generation_result = generator.generate_with_metadata(prompt, docs)
            generation_time = time.time() - start_time
            
            answer = formatter.format_final_answer(generation_result)
            
            # Store result
            result = {
                "question": question,
                "category": test["category"],
                "difficulty": test["difficulty"],
                "answer": answer,
                "sources": generation_result["sources"],
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "num_docs": len(docs),
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
            print(f"  ✅ Success (R:{retrieval_time:.2f}s, G:{generation_time:.2f}s)")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"outputs/results/batch_test_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            "test_config": {
                "total_queries": len(CUSTOM_TEST_QUERIES),
                "model": Config.LLM_MODEL,
                "top_k": Config.TOP_K
            },
            "results": results,
            "summary": {
                "avg_retrieval_time": sum(r["retrieval_time"] for r in results) / len(results),
                "avg_generation_time": sum(r["generation_time"] for r in results) / len(results),
                "success_rate": len(results) / len(CUSTOM_TEST_QUERIES) * 100
            }
        }, f, indent=2)
    
    print(f"\n📁 Results saved to: {filename}")
    print(f"\n📊 Summary:")
    print(f"  • Success Rate: {len(results)}/{len(CUSTOM_TEST_QUERIES)} ({len(results)/len(CUSTOM_TEST_QUERIES)*100:.1f}%)")
    print(f"  • Avg Retrieval: {sum(r['retrieval_time'] for r in results)/len(results):.2f}s")
    print(f"  • Avg Generation: {sum(r['generation_time'] for r in results)/len(results):.2f}s")

if __name__ == "__main__":
    test_batch()