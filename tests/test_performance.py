#!/usr/bin/env python3
"""
Performance benchmarking for RAG QA System
Tests response times and accuracy under load
"""

import sys
import os
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.config import Config
from src.phase2_retrieval import VectorStoreManager, Retriever
from src.phase3_generation import LLMGenerator, AnswerFormatter, PromptTemplates
from src.phase4_evaluation.test_queries import SAMPLE_QUERIES

def benchmark_retrieval():
    """Benchmark retrieval speed"""
    print("\n📊 Benchmarking Retrieval Speed...")
    
    vs_manager = VectorStoreManager(Config.VECTOR_STORE_PATH)
    retriever = Retriever(vs_manager.vectorstore, top_k=Config.TOP_K)
    
    times = []
    for query in SAMPLE_QUERIES[:5]:  # Test with first 5 queries
        start = time.time()
        docs = retriever.retrieve(query["question"])
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  • {query['question'][:30]}...: {elapsed:.3f}s ({len(docs)} docs)")
    
    print(f"\n  ✅ Avg Retrieval Time: {statistics.mean(times):.3f}s")
    print(f"  ✅ Min: {min(times):.3f}s, Max: {max(times):.3f}s")
    
    return times

def benchmark_generation():
    """Benchmark generation speed"""
    print("\n📊 Benchmarking Generation Speed...")
    
    vs_manager = VectorStoreManager(Config.VECTOR_STORE_PATH)
    retriever = Retriever(vs_manager.vectorstore, top_k=Config.TOP_K)
    generator = LLMGenerator(
        model_name=Config.LLM_MODEL,
        temperature=Config.TEMPERATURE,
        max_tokens=Config.MAX_TOKENS,
        api_key=Config.GOOGLE_API_KEY
    )
    formatter = AnswerFormatter()
    
    times = []
    for query in SAMPLE_QUERIES[:3]:  # Test with first 3 queries
        # Retrieve first
        retrieval_result = retriever.retrieve_with_context(query["question"])
        docs = retrieval_result["documents"]
        context = retrieval_result["context"]
        
        prompt = PromptTemplates.get_template(query["question"], context)
        
        start = time.time()
        generation_result = generator.generate_with_metadata(prompt, docs)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  • {query['question'][:30]}...: {elapsed:.3f}s")
    
    print(f"\n  ✅ Avg Generation Time: {statistics.mean(times):.3f}s")
    
    return times

def benchmark_end_to_end():
    """Benchmark complete pipeline"""
    print("\n📊 Benchmarking End-to-End Performance...")
    
    vs_manager = VectorStoreManager(Config.VECTOR_STORE_PATH)
    retriever = Retriever(vs_manager.vectorstore, top_k=Config.TOP_K)
    generator = LLMGenerator(
        model_name=Config.LLM_MODEL,
        temperature=Config.TEMPERATURE,
        max_tokens=Config.MAX_TOKENS,
        api_key=Config.GOOGLE_API_KEY
    )
    formatter = AnswerFormatter()
    
    times = []
    for query in SAMPLE_QUERIES[:3]:
        start = time.time()
        
        # Retrieve
        retrieval_result = retriever.retrieve_with_context(query["question"])
        docs = retrieval_result["documents"]
        context = retrieval_result["context"]
        
        # Generate
        prompt = PromptTemplates.get_template(query["question"], context)
        generation_result = generator.generate_with_metadata(prompt, docs)
        
        # Format
        answer = formatter.format_final_answer(generation_result)
        
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  • {query['question'][:30]}...: {elapsed:.3f}s")
    
    print(f"\n  ✅ Avg End-to-End Time: {statistics.mean(times):.3f}s")
    
    return times

def main():
    print("="*60)
    print("⚡ RAG QA System - Performance Benchmarking")
    print("="*60)
    
    # Run benchmarks
    retrieval_times = benchmark_retrieval()
    generation_times = benchmark_generation()
    e2e_times = benchmark_end_to_end()
    
    # Summary
    print("\n" + "="*60)
    print("📈 BENCHMARK SUMMARY")
    print("="*60)
    print(f"Retrieval (avg):   {statistics.mean(retrieval_times):.3f}s")
    print(f"Generation (avg):  {statistics.mean(generation_times):.3f}s")
    print(f"End-to-End (avg):  {statistics.mean(e2e_times):.3f}s")
    print("="*60)

if __name__ == "__main__":
    main()