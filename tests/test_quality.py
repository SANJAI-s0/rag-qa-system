#!/usr/bin/env python3
"""
Quality assessment tests for RAG QA System
Evaluates answer quality, relevance, and source accuracy
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.config import Config
from src.phase2_retrieval import VectorStoreManager, Retriever
from src.phase3_generation import LLMGenerator, AnswerFormatter, PromptTemplates
from src.phase4_evaluation.test_queries import SAMPLE_QUERIES

def test_source_accuracy():
    """Test if answers correctly cite sources"""
    print("\n📚 Testing Source Attribution Accuracy...")
    
    vs_manager = VectorStoreManager(Config.VECTOR_STORE_PATH)
    retriever = Retriever(vs_manager.vectorstore, top_k=Config.TOP_K)
    generator = LLMGenerator(
        model_name=Config.LLM_MODEL,
        temperature=Config.TEMPERATURE,
        max_tokens=Config.MAX_TOKENS,
        api_key=Config.GOOGLE_API_KEY
    )
    
    correct_sources = 0
    total = 0
    
    for query in SAMPLE_QUERIES:
        if not query.get("expected_paper"):
            continue
            
        retrieval_result = retriever.retrieve_with_context(query["question"])
        docs = retrieval_result["documents"]
        context = retrieval_result["context"]
        
        prompt = PromptTemplates.get_template(query["question"], context)
        generation_result = generator.generate_with_metadata(prompt, docs)
        
        # Check if expected paper is in sources
        expected = query["expected_paper"]
        found = any(expected in s["paper"] for s in generation_result["sources"])
        
        print(f"  • {query['question'][:40]}...: {'✅' if found else '❌'}")
        
        if found:
            correct_sources += 1
        total += 1
    
    accuracy = (correct_sources / total) * 100 if total > 0 else 0
    print(f"\n  ✅ Source Attribution Accuracy: {accuracy:.1f}%")
    
    print(f"Accuracy: {accuracy}%")

def test_answer_relevance():
    """Test if answers are relevant to questions"""
    print("\n🎯 Testing Answer Relevance...")
    
    vs_manager = VectorStoreManager(Config.VECTOR_STORE_PATH)
    retriever = Retriever(vs_manager.vectorstore, top_k=Config.TOP_K)
    generator = LLMGenerator(
        model_name=Config.LLM_MODEL,
        temperature=Config.TEMPERATURE,
        max_tokens=Config.MAX_TOKENS,
        api_key=Config.GOOGLE_API_KEY
    )
    formatter = AnswerFormatter()
    
    # Simple relevance check - answer should contain key terms from question
    relevance_scores = []
    
    for query in SAMPLE_QUERIES:
        question = query["question"]
        
        retrieval_result = retriever.retrieve_with_context(question)
        docs = retrieval_result["documents"]
        context = retrieval_result["context"]
        
        prompt = PromptTemplates.get_template(question, context)
        generation_result = generator.generate_with_metadata(prompt, docs)
        answer = formatter.format_final_answer(generation_result)
        
        # Extract key terms from question
        key_terms = [word.lower() for word in question.split() 
                    if len(word) > 3 and word.lower() not in ['what', 'are', 'the', 'and', 'how', 'does']]
        
        # Check if key terms appear in answer
        answer_lower = answer.lower()
        matches = sum(1 for term in key_terms if term in answer_lower)
        relevance = matches / len(key_terms) if key_terms else 1.0
        
        relevance_scores.append(relevance)
        print(f"  • {question[:40]}...: Relevance {relevance:.2f}")
    
    avg_relevance = sum(relevance_scores) / len(relevance_scores)
    print(f"\n  ✅ Average Relevance Score: {avg_relevance:.2f}")
    
    return avg_relevance

def test_response_consistency():
    """Test if responses are consistent for same question"""
    print("\n🔄 Testing Response Consistency...")
    
    vs_manager = VectorStoreManager(Config.VECTOR_STORE_PATH)
    retriever = Retriever(vs_manager.vectorstore, top_k=Config.TOP_K)
    generator = LLMGenerator(
        model_name=Config.LLM_MODEL,
        temperature=0.0,  # Deterministic
        max_tokens=Config.MAX_TOKENS,
        api_key=Config.GOOGLE_API_KEY
    )
    formatter = AnswerFormatter()
    
    question = "What is multi-head attention?"
    
    answers = []
    for i in range(3):
        retrieval_result = retriever.retrieve_with_context(question)
        docs = retrieval_result["documents"]
        context = retrieval_result["context"]
        
        prompt = PromptTemplates.get_template(question, context)
        generation_result = generator.generate_with_metadata(prompt, docs)
        answer = formatter.format_final_answer(generation_result)
        answers.append(answer)
        print(f"  • Run {i+1}: {len(answer)} chars")
    
    # Check if answers are identical (should be with temperature=0)
    consistent = all(a == answers[0] for a in answers)
    print(f"\n  ✅ Responses Consistent: {'Yes' if consistent else 'No'}")
    
    return consistent

def main():
    print("="*60)
    print("📋 RAG QA System - Quality Assessment")
    print("="*60)
    
    # Run quality tests
    source_acc = test_source_accuracy()
    relevance = test_answer_relevance()
    consistent = test_response_consistency()
    
    print("\n" + "="*60)
    print("📊 QUALITY SUMMARY")
    print("="*60)
    print(f"Source Attribution: {source_acc:.1f}%")
    print(f"Answer Relevance:   {relevance:.2f}")
    print(f"Response Consistency: {'✅ Pass' if consistent else '❌ Fail'}")
    print("="*60)

if __name__ == "__main__":
    main()