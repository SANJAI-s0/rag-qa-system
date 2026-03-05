#!/usr/bin/env python3
"""
Interactive testing script for RAG QA System
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.config import Config
from src.phase2_retrieval import VectorStoreManager, Retriever
from src.phase3_generation import LLMGenerator, AnswerFormatter, PromptTemplates

def setup_system():
    """Initialize the RAG system"""
    print("🔧 Initializing RAG System...")
    
    # Load vector store
    vs_manager = VectorStoreManager(Config.VECTOR_STORE_PATH)
    if vs_manager.vectorstore is None:
        print("❌ Failed to load vector store. Run preprocessing first.")
        return None
    
    # Create retriever
    retriever = Retriever(vs_manager.vectorstore, top_k=Config.TOP_K)
    
    # Initialize generator
    generator = LLMGenerator(
        model_name=Config.LLM_MODEL,
        temperature=Config.TEMPERATURE,
        max_tokens=Config.MAX_TOKENS,
        api_key=Config.GOOGLE_API_KEY
    )
    
    formatter = AnswerFormatter()
    
    return retriever, generator, formatter

def ask_question(retriever, generator, formatter, question):
    """Ask a single question and get answer"""
    print(f"\n📝 Question: {question}")
    print("-" * 60)
    
    # Retrieve documents
    retrieval_result = retriever.retrieve_with_context(question)
    docs = retrieval_result["documents"]
    context = retrieval_result["context"]
    
    print(f"📚 Retrieved {len(docs)} documents")
    
    # Generate prompt
    prompt = PromptTemplates.get_template(question, context)
    
    # Generate answer
    generation_result = generator.generate_with_metadata(prompt, docs)
    answer = formatter.format_final_answer(generation_result)
    
    print(f"\n💡 Answer: {answer}")
    print("-" * 60)
    
    # Show sources
    print("\n📖 Sources:")
    for source in generation_result["sources"]:
        print(f"  • {source['paper']} (page {source['page']})")
    
    return answer

def main():
    """Interactive testing loop"""
    # Setup system
    setup_result = setup_system()
    if setup_result is None:
        return
    
    retriever, generator, formatter = setup_result
    
    print("\n" + "="*60)
    print("🤖 RAG QA System - Interactive Testing Mode")
    print("="*60)
    print("\nCommands:")
    print("  • Type your question and press Enter")
    print("  • Type 'quit' or 'exit' to stop")
    print("  • Type 'list' to see sample questions")
    print("="*60)
    
    # Sample questions for reference
    sample_questions = [
        "What are the main components of a RAG model?",
        "What are the two sub-layers in each encoder layer of the Transformer?",
        "Explain positional encoding in Transformers",
        "What is multi-head attention and why is it beneficial?",
        "What is few-shot learning in GPT-3?",
        "How large is GPT-3?",
        "How does the retriever in RAG work?",
        "What is the role of the decoder in Transformer?"
    ]
    
    while True:
        print("\n" + "-"*60)
        user_input = input("❓ Your question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() == 'list':
            print("\n📋 Sample questions:")
            for i, q in enumerate(sample_questions, 1):
                print(f"  {i}. {q}")
            continue
        
        if not user_input:
            continue
        
        # Ask the question
        try:
            ask_question(retriever, generator, formatter, user_input)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()