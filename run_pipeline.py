#!/usr/bin/env python3
"""
Main Pipeline Runner for RAG QA System
Executes all phases of the RAG system
"""

import os
import json
import logging
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging - REMOVE EMOJIS for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import project modules
from src.utils.config import Config
from src.phase1_preprocessing import PDFLoader, DocumentChunker, EmbeddingGenerator
from src.phase2_retrieval import VectorStoreManager, Retriever
from src.phase3_generation import LLMGenerator, AnswerFormatter, PromptTemplates
from src.phase4_evaluation import SAMPLE_QUERIES, ResultsAnalyzer, GenerationMetrics

def ensure_directories():
    """Create necessary directories if they don't exist"""
    dirs = [
        Config.RAW_DATA_DIR,
        Config.PROCESSED_DIR,
        Config.VECTOR_STORE_PATH,
        Config.OUTPUT_DIR,
        Config.LOG_DIR,
        "outputs/results"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("Directories created/verified")

def run_phase1_preprocessing(config):
    """Execute Phase 1: Document Preprocessing"""
    logger.info("="*50)
    logger.info("PHASE 1: Document Preprocessing")
    logger.info("="*50)
    
    # Step 1: Load PDFs
    loader = PDFLoader(config.RAW_DATA_DIR)
    documents = loader.load_all_pdfs()
    
    if not documents:
        logger.error("No documents found. Please add PDF files to data/raw/")
        return None, None
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Step 2: Chunk documents
    chunker = DocumentChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = chunker.chunk_documents(documents)
    
    # Step 3: Generate embeddings and create vector store
    embedding_gen = EmbeddingGenerator(
        model_type=config.EMBEDDING_MODEL,
        model_name=config.EMBEDDING_MODEL_NAME
    )
    vectorstore = embedding_gen.create_vector_store(
        chunks,
        config.VECTOR_STORE_PATH
    )
    
    # Step 4: Save chunks
    chunker.save_chunks(chunks, config.CHUNKS_OUTPUT_PATH)
    
    logger.info("Phase 1 completed successfully")
    return vectorstore, chunks

def run_phase2_retrieval(config):
    """Execute Phase 2: Setup Retrieval System"""
    logger.info("="*50)
    logger.info("PHASE 2: Retrieval System Setup")
    logger.info("="*50)
    
    # Load vector store
    vs_manager = VectorStoreManager(config.VECTOR_STORE_PATH)
    
    if vs_manager.vectorstore is None:
        logger.error("Failed to load vector store")
        return None
    
    # Create retriever
    retriever = Retriever(vs_manager.vectorstore, top_k=config.TOP_K)
    
    logger.info("Retrieval system ready")
    return retriever

def run_phase3_generation(query, retriever, config):
    """Execute Phase 3: Answer Generation for a single query"""
    logger.info("-"*50)
    logger.info(f"Processing Query: {query[:50]}...")
    logger.info("-"*50)
    
    # Initialize components
    llm = LLMGenerator(
        model_name=config.LLM_MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        api_key=config.GOOGLE_API_KEY
    )
    formatter = AnswerFormatter()
    
    # Measure retrieval time
    start_time = time.time()
    retrieval_result = retriever.retrieve_with_context(query, k=config.TOP_K)
    retrieval_time = time.time() - start_time
    
    docs = retrieval_result["documents"]
    context = retrieval_result["context"]
    
    logger.info(f"Retrieved {len(docs)} documents in {retrieval_time:.2f}s")
    
    if not docs:
        return {
            "query": query,
            "answer": "No relevant documents found.",
            "sources": [],
            "retrieved_count": 0,
            "retrieval_time": retrieval_time,
            "generation_time": 0
        }
    
    # Generate prompt and answer
    prompt = PromptTemplates.get_template(query, context)
    
    start_time = time.time()
    generation_result = llm.generate_with_metadata(prompt, docs)
    generation_time = time.time() - start_time
    
    # Format final answer
    final_answer = formatter.format_final_answer(generation_result)
    final_answer = formatter.add_confidence_score(final_answer, docs)
    
    logger.info(f"Generated answer in {generation_time:.2f}s")
    
    return {
        "query": query,
        "answer": final_answer,
        "sources": generation_result["sources"],
        "retrieved_count": len(docs),
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "timestamp": datetime.now().isoformat()
    }

def run_phase4_evaluation(results, config):
    """Execute Phase 4: Evaluation"""
    logger.info("="*50)
    logger.info("PHASE 4: Evaluation")
    logger.info("="*50)
    
    analyzer = ResultsAnalyzer()
    metrics = GenerationMetrics()
    
    # Add results to analyzer and calculate metrics
    for result in results:
        # Find matching query to get expected paper
        expected_paper = None
        for q in SAMPLE_QUERIES:
            if q["question"] == result["query"]:
                expected_paper = q.get("expected_paper")
                result["category"] = q.get("category", "unknown")
                break
        
        # Calculate source accuracy
        if expected_paper and result["sources"]:
            source_papers = [s["paper"] for s in result["sources"]]
            source_acc = metrics.source_accuracy(source_papers, expected_paper)
            result["source_accuracy"] = source_acc
        
        analyzer.add_result(result)
    
    # Save and plot results
    analyzer.save_results()
    analyzer.plot_results()
    
    summary = analyzer.generate_summary()
    logger.info(f"Evaluation Summary: {json.dumps(summary, indent=2)}")
    
    return summary

def main():
    """Main execution function"""
    start_time = time.time()
    logger.info("Starting RAG QA System Pipeline")
    
    # Validate configuration
    Config.validate()
    
    # Ensure directories exist
    ensure_directories()
    
    # Check if preprocessing needed
    index_path = os.path.join(Config.VECTOR_STORE_PATH, "index.faiss")
    if not os.path.exists(index_path):
        logger.info("Vector store not found. Running preprocessing...")
        vectorstore, chunks = run_phase1_preprocessing(Config)
        if vectorstore is None:
            logger.error("Preprocessing failed. Exiting.")
            return None, None
    else:
        logger.info("Vector store found. Skipping preprocessing.")
    
    # Setup retrieval
    retriever = run_phase2_retrieval(Config)
    if retriever is None:
        logger.error("Failed to setup retrieval. Exiting.")
        return None, None
    
    # Process each query
    all_results = []
    for i, query_data in enumerate(SAMPLE_QUERIES, 1):
        query = query_data["question"]
        logger.info(f"\nProcessing query {i}/{len(SAMPLE_QUERIES)}")
        
        try:
            result = run_phase3_generation(query, retriever, Config)
            all_results.append(result)
            
            # Save individual answer
            output_file = f"{Config.OUTPUT_DIR}/answer_{i}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Query: {result['query']}\n")
                f.write("="*80 + "\n\n")
                f.write(f"Answer: {result['answer']}\n\n")
                f.write("="*80 + "\n")
                f.write("Sources:\n")
                for source in result['sources']:
                    f.write(f"- {source['paper']} (page {source['page']}, relevance: {source['relevance']:.3f})\n")
                    
            logger.info(f"Answer saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing query {query}: {e}")
            import traceback
            traceback.print_exc()
    
    # Run evaluation
    if all_results:
        summary = run_phase4_evaluation(all_results, Config)
    else:
        logger.error("No results to evaluate")
        summary = {}
    
    # Print final summary
    elapsed = time.time() - start_time
    logger.info("="*50)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed:.2f} seconds")
    logger.info(f"Processed {len(all_results)} queries")
    if all_results:
        success_rate = sum(1 for r in all_results if r.get('sources'))/len(all_results)*100
        logger.info(f"Success rate: {success_rate:.1f}%")
    logger.info("="*50)
    
    return all_results, summary

if __name__ == "__main__":
    results, summary = main()