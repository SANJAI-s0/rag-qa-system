"""
Metrics Module
Calculates evaluation metrics for retrieval and generation
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RetrievalMetrics:
    """Metrics for evaluating retrieval quality"""
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Precision@K"""
        if k == 0 or not retrieved_docs:
            return 0.0
        retrieved_at_k = retrieved_docs[:min(k, len(retrieved_docs))]
        relevant_retrieved = [doc for doc in retrieved_at_k if doc in relevant_docs]
        return len(relevant_retrieved) / k
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Recall@K"""
        if not relevant_docs or k == 0:
            return 0.0
        retrieved_at_k = retrieved_docs[:min(k, len(retrieved_docs))]
        relevant_retrieved = [doc for doc in retrieved_at_k if doc in relevant_docs]
        return len(relevant_retrieved) / len(relevant_docs)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def average_precision(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Average Precision"""
        if not relevant_docs:
            return 0.0
            
        score = 0.0
        num_hits = 0
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                num_hits += 1
                score += num_hits / i
                
        return score / len(relevant_docs) if relevant_docs else 0.0

class GenerationMetrics:
    """Metrics for evaluating generated answers"""
    
    def __init__(self):
        try:
            import evaluate
            self.rouge = evaluate.load('rouge')
            self.bleu = evaluate.load('bleu')
            logger.info("Evaluation metrics loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load evaluation metrics: {e}")
            self.rouge = None
            self.bleu = None
        
    def calculate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        if self.rouge is None:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            
        try:
            results = self.rouge.compute(
                predictions=[prediction],
                references=[reference],
                use_aggregator=True
            )
            return results
        except Exception as e:
            logger.error(f"Error calculating ROUGE: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def calculate_bleu(self, prediction: str, reference: str) -> float:
        """Calculate BLEU score"""
        if self.bleu is None:
            return 0.0
            
        try:
            results = self.bleu.compute(
                predictions=[prediction],
                references=[[reference]]
            )
            return results['bleu']
        except Exception as e:
            logger.error(f"Error calculating BLEU: {e}")
            return 0.0
    
    def source_accuracy(self, generated_sources: List[str], 
                        expected_paper: str) -> float:
        """Check if correct paper was cited"""
        if not expected_paper:
            return 0.0
            
        for source in generated_sources:
            if expected_paper.lower() in source.lower():
                return 1.0
        return 0.0
    