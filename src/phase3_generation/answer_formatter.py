"""
Answer Formatter Module
Formats and post-processes generated answers
"""

import re
from typing import Dict, Any, List

class AnswerFormatter:
    """Format and enhance generated answers"""
    
    @staticmethod
    def format_final_answer(generation_result: Dict[str, Any]) -> str:
        """Format the final answer with proper citations"""
        
        answer = generation_result["answer"]
        sources = generation_result["sources"]
        
        # Clean up answer (remove extra whitespace, fix formatting)
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Ensure answer ends with proper punctuation
        if answer and answer[-1] not in ['.', '!', '?']:
            answer += '.'
            
        # Add source attribution
        if sources:
            answer += "\n\n**Sources:**"
            for source in sources:
                paper = source['paper'].replace('.pdf', '')
                page = source['page']
                if page != 'N/A':
                    answer += f"\n- {paper}, page {page}"
                else:
                    answer += f"\n- {paper}"
                    
        return answer
    
    @staticmethod
    def extract_citations(text: str) -> List[str]:
        """Extract citations from text for verification"""
        # Look for patterns like (Author, Year) or [1]
        citation_pattern = r'\([A-Za-z\s]+,\s*\d{4}\)|\[\d+\]'
        return re.findall(citation_pattern, text)
    
    @staticmethod
    def add_confidence_score(answer: str, docs: List) -> str:
        """Add confidence score based on retrieval relevance"""
        if not docs:
            return answer
            
        avg_relevance = sum(d.metadata.get('relevance_score', 0) 
                           for d in docs) / len(docs)
        
        # Convert to confidence level
        if avg_relevance > 0.8:
            confidence = "High confidence"
        elif avg_relevance > 0.6:
            confidence = "Medium confidence"
        else:
            confidence = "Low confidence - answer may be incomplete"
            
        return f"{answer}\n\n*Confidence: {confidence}*"