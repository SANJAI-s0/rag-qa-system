"""
Prompt Templates Module
Manages different prompt templates for various query types
"""

from typing import Dict, Any

class PromptTemplates:
    """Collection of prompt templates for different scenarios"""
    
    # Standard QA template with source attribution
    QA_TEMPLATE = """You are an expert AI researcher answering questions based on provided research papers.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. If the answer cannot be found in the context, say "I cannot find this information in the provided papers"
3. Cite specific papers and page numbers when possible
4. Keep answers concise but complete
5. Use technical terminology appropriately

ANSWER:"""
    
    # Template for comparative questions
    COMPARATIVE_TEMPLATE = """You are analyzing similarities and differences between AI research papers.

CONTEXT FROM MULTIPLE PAPERS:
{context}

COMPARISON QUESTION: {question}

INSTRUCTIONS:
1. Compare information from different papers where relevant
2. Clearly indicate which paper each point comes from
3. Highlight both similarities and differences
4. Use bullet points for clarity when appropriate

ANALYSIS:"""
    
    # Template for definition questions
    DEFINITION_TEMPLATE = """You are explaining technical concepts from AI research papers.

CONTEXT:
{context}

DEFINE: {question}

INSTRUCTIONS:
1. Provide a clear, accurate definition
2. Include the paper and author who introduced the concept
3. Add relevant details from the context
4. Keep the explanation accessible yet precise

DEFINITION:"""
    
    @classmethod
    def get_template(cls, query: str, context: str) -> str:
        """Select appropriate template based on query type"""
        
        query_lower = query.lower()
        
        # Check for comparison keywords
        if any(word in query_lower for word in ['compare', 'contrast', 'difference', 'similar']):
            template = cls.COMPARATIVE_TEMPLATE
        # Check for definition keywords
        elif any(word in query_lower for word in ['what is', 'define', 'explain', 'meaning']):
            template = cls.DEFINITION_TEMPLATE
        else:
            template = cls.QA_TEMPLATE
            
        return template.format(context=context, question=query)