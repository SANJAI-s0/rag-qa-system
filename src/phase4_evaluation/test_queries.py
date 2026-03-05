"""
Test Queries Module
Contains sample questions for testing the RAG system
"""

# Sample questions from the project requirements
SAMPLE_QUERIES = [
    {
        "question": "What are the main components of a RAG model, and how do they interact?",
        "expected_paper": "2005.11401v4.pdf",
        "category": "rag"
    },
    {
        "question": "What are the two sub-layers in each encoder layer of the Transformer model?",
        "expected_paper": "1706.03762v7.pdf",
        "category": "transformer"
    },
    {
        "question": "Explain how positional encoding is implemented in Transformers and why it is necessary.",
        "expected_paper": "1706.03762v7.pdf",
        "category": "transformer"
    },
    {
        "question": "Describe the concept of multi-head attention in the Transformer architecture. Why is it beneficial?",
        "expected_paper": "1706.03762v7.pdf",
        "category": "transformer"
    },
    {
        "question": "What is few-shot learning, and how does GPT-3 implement it during inference?",
        "expected_paper": "2005.14165v4.pdf",
        "category": "gpt3"
    },
    # Additional test queries
    {
        "question": "What is the size of GPT-3 in terms of parameters?",
        "expected_paper": "2005.14165v4.pdf",
        "category": "gpt3"
    },
    {
        "question": "How does the retriever in RAG models work?",
        "expected_paper": "2005.11401v4.pdf",
        "category": "rag"
    },
    {
        "question": "What is the role of the decoder in the Transformer architecture?",
        "expected_paper": "1706.03762v7.pdf",
        "category": "transformer"
    }
]

def get_queries_by_category(category: str):
    """Get all queries in a specific category"""
    return [q for q in SAMPLE_QUERIES if q.get("category") == category]