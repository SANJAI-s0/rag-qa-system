<div align="center">

# RAG QA System for AI Research Papers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-supported-blue)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://makeapullrequest.com)
[![GitHub stars](https://img.shields.io/github/stars/SANJAI-s0/rag-qa-system)](https://github.com/SANJAI-s0/rag-qa-system/stargazers)

**Ask questions about Transformer, RAG & GPT-3 papers - get answers with sources!**

</div>

A powerful Retrieval-Augmented Generation (RAG) system that answers questions based on seminal AI research papers including the Transformer, RAG, and GPT-3 papers. The system combines dense retrieval with Google's Gemini LLM to provide accurate, context-aware answers with source attribution.

---

## 🌟 Features

- **Document Processing**: Automatically loads PDFs, splits into intelligent chunks, generates embeddings
- **Intelligent Retrieval**: FAISS-based dense retrieval with hybrid search capabilities
- **Answer Generation**: Google Gemini 2.5 Flash integration for high-quality answers
- **Source Attribution**: Every answer includes paper name and page number references
- **Configurable**: All parameters adjustable via `.env` file
- **Extensible**: Easy to add new papers or modify queries
- **Docker Support**: Ready-to-run containerized version

---

## 📚 Supported Papers

The system is pre-configured for these seminal papers:
1. **"Attention Is All You Need"** (Transformer, 2017) - `1706.03762v7.pdf`
2. **"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"** (RAG, 2020) - `2005.11401v4.pdf`
3. **"Language Models are Few-Shot Learners"** (GPT-3, 2020) - `2005.14165v4.pdf`

---

## 🏗️ Project Structure

```
rag-qa-system/               # Root project folder
│
├── data/                    # All data files
│   ├── processed/           # Processed chunks (saved after preprocessing)
│   └── raw/                 # Original PDF research papers
│       ├── 1706.03762v7.pdf     # "Attention Is All You Need" (Transformer)
│       ├── 2005.11401v4.pdf     # RAG paper
│       └── 2005.14165v4.pdf     # GPT-3 paper
│
├── models/                  # Saved models and vector indices
│   └── faiss_index/
│       ├── index.faiss      # FAISS vector index
│       └── index.pkl        # Metadata associated with the index
│
├── notebooks/               # Jupyter notebooks for exploration & testing
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_retrieval_testing.ipynb
│   └── 03_results_visualization.ipynb
│
├── outputs/                 # Generated outputs
│   ├── answers/             # Final answers for each query
│   │   ├── answer_1.txt
│   │   ├── answer_2.txt
│   │   ├── answer_3.txt
│   │   ├── answer_4.txt
│   │   ├── answer_5.txt
│   │   ├── answer_6.txt
│   │   ├── answer_7.txt
│   │   └── answer_8.txt
│   │
│   ├── logs/                # Execution logs
│   │   └── pipeline.log
│   │
│   └── results/             # Evaluation results (JSON, plots)
│
├── src/                     # Source code (organized by pipeline phase)
│   │
│   ├── phase1_preprocessing/        # Phase 1: Document preprocessing
│   │   ├── __init__.py
│   │   ├── document_loader.py       # Load PDFs and extract text
│   │   ├── embedding_generator.py   # Generate embeddings
│   │   └── text_splitter.py         # Split documents into chunks
│   │
│   ├── phase2_retrieval/            # Phase 2: Retrieval system
│   │   ├── __init__.py
│   │   ├── hybrid_search.py         # Optional BM25 + dense hybrid search
│   │   ├── retriever.py             # Core retrieval logic
│   │   └── vector_store.py          # FAISS vector store management
│   │
│   ├── phase3_generation/           # Phase 3: Answer generation
│   │   ├── __init__.py
│   │   ├── answer_formatter.py      # Format answers with citations
│   │   ├── llm_integration.py       # Google Gemini API integration
│   │   └── prompt_templates.py      # Prompt templates
│   │
│   ├── phase4_evaluation/           # Phase 4: Evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py               # Retrieval & generation metrics
│   │   ├── results_analyzer.py      # Analyze and visualize results
│   │   └── test_queries.py          # Sample evaluation queries
│   │
│   └── utils/                       # Utility modules
│       ├── __init__.py
│       ├── config.py                # Configuration management
│       ├── file_utils.py            # File I/O helpers
│       └── visualization.py         # Plotting utilities
│
├── tests/                   # Unit tests
│   ├── test_generation.py
│   ├── test_preprocessing.py
│   └── test_retrieval.py
│
├── .dockerignore            # Files ignored during Docker build
├── .env                     # Local environment variables (not committed)
├── .env.example             # Example environment variables
├── .gitignore               # Git ignore rules
│
├── check_models.py          # Script to list available Gemini models
├── cleanup.bat              # Windows script to clean FAISS index
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile               # Docker image definition
├── fix_init_files.py        # Helper to fix missing __init__ exports
│
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
│
├── run_pipeline.py          # Main entry point for the RAG pipeline
└── test_gemini_working.py   # Script to verify Gemini API connection
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google Gemini API key (get one from [Google AI Studio](https://aistudio.google.com/app/apikey))
- 8GB+ RAM recommended

### Installation

#### 1. **Clone the repository**
```bash
git clone https://github.com/SANJAI-s0/rag-qa-system.git
cd rag-qa-system
```

#### 2. **Create virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

#### 3. **Install dependencies**
```bash
pip install -r requirements.txt
```

#### 4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your Gemini API key
```

#### 5. Add PDF files

Place your research papers in `data/raw/`. Use the following links to download them, or skip this step and manually add the files into the `data/raw/` folder.

| Paper | Title of the File | Filename | PDF Download Link |
|------|-------------------|----------|------------------|
| Transformer | *Attention Is All You Need* | `1706.03762v7.pdf` | [Download](https://arxiv.org/pdf/1706.03762v7.pdf) |
| RAG | *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | `2005.11401v4.pdf` | [Download](https://arxiv.org/pdf/2005.11401v4.pdf) |
| GPT-3 | *Language Models are Few-Shot Learners* | `2005.14165v4.pdf` | [Download](https://arxiv.org/pdf/2005.14165v4.pdf) |

**Download Research Papers**

Run the download script to automatically get all required papers:

### Option 1: Python Script (All Platforms)
```bash
# Install requests if you haven't
pip install requests

# Run download script
python download_papers.py
```

### Option 2: Windows Batch Script
```bash
# Run download script
download_papers.bat
```
**`or`**
```bash
# Run download script
.\download_papers.bat
```

### Option 3: Linux/Mac Shell Script
```bash
# Run download script
chmod +x download_papers.sh
./download_papers.sh
```

6. **Run the pipeline**
```bash
python run_pipeline.py
```

---

## 🐳 Docker Support

Using Docker Compose (these are other way to run the project using docker container)

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

**Using Docker Directly**
```bash
# Build image
docker build -t rag-qa-system .

# Run container
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/outputs:/app/outputs \
           -v $(pwd)/.env:/app/.env \
           rag-qa-system
```

---

## ⚙️ Configuration

All settings are managed through `.env` file:

```bash
# Google Gemini API Configuration
# Get your API key from: https://aistudio.google.com/app/apikey

GOOGLE_API_KEY=your-actual-gemini-api-key-here

# Use one of these working models from your list:
LLM_MODEL=gemini-2.5-flash  # Latest stable flash model
# Alternative options (uncomment one):
# LLM_MODEL=gemini-2.5-pro      # More powerful but slower
# LLM_MODEL=gemini-2.0-flash    # Previous version
# LLM_MODEL=gemini-2.0-flash-001  # Specific version

# Embedding model (huggingface is free)
EMBEDDING_MODEL=huggingface

# LLM parameters
TEMPERATURE=0.0
MAX_TOKENS=500

# Retrieval configuration
TOP_K=5
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

---

## 📊 Sample Queries

The system comes pre-configured with these test queries:

1. RAG Model: "What are the main components of a RAG model, and how do they interact?"

2. Transformer Architecture: "What are the two sub-layers in each encoder layer of the Transformer model?"

3. Positional Encoding: "Explain how positional encoding is implemented in Transformers and why it is necessary."

4. Multi-Head Attention: "Describe the concept of multi-head attention in the Transformer architecture. Why is it beneficial?"

5. Few-Shot Learning: "What is few-shot learning, and how does GPT-3 implement it during inference?"

6. GPT-3 Size: "What is the size of GPT-3 in terms of parameters?"

7. RAG Retriever: "How does the retriever in RAG models work?"

8. Transformer Decoder: "What is the role of the decoder in the Transformer architecture?"

---

## 📈 Output Examples

> Answer File (`outputs/answers/answer_1.txt`)

```text
Query: What are the main components of a RAG model, and how do they interact?
================================================================================

Answer: A RAG model consists of two main components: 1. A mechanism that provides distributions.

**Sources:**
- 2005.11401v4, page 10
- 2005.11401v4, page 17
- 2005.11401v4, page 6
- 2005.11401v4, page 2

*Confidence: High confidence*

================================================================================
Sources:
- 2005.11401v4.pdf (page 10, relevance: 0.578)
- 2005.11401v4.pdf (page 17, relevance: 0.628)
- 2005.11401v4.pdf (page 6, relevance: 0.967)
- 2005.11401v4.pdf (page 2, relevance: 1.074)
```

---

## Evaluation Results

After running, check `outputs/results/` for:

- `results_YYYYMMDD_HHMMSS.json` - Detailed metrics
- `evaluation_plots.png` - Visualization of:
    - Source accuracy by category
    - Average response times
    - Source accuracy over time

---

## 🧪 Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_preprocessing.py
pytest tests/test_retrieval.py
pytest tests/test_generation.py
```

---

## 📝 Adding Custom Questions

Edit `src/phase4_evaluation/test_queries.py`:

```python
SAMPLE_QUERIES = [
    {
        "question": "Your custom question here",
        "expected_paper": "relevant_paper.pdf",  # Optional
        "category": "custom"
    },
    # Add more...
]
```

---

## 📊 Performance Metrics
```
| Component                | Average Time        |
|--------------------------|---------------------|
| Document Loading         | 2 – 3 seconds       |
| Chunking                 | 1 – 2 seconds       |
| Embedding Generation     | 30 – 60 seconds     |
| Retrieval per Query      | 0.05 – 0.23 seconds |
| Generation per Query     | 2 – 4 seconds       |
| Total Pipeline (8 queries) | 45 – 60 seconds   |
```

---

## 📄 License
MIT License - see LICENSE file for details

---

## 🤝 Acknowledgments

- Google Gemini API for LLM capabilities
- HuggingFace for free embeddings
- LangChain for excellent tools
- FAISS for fast similarity search

---

