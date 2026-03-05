#!/bin/bash

echo "==================================================="
echo "📥 Downloading Research Papers for RAG QA System"
echo "==================================================="

# Create directory if it doesn't exist
mkdir -p data/raw

# Download Transformer paper
if [ ! -f "data/raw/1706.03762v7.pdf" ]; then
    echo "Downloading Transformer paper..."
    wget -q --show-progress -O data/raw/1706.03762v7.pdf https://arxiv.org/pdf/1706.03762v7.pdf
    echo "✅ Transformer paper downloaded"
else
    echo "⏩ Transformer paper already exists"
fi

# Download RAG paper
if [ ! -f "data/raw/2005.11401v4.pdf" ]; then
    echo "Downloading RAG paper..."
    wget -q --show-progress -O data/raw/2005.11401v4.pdf https://arxiv.org/pdf/2005.11401v4.pdf
    echo "✅ RAG paper downloaded"
else
    echo "⏩ RAG paper already exists"
fi

# Download GPT-3 paper
if [ ! -f "data/raw/2005.14165v4.pdf" ]; then
    echo "Downloading GPT-3 paper..."
    wget -q --show-progress -O data/raw/2005.14165v4.pdf https://arxiv.org/pdf/2005.14165v4.pdf
    echo "✅ GPT-3 paper downloaded"
else
    echo "⏩ GPT-3 paper already exists"
fi

echo ""
echo "==================================================="
ls -lh data/raw/*.pdf
echo "==================================================="
echo ""
echo "🎉 Download complete! Run: python run_pipeline.py"
