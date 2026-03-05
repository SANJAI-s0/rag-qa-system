#!/bin/bash
echo "==================================================="
echo "🧪 RAG QA System - Complete Test Suite"
echo "==================================================="

echo -e "\n[1/6] Running Unit Tests..."
pytest tests/ -v

echo -e "\n[2/6] Running Interactive Test..."
python test_interactive.py < test_input.txt

echo -e "\n[3/6] Running Batch Test..."
python test_batch.py

echo -e "\n[4/6] Running Performance Benchmark..."
python test_performance.py

echo -e "\n[5/6] Running Quality Assessment..."
python test_quality.py

echo -e "\n[6/6] Running Pipeline..."
python run_pipeline.py

echo -e "\n==================================================="
echo "✅ All tests completed!"
echo "==================================================="