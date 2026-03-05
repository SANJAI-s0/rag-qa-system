@echo off
echo ===================================================
echo 🧪 RAG QA System - Complete Test Suite
echo ===================================================

echo.
echo [1/6] Running Unit Tests...
pytest tests/ -v

echo.
echo [2/6] Running Interactive Test (5 seconds)...
python test_interactive.py < test_input.txt

echo.
echo [3/6] Running Batch Test...
python test_batch.py

echo.
echo [4/6] Running Performance Benchmark...
python test_performance.py

echo.
echo [5/6] Running Quality Assessment...
python test_quality.py

echo.
echo [6/6] Running Pipeline...
python run_pipeline.py

echo.
echo ===================================================
echo ✅ All tests completed!
echo ===================================================