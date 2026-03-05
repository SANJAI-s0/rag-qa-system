@echo off
echo ===================================================
echo 📥 Downloading Research Papers for RAG QA System
echo ===================================================

:: Create directory if it doesn't exist
if not exist "data\raw" mkdir "data\raw"

:: Download Transformer paper
if not exist "data\raw\1706.03762v7.pdf" (
    echo Downloading Transformer paper...
    powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/1706.03762v7.pdf' -OutFile 'data\raw\1706.03762v7.pdf'"
    echo ✅ Transformer paper downloaded
) else (
    echo ⏩ Transformer paper already exists
)

:: Download RAG paper
if not exist "data\raw\2005.11401v4.pdf" (
    echo Downloading RAG paper...
    powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/2005.11401v4.pdf' -OutFile 'data\raw\2005.11401v4.pdf'"
    echo ✅ RAG paper downloaded
) else (
    echo ⏩ RAG paper already exists
)

:: Download GPT-3 paper
if not exist "data\raw\2005.14165v4.pdf" (
    echo Downloading GPT-3 paper...
    powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/2005.14165v4.pdf' -OutFile 'data\raw\2005.14165v4.pdf'"
    echo ✅ GPT-3 paper downloaded
) else (
    echo ⏩ GPT-3 paper already exists
)

echo.
echo ===================================================
dir "data\raw\*.pdf"
echo ===================================================
echo.
echo 🎉 Download complete! Run: python run_pipeline.py
