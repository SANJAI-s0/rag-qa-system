@echo off
echo Cleaning up corrupted FAISS index...
if exist models\faiss_index (
    rmdir /s /q models\faiss_index
    echo Removed models\faiss_index
)
if exist data\processed\chunks.pkl (
    del /q data\processed\chunks.pkl
    echo Removed data\processed\chunks.pkl
)
echo Cleanup complete!
