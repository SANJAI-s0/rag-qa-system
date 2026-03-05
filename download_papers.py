#!/usr/bin/env python3
"""
Download all required research papers for the RAG QA System
Papers are saved to data/raw/ directory
"""

import os
import requests
from pathlib import Path

# Configuration
PDF_URLS = {
    "1706.03762v7.pdf": "https://arxiv.org/pdf/1706.03762v7.pdf",  # Transformer
    "2005.11401v4.pdf": "https://arxiv.org/pdf/2005.11401v4.pdf",  # RAG
    "2005.14165v4.pdf": "https://arxiv.org/pdf/2005.14165v4.pdf",  # GPT-3
}

def download_papers():
    """Download all papers to data/raw/ directory"""
    
    # Create target directory
    target_dir = Path("data/raw")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("📥 Downloading Research Papers for RAG QA System")
    print("=" * 60)
    
    for filename, url in PDF_URLS.items():
        filepath = target_dir / filename
        
        # Skip if already exists
        if filepath.exists():
            print(f"⏩ {filename} already exists, skipping...")
            continue
        
        print(f"\n📄 Downloading {filename}...")
        print(f"   From: {url}")
        
        try:
            # Download with progress indicator
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Show progress
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r   Progress: {percent:.1f}%", end="")
            
            print(f"\r✅ Successfully downloaded: {filename} ({total_size/1024/1024:.1f} MB)")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error downloading {filename}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    print("\n" + "=" * 60)
    print("📊 Download Summary:")
    print("=" * 60)
    
    # List all files in directory
    files = list(target_dir.glob("*.pdf"))
    if files:
        for f in files:
            size = f.stat().st_size / 1024 / 1024
            print(f"✅ {f.name} - {size:.1f} MB")
    else:
        print("❌ No PDF files found!")
    
    print("\n📍 Location: data/raw/")
    print("=" * 60)

def verify_papers():
    """Verify all required papers are present"""
    print("\n🔍 Verifying papers...")
    
    target_dir = Path("data/raw")
    missing = []
    
    for filename in PDF_URLS.keys():
        filepath = target_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size / 1024 / 1024
            print(f"✅ {filename} - {size:.1f} MB")
        else:
            print(f"❌ {filename} - MISSING")
            missing.append(filename)
    
    return missing

if __name__ == "__main__":
    # Download papers
    download_papers()
    
    # Verify downloads
    missing = verify_papers()
    
    if missing:
        print("\n⚠️  Some papers are missing. You can download them manually from:")
        for filename in missing:
            print(f"   - {PDF_URLS[filename]}")
    else:
        print("\n🎉 All papers downloaded successfully! You can now run:")
        print("   python run_pipeline.py")
