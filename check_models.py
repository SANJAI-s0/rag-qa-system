#!/usr/bin/env python3
"""
Check available Gemini models with your API key
"""

import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

def check_models():
    """List all available models"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env file")
        return
    
    print(f"🔑 API Key found: {api_key[:8]}...{api_key[-4:]}")
    print("\n📋 Fetching available models...\n")
    
    try:
        client = genai.Client(api_key=api_key)
        models = client.models.list()
        
        print("✅ Available models:\n")
        for model in models:
            print(f"  - {model.name}")
            print(f"    Display name: {model.display_name}")
            print(f"    Description: {model.description[:100]}...")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_models()
