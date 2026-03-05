#!/usr/bin/env python3
"""
Test Gemini API with the correct model
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def test_model(model_name):
    """Test a specific model"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ No API key found")
        return False
    
    print(f"\n🔍 Testing model: {model_name}")
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents="What is a transformer in AI? Answer in one sentence.",
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=100,
            )
        )
        print(f"✅ SUCCESS!")
        print(f"   Response: {response.text[:100]}...")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    # Test the models we know are available
    models_to_test = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
    ]
    
    working_models = []
    for model in models_to_test:
        if test_model(model):
            working_models.append(model)
    
    if working_models:
        print(f"\n✅ Working models: {working_models}")
        print(f"\n📝 Add this to your .env file:")
        print(f"LLM_MODEL={working_models[0]}")
    else:
        print("\n❌ No working models found")
