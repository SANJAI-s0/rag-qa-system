"""
LLM Integration Module
Handles communication with Google's Gemini language model
"""

from typing import Optional, Dict, Any, List
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import time
import os

logger = logging.getLogger(__name__)

class LLMGenerator:
    """Generate answers using Google's Gemini models"""
    
    # Common working model names to try
    COMMON_MODELS = [
        "gemini-2.0-flash-exp",      # Latest experimental
        "gemini-1.5-pro",             # Pro version
        "gemini-1.5-flash",           # Flash version
        "gemini-1.5-flash-001",       # Specific version
        "gemini-1.5-flash-002",       # Another version
        "gemini-1.0-pro",             # Older version
        "gemini-pro",                  # Generic pro
        "gemini-1.5-pro-001",          # Pro specific
        "gemini-1.5-pro-002",          # Pro specific
    ]
    
    def __init__(self, model_name: str = "gemini-1.5-flash", 
                 temperature: float = 0.0,
                 max_tokens: int = 500,
                 api_key: Optional[str] = None):
        """
        Args:
            model_name: Name of the Gemini model to use
            temperature: Sampling temperature (0 = deterministic)
            max_tokens: Maximum tokens in response
            api_key: Google API key
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            logger.warning("Google API key not found. Using mock responses.")
            self.use_mock = True
            self.client = None
            return
        
        try:
            self.client = genai.Client(api_key=self.api_key)
            self.use_mock = False
            
            # Test the connection and find working model
            self.model_name = self._find_working_model(model_name)
            logger.info(f"Using model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {e}")
            self.use_mock = True
            self.client = None
    
    def _find_working_model(self, preferred_model: str) -> str:
        """Find a working model by trying common options"""
        
        # Try the preferred model first
        if self._test_model(preferred_model):
            return preferred_model
        
        logger.warning(f"Preferred model {preferred_model} not working, trying alternatives...")
        
        # Try common models
        for model in self.COMMON_MODELS:
            if model != preferred_model and self._test_model(model):
                logger.info(f"Found working model: {model}")
                return model
        
        # If all else fails, try to get from API
        try:
            models = self.client.models.list()
            for model in models:
                model_name = model.name.replace("models/", "")
                if "gemini" in model_name.lower():
                    logger.info(f"Using model from API: {model_name}")
                    return model_name
        except:
            pass
        
        # Fallback to the original
        logger.warning("No working model found, using preferred model anyway")
        return preferred_model
    
    def _test_model(self, model_name: str) -> bool:
        """Test if a model works by making a simple request"""
        try:
            # Make a minimal test request
            response = self.client.models.generate_content(
                model=model_name,
                contents="test",
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=1,
                )
            )
            return True
        except:
            return False
    
    @retry(stop=stop_after_attempt(3), 
           wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, prompt: str) -> str:
        """
        Generate answer from prompt with retry logic
        
        Args:
            prompt: Formatted prompt for the LLM
            
        Returns:
            Generated answer
        """
        if self.use_mock or not self.client:
            return self._mock_generate(prompt)
            
        try:
            start_time = time.time()
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            
            elapsed = time.time() - start_time
            answer = response.text
            
            logger.info(f"Generated answer in {elapsed:.2f}s ({len(answer)} chars)")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            
            # Try to find another model if this one fails
            if "not found" in str(e).lower() or "404" in str(e):
                logger.info("Model not found, trying to find alternative...")
                new_model = self._find_working_model(self.model_name)
                if new_model != self.model_name:
                    self.model_name = new_model
                    return self.generate(prompt)  # Retry with new model
            
            return f"Error generating answer: {str(e)}"
    
    def _mock_generate(self, prompt: str) -> str:
        """Generate mock response when API is not available"""
        # Extract a simple answer from the context if possible
        if "CONTEXT:" in prompt:
            # Try to find a relevant sentence
            lines = prompt.split('\n')
            for line in lines:
                if 'Document 1]' in line and ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        return f"[MOCK] Based on the documents: {parts[1][:200]}..."
        
        return "[MOCK] This is a mock response. Please check your Gemini API configuration."
    
    def generate_with_metadata(self, prompt: str, context_docs: List) -> Dict[str, Any]:
        """Generate answer with additional metadata"""
        
        answer = self.generate(prompt)
        
        # Extract sources from context documents
        sources = []
        for doc in context_docs:
            source_info = {
                "paper": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'N/A'),
                "relevance": doc.metadata.get('relevance_score', 0)
            }
            # Avoid duplicates
            if not any(s['paper'] == source_info['paper'] and 
                      s['page'] == source_info['page'] for s in sources):
                sources.append(source_info)
        
        return {
            "answer": answer,
            "sources": sources,
            "model": getattr(self, 'model_name', 'unknown'),
            "prompt": prompt
        }