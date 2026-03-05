"""Tests for generation phase"""

import unittest
from src.phase3_generation.prompt_templates import PromptTemplates

class TestGeneration(unittest.TestCase):
    
    def test_template_selection(self):
        # Test definition template
        query = "What is attention?"
        context = "Some context"
        template = PromptTemplates.get_template(query, context)
        self.assertIn("DEFINE:", template)
        
        # Test comparative template
        query = "Compare transformers and RNNs"
        template = PromptTemplates.get_template(query, context)
        self.assertIn("COMPARISON QUESTION:", template)
        
        # Test QA template
        query = "How does it work?"
        template = PromptTemplates.get_template(query, context)
        self.assertIn("QUESTION:", template)

if __name__ == '__main__':
    unittest.main()