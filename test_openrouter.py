#!/usr/bin/env python3
"""Test OpenRouter integration"""

import asyncio
import os
from pathlib import Path
import sys
from dotenv import load_dotenv  # Add this import

sys.path.insert(0, str(Path(__file__).parent / "src"))

load_dotenv()

from models.openrouter_model import OpenRouterModel

async def test_openrouter():
    """Test OpenRouter model"""
    
    # Make sure environment is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set")
        print("Add to .env file: OPENROUTER_API_KEY=your-api-key")
        return
    
    config = {
        "temperature": 0.7,
        "max_tokens": 500,
        "timeout": 30
    }
    
    try:
        model = OpenRouterModel(config)
        
        # Test simple generation
        print("🧪 Testing OpenRouter model...")
        
        response = await model.generate("What is 2+2?")
        print(f"✅ Simple query response: {response}")
        
        # Test document analysis
        test_text = "Invoice #123 from Tech Solutions for $1,000.00 dated 2024-01-15"
        analysis = await model.analyze_document(test_text, "invoice")
        print(f"✅ Document analysis: {analysis}")
        
        await model.close()
        print("✅ OpenRouter integration successful!")
        
    except Exception as e:
        print(f"❌ OpenRouter test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_openrouter())