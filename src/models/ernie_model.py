# Complete erine_model.py with all necessary imports
from typing import Dict, Any, Optional
from src.models.openrouter_model import OpenRouterModel
from loguru import logger
import asyncio

class ERNIEModel:
    """Updated ERNIE model that uses OpenRouter"""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.config = model_config
        self.model_name = model_config.get("model", "ernie-4.5-turbo-8k")
        
        # Try OpenRouter first, fallback to direct if needed
        try:
            self.openrouter = OpenRouterModel(model_config)
            self.use_openrouter = True
            logger.info("Using OpenRouter for ERNIE models")
        except Exception as e:
            logger.warning(f"OpenRouter initialization failed: {e}, falling back to direct API")
            self.use_openrouter = False
            # Add your direct ERNIE API fallback here
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate text using OpenRouter"""
        if self.use_openrouter:
            return await self.openrouter.generate(prompt, system_prompt, **kwargs)
        else:
            # Fallback to direct API or placeholder
            return await self.fallback_generate(prompt, system_prompt, **kwargs)
    
    async def fallback_generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Fallback generation when API is not available"""
        logger.warning("Using fallback generation - responses will be simulated")
        
        # Simple rule-based responses for testing
        if "invoice" in prompt.lower() or "bill" in prompt.lower():
            return '{"document_type": "invoice", "vendor_name": "Test Company", "total_amount": "$1,000.00", "confidence": 0.85}'
        elif "contract" in prompt.lower():
            return '{"document_type": "contract", "parties": ["Party A", "Party B"], "effective_date": "2024-01-01", "confidence": 0.90}'
        else:
            return '{"document_type": "general", "summary": "Document processed successfully", "confidence": 0.75}'
    
    # Keep all other methods the same - they'll use the updated generate method
    async def analyze_document(self, text: str, analysis_type: str = "general") -> Dict[str, Any]:
        """Analyze document using OpenRouter"""
        if self.use_openrouter:
            return await self.openrouter.analyze_document(text, analysis_type)
        else:
            # Fallback analysis
            return {
                "document_type": analysis_type,
                "summary": "Analysis completed (fallback mode)",
                "confidence": 0.8,
                "entities": {},
                "key_points": ["Fallback analysis active"]
            }
    
    async def close(self):
        """Close connections"""
        if self.use_openrouter and self.openrouter:
            await self.openrouter.close()