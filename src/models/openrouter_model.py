import asyncio
import aiohttp
import json
# Add this line at the top of the file
from typing import Dict, Any, Optional, List
from loguru import logger
import os
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

class OpenRouterModel:
    """OpenRouter API adapter for ERNIE models"""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.config = model_config
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model_name = os.getenv("ERNIE_MODEL_NAME", "baidu/ernie-4.0-turbo-8k")
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        # Session management
        self.session = None
        self.timeout = model_config.get("timeout", 30)
        
        # Model parameters
        self.temperature = model_config.get("temperature", 0.7)
        self.max_tokens = model_config.get("max_tokens", 2000)
        
        logger.info(f"OpenRouter model initialized: {self.model_name}")
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://your-app.com",  # Replace with your app URL
                    "X-Title": "Smart Document Processor"    # Replace with your app name
                }
            )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate text using OpenRouter API"""
        try:
            await self._ensure_session()
            
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Prepare request
            url = f"{self.base_url}/chat/completions"
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "top_p": kwargs.get("top_p", 0.8),
                "stream": False
            }
            
            start_time = datetime.now()
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        processing_time = (datetime.now() - start_time).total_seconds()
                        
                        logger.info(f"OpenRouter generation completed in {processing_time:.2f}s")
                        return content.strip()
                    else:
                        raise Exception(f"No choices in response: {result}")
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"OpenRouter generation failed: {str(e)}")
            raise
    
    async def analyze_document(self, text: str, analysis_type: str = "general") -> Dict[str, Any]:
        """Analyze document text"""
        prompt = self._build_analysis_prompt(text, analysis_type)
        
        try:
            response = await self.generate(prompt, temperature=0.4)
            
            # Parse response into structured format
            analysis = self._parse_analysis_response(response, analysis_type)
            return analysis
            
        except Exception as e:
            logger.error(f"Document analysis failed: {str(e)}")
            return {"error": str(e), "entities": [], "key_points": []}
    
    def _build_analysis_prompt(self, text: str, analysis_type: str) -> str:
        """Build analysis prompt based on type"""
        base_prompts = {
            "general": f"""
                Analyze this document text and provide structured information in JSON format:
                
                Document Text: {text[:3000]}
                
                Return JSON with: document_type, main_topics, key_entities, sentiment, summary, confidence
            """,
            
            "invoice": f"""
                Extract invoice information from this text:
                
                Text: {text[:2000]}
                
                Return JSON with: vendor_name, invoice_number, invoice_date, total_amount, items, tax_amount, confidence
            """,
            
            "contract": f"""
                Extract contract information from this text:
                
                Text: {text[:2000]}
                
                Return JSON with: parties, contract_date, effective_date, key_clauses, obligations, confidence
            """,
            
            "form": f"""
                Extract form fields from this text:
                
                Text: {text[:2000]}
                
                Return JSON with: form_fields, form_type, total_fields, completed_fields, confidence
            """
        }
        
        return base_prompts.get(analysis_type, base_prompts["general"])
    
    def _parse_analysis_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Parse analysis response"""
        try:
            # Try to extract JSON from response
            import json
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                parsed["raw_analysis"] = response
                return parsed
        except:
            pass
        
        # Fallback - return structured response
        return {
            "raw_analysis": response,
            "document_type": analysis_type,
            "summary": response[:200],
            "confidence": 0.8,
            "entities": self._extract_entities_fallback(response),
            "key_points": self._extract_key_points_fallback(response)
        }
    
    def _extract_entities_fallback(self, text: str) -> Dict[str, List[str]]:
        """Fallback entity extraction"""
        import re
        
        entities = {
            "people": [],
            "organizations": [],
            "dates": [],
            "amounts": [],
            "emails": [],
            "phone_numbers": []
        }
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities["emails"] = re.findall(email_pattern, text)
        
        # Amount extraction
        amount_pattern = r'\$\s*[\d,]+\.\d{2}|\d+(?:,\d{3})*(?:\.\d{2})?'
        entities["amounts"] = re.findall(amount_pattern, text)
        
        # Date extraction
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        entities["dates"] = re.findall(date_pattern, text)
        
        return entities
    
    def _extract_key_points_fallback(self, text: str) -> List[str]:
        """Fallback key points extraction"""
        lines = text.split('\n')
        key_points = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 20 and (line.startswith('-') or line.startswith('•') or 'important' in line.lower()):
                key_points.append(line.lstrip('- •').strip())
        
        return key_points[:5]
    
    async def validate_information(self, information: Dict[str, Any], validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate information using OpenRouter"""
        validation_prompt = f"""
        Validate this information against the provided rules and return JSON:
        
        Information: {json.dumps(information, indent=2)}
        Validation Rules: {json.dumps(validation_rules, indent=2)}
        
        Return JSON with: is_valid, confidence, issues[], suggestions[]
        """
        
        try:
            response = await self.generate(validation_prompt, temperature=0.3)
            
            # Try to parse as JSON
            try:
                return json.loads(response)
            except:
                # Fallback validation
                return {
                    "is_valid": "valid" in response.lower() or "correct" in response.lower(),
                    "confidence": 0.7,
                    "issues": [],
                    "suggestions": ["Manual review recommended"]
                }
                
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {
                "is_valid": False,
                "confidence": 0.0,
                "issues": [f"Validation error: {str(e)}"],
                "suggestions": ["Manual validation required"]
            }
    
    async def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate summary using OpenRouter"""
        prompt = f"Summarize this text in no more than {max_length} characters:\n\n{text}"
        
        try:
            summary = await self.generate(prompt, max_tokens=max_length)
            return summary.strip()[:max_length]
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    async def close(self):
        """Close session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "provider": "OpenRouter",
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "status": "active" if self.api_key else "inactive"
        }