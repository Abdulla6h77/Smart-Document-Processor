from typing import Dict, Any, Optional, List
import asyncio
import json
from datetime import datetime
from loguru import logger

from src.models.ernie_model import ERNIEModel
from src.agents.base_agent import BaseAgent

class AnalysisAgent(BaseAgent):
    """Analysis Agent for processing extracted text and generating insights"""
    
    def __init__(self, name: str, model_config: Dict[str, Any]):
        super().__init__(name, model_config)
        self.ernie_model = ERNIEModel(model_config)
        self.analysis_types = {
            "general": "General document analysis",
            "invoice": "Invoice/receipt analysis", 
            "contract": "Contract/legal document analysis",
            "form": "Form/structured document analysis",
            "report": "Report/technical document analysis",
            "correspondence": "Letter/correspondence analysis"
        }
    
    async def _execute_task(self, task: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute analysis task"""
        extracted_text = task.get("extracted_text")
        analysis_type = task.get("analysis_type", "general")
        document_type = task.get("document_type", "unknown")
        
        if not extracted_text:
            raise ValueError("extracted_text is required for analysis task")
        
        logger.info(f"Starting analysis: {analysis_type} for {document_type}")
        
        # Perform analysis based on type
        if analysis_type == "invoice":
            analysis_result = await self._analyze_invoice(extracted_text)
        elif analysis_type == "contract":
            analysis_result = await self._analyze_contract(extracted_text)
        elif analysis_type == "form":
            analysis_result = await self._analyze_form(extracted_text)
        elif analysis_type == "report":
            analysis_result = await self._analyze_report(extracted_text)
        else:
            analysis_result = await self._analyze_general(extracted_text)
        
        # Generate summary
        summary = await self.ernie_model.generate_summary(extracted_text, max_length=300)
        
        # Extract entities
        entities = await self._extract_entities(extracted_text)
        
        # Categorize document
        category = await self._categorize_document(extracted_text)
        
        result = {
            "agent_name": self.name,
            "analysis_type": analysis_type,
            "document_type": document_type,
            "category": category,
            "summary": summary,
            "entities": entities,
            "detailed_analysis": analysis_result,
            "confidence_score": analysis_result.get("confidence", 0.8),
            "processing_timestamp": datetime.now().isoformat(),
            "text_stats": {
                "total_characters": len(extracted_text),
                "total_words": len(extracted_text.split()),
                "total_lines": len(extracted_text.split('\n'))
            }
        }
        
        logger.info(f"Analysis completed: {analysis_type} - Confidence: {result['confidence_score']}")
        return result
    
    async def _analyze_general(self, text: str) -> Dict[str, Any]:
        """General document analysis"""
        prompt = f"""
        Analyze this document text comprehensively and provide structured output in JSON format:
        
        Document Text:
        {text}
        
        Provide analysis in this JSON format:
        {{
            "document_type": "type of document",
            "main_topics": ["topic1", "topic2"],
            "key_entities": {{"organizations": [], "people": [], "dates": [], "amounts": []}},
            "sentiment": "positive/neutral/negative",
            "urgency_level": "high/medium/low",
            "action_items": ["action1", "action2"],
            "summary": "Brief summary of the document",
            "confidence": 0.85
        }}
        """
        
        try:
            response = await self.ernie_model.generate(prompt, temperature=0.4)
            
            # Try to parse JSON response
            try:
                analysis = json.loads(response)
            except json.JSONDecodeError:
                # Fallback to manual parsing
                analysis = self._parse_general_analysis(response)
            
            return analysis
            
        except Exception as e:
            logger.error(f"General analysis failed: {str(e)}")
            return {"error": str(e), "confidence": 0.0}
    
    async def _analyze_invoice(self, text: str) -> Dict[str, Any]:
        """Specialized invoice analysis"""
        prompt = f"""
        Analyze this invoice/receipt and extract all financial information. Return JSON format:
        
        Invoice Text:
        {text}
        
        Extract in this JSON format:
        {{
            "vendor_name": "company name",
            "vendor_address": "company address",
            "invoice_number": "invoice number",
            "invoice_date": "date",
            "due_date": "due date",
            "total_amount": "total amount",
            "tax_amount": "tax amount",
            "items": [
                {{
                    "description": "item description",
                    "quantity": "quantity",
                    "unit_price": "unit price",
                    "total_price": "total price"
                }}
            ],
            "payment_terms": "payment terms",
            "confidence": 0.9
        }}
        """
        
        try:
            response = await self.ernie_model.generate(prompt, temperature=0.3)
            
            try:
                invoice_data = json.loads(response)
            except json.JSONDecodeError:
                invoice_data = self._parse_invoice_data(response)
            
            # Validate amounts
            if "total_amount" in invoice_data:
                invoice_data["total_amount_numeric"] = self._extract_numeric_amount(invoice_data["total_amount"])
            
            return invoice_data
            
        except Exception as e:
            logger.error(f"Invoice analysis failed: {str(e)}")
            return {"error": str(e), "confidence": 0.0}
    
    async def _analyze_contract(self, text: str) -> Dict[str, Any]:
        """Contract analysis"""
        prompt = f"""
        Analyze this contract/document and extract key legal information. Return JSON format:
        
        Contract Text:
        {text}
        
        Extract in this JSON format:
        {{
            "parties": [
                {{"name": "party name", "role": "role", "address": "address"}}
            ],
            "contract_date": "contract date",
            "effective_date": "effective date",
            "expiration_date": "expiration date",
            "key_clauses": [
                {{"type": "clause type", "summary": "clause summary"}}
            ],
            "obligations": [
                {{"party": "party name", "obligation": "obligation description"}}
            ],
            "termination_conditions": ["condition1", "condition2"],
            "governing_law": "governing law",
            "jurisdiction": "jurisdiction",
            "confidence": 0.85
        }}
        """
        
        try:
            response = await self.ernie_model.generate(prompt, temperature=0.3)
            
            try:
                contract_data = json.loads(response)
            except json.JSONDecodeError:
                contract_data = self._parse_contract_data(response)
            
            return contract_data
            
        except Exception as e:
            logger.error(f"Contract analysis failed: {str(e)}")
            return {"error": str(e), "confidence": 0.0}
    
    async def _analyze_form(self, text: str) -> Dict[str, Any]:
        """Form analysis"""
        prompt = f"""
        Analyze this form and extract all field-value pairs. Return JSON format:
        
        Form Text:
        {text}
        
        Extract in this JSON format:
        {{
            "form_fields": [
                {{"field_name": "field name", "field_value": "field value", "confidence": 0.9}}
            ],
            "form_type": "type of form",
            "total_fields": 0,
            "completed_fields": 0,
            "missing_fields": [],
            "confidence": 0.85
        }}
        """
        
        try:
            response = await self.ernie_model.generate(prompt, temperature=0.3)
            
            try:
                form_data = json.loads(response)
            except json.JSONDecodeError:
                form_data = self._parse_form_data(response)
            
            return form_data
            
        except Exception as e:
            logger.error(f"Form analysis failed: {str(e)}")
            return {"error": str(e), "confidence": 0.0}
    
    async def _analyze_report(self, text: str) -> Dict[str, Any]:
        """Report/technical document analysis"""
        prompt = f"""
        Analyze this report/technical document. Return JSON format:
        
        Report Text:
        {text}
        
        Extract in this JSON format:
        {{
            "report_type": "type of report",
            "executive_summary": "brief summary",
            "key_findings": ["finding1", "finding2"],
            "recommendations": ["recommendation1", "recommendation2"],
            "data_points": [
                {{"metric": "metric name", "value": "metric value", "context": "context"}}
            ],
            "charts_tables": ["chart1 description", "chart2 description"],
            "confidence": 0.85
        }}
        """
        
        try:
            response = await self.ernie_model.generate(prompt, temperature=0.4)
            
            try:
                report_data = json.loads(response)
            except json.JSONDecodeError:
                report_data = self._parse_report_data(response)
            
            return report_data
            
        except Exception as e:
            logger.error(f"Report analysis failed: {str(e)}")
            return {"error": str(e), "confidence": 0.0}
    
    async def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        prompt = f"""
        Extract all entities from this text and categorize them:
        
        Text: {text[:2000]}  # Limit text length
        
        Return JSON format:
        {{
            "people": ["person1", "person2"],
            "organizations": ["org1", "org2"],
            "locations": ["location1", "location2"],
            "dates": ["date1", "date2"],
            "amounts": ["amount1", "amount2"],
            "emails": ["email1", "email2"],
            "phone_numbers": ["phone1", "phone2"],
            "urls": ["url1", "url2"]
        }}
        """
        
        try:
            response = await self.ernie_model.generate(prompt, temperature=0.3)
            
            try:
                entities = json.loads(response)
            except json.JSONDecodeError:
                entities = self._parse_entities(response)
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return {}
    
    async def _categorize_document(self, text: str) -> str:
        """Categorize document type"""
        categories = {
            "invoice": ["invoice", "bill", "receipt", "payment"],
            "contract": ["contract", "agreement", "terms", "conditions"],
            "report": ["report", "analysis", "summary", "findings"],
            "form": ["form", "application", "registration", "survey"],
            "correspondence": ["letter", "email", "memo", "notice"],
            "legal": ["legal", "court", "lawsuit", "attorney"],
            "financial": ["financial", "statement", "budget", "accounting"],
            "medical": ["medical", "health", "prescription", "diagnosis"],
            "technical": ["technical", "specification", "manual", "documentation"]
        }
        
        text_lower = text.lower()
        
        # Simple keyword matching
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return "general"
    
    def _parse_general_analysis(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for general analysis"""
        return {
            "document_type": "general",
            "main_topics": [],
            "key_entities": {"organizations": [], "people": [], "dates": [], "amounts": []},
            "sentiment": "neutral",
            "urgency_level": "medium",
            "action_items": [],
            "summary": response[:200],
            "confidence": 0.5
        }
    
    def _parse_invoice_data(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for invoice data"""
        import re
        
        data = {
            "vendor_name": "",
            "vendor_address": "",
            "invoice_number": "",
            "invoice_date": "",
            "total_amount": "",
            "tax_amount": "",
            "items": [],
            "confidence": 0.5
        }
        
        # Simple regex extraction
        amount_pattern = r'\$\s*[\d,]+\.\d{2}'
        amounts = re.findall(amount_pattern, response)
        if amounts:
            data["total_amount"] = amounts[-1]  # Take the last amount as total
        
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        dates = re.findall(date_pattern, response)
        if dates:
            data["invoice_date"] = dates[0]
        
        return data
    
    def _parse_contract_data(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for contract data"""
        return {
            "parties": [],
            "contract_date": "",
            "effective_date": "",
            "key_clauses": [],
            "obligations": [],
            "confidence": 0.5
        }
    
    def _parse_form_data(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for form data"""
        return {
            "form_fields": [],
            "form_type": "unknown",
            "total_fields": 0,
            "completed_fields": 0,
            "confidence": 0.5
        }
    
    def _parse_report_data(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for report data"""
        return {
            "report_type": "unknown",
            "executive_summary": "",
            "key_findings": [],
            "recommendations": [],
            "data_points": [],
            "confidence": 0.5
        }
    
    def _parse_entities(self, response: str) -> Dict[str, List[str]]:
        """Fallback entity parsing"""
        import re
        
        entities = {
            "people": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "amounts": [],
            "emails": [],
            "phone_numbers": [],
            "urls": []
        }
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities["emails"] = re.findall(email_pattern, response)
        
        # Amount extraction
        amount_pattern = r'\$\s*[\d,]+\.\d{2}|\d+(?:,\d{3})*(?:\.\d{2})?'
        entities["amounts"] = re.findall(amount_pattern, response)
        
        # Date extraction
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        entities["dates"] = re.findall(date_pattern, response)
        
        return entities
    
    def _extract_numeric_amount(self, amount_str: str) -> float:
        """Extract numeric value from amount string"""
        import re
        
        try:
            # Remove currency symbols and commas
            numeric_str = re.sub(r'[^\d.]', '', amount_str)
            return float(numeric_str) if numeric_str else 0.0
        except:
            return 0.0
    
    async def close(self):
        """Clean up resources"""
        if self.ernie_model:
            await self.ernie_model.close()