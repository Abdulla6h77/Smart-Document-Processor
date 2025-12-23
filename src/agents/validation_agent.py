from typing import Dict, Any, Optional, List
import asyncio
import json
from datetime import datetime
from loguru import logger
import re

from src.models.ernie_model import ERNIEModel
from src.agents.base_agent import BaseAgent

class ValidationAgent(BaseAgent):
    """Validation Agent for cross-checking and validating extracted information"""
    
    def __init__(self, name: str, model_config: Dict[str, Any]):
        super().__init__(name, model_config)
        self.ernie_model = ERNIEModel(model_config)
        self.validation_rules = self._load_validation_rules()
    
    async def _execute_task(self, task: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute validation task"""
        extracted_data = task.get("extracted_data")
        validation_type = task.get("validation_type", "comprehensive")
        original_text = task.get("original_text", "")
        
        if not extracted_data:
            raise ValueError("extracted_data is required for validation task")
        
        logger.info(f"Starting validation: {validation_type}")
        
        # Perform different types of validation
        validations = {}
        
        if validation_type in ["comprehensive", "consistency"]:
            validations["consistency"] = await self._validate_consistency(extracted_data, original_text)
        
        if validation_type in ["comprehensive", "format"]:
            validations["format"] = await self._validate_format(extracted_data)
        
        if validation_type in ["comprehensive", "business_rules"]:
            validations["business_rules"] = await self._validate_business_rules(extracted_data)
        
        if validation_type in ["comprehensive", "completeness"]:
            validations["completeness"] = await self._validate_completeness(extracted_data, original_text)
        
        # Cross-reference validation
        if context and context.get("ocr_results"):
            validations["ocr_consistency"] = await self._validate_ocr_consistency(
                extracted_data, context["ocr_results"]
            )
        
        # Generate overall validation score
        overall_score = self._calculate_overall_score(validations)
        
        # Generate validation report
        validation_report = {
            "agent_name": self.name,
            "validation_type": validation_type,
            "overall_score": overall_score,
            "is_valid": overall_score >= 0.7,  # 70% threshold
            "validations": validations,
            "recommendations": self._generate_recommendations(validations),
            "confidence": overall_score,
            "validation_timestamp": datetime.now().isoformat(),
            "validated_fields": list(extracted_data.keys())
        }
        
        logger.info(f"Validation completed: Overall score: {overall_score:.2f}")
        return validation_report
    
    async def _validate_consistency(self, data: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Validate internal consistency of extracted data"""
        inconsistencies = []
        confidence = 1.0
        
        try:
            # Check for common consistency issues
            checks = [
                self._check_amount_consistency(data),
                self._check_date_consistency(data),
                self._check_text_consistency(data, original_text),
                self._check_field_dependencies(data)
            ]
            
            # Run all checks
            results = await asyncio.gather(*checks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    inconsistencies.append(f"Check failed: {str(result)}")
                    confidence -= 0.2
                elif result:
                    inconsistencies.extend(result)
                    confidence -= 0.1 * len(result)
            
            return {
                "status": "passed" if not inconsistencies else "failed",
                "inconsistencies": inconsistencies,
                "confidence": max(0.0, confidence),
                "details": "Consistency validation completed"
            }
            
        except Exception as e:
            logger.error(f"Consistency validation failed: {str(e)}")
            return {
                "status": "error",
                "inconsistencies": [str(e)],
                "confidence": 0.0,
                "details": "Consistency validation error"
            }
    
    async def _validate_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate format of extracted data"""
        format_issues = []
        confidence = 1.0
        
        try:
            # Common format validations
            format_checks = {
                "email": self._validate_email_format,
                "phone": self._validate_phone_format,
                "date": self._validate_date_format,
                "amount": self._validate_amount_format,
                "invoice_number": self._validate_invoice_format,
                "tax_id": self._validate_tax_id_format
            }
            
            for field, value in data.items():
                if isinstance(value, str):
                    # Check field-specific formats
                    for check_name, check_func in format_checks.items():
                        if check_name in field.lower():
                            is_valid, issue = check_func(value)
                            if not is_valid:
                                format_issues.append(f"{field}: {issue}")
                                confidence -= 0.1
            
            return {
                "status": "passed" if not format_issues else "failed",
                "format_issues": format_issues,
                "confidence": max(0.0, confidence),
                "validated_fields": list(data.keys())
            }
            
        except Exception as e:
            logger.error(f"Format validation failed: {str(e)}")
            return {
                "status": "error",
                "format_issues": [str(e)],
                "confidence": 0.0,
                "validated_fields": []
            }
    
    async def _validate_business_rules(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate against business rules"""
        rule_violations = []
        confidence = 1.0
        
        try:
            # Apply business rules
            rules = [
                self._check_amount_ranges(data),
                self._check_date_ranges(data),
                self._check_vendor_validity(data),
                self._check_tax_calculations(data)
            ]
            
            results = await asyncio.gather(*rules, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    rule_violations.append(f"Rule check failed: {str(result)}")
                    confidence -= 0.2
                elif result:
                    rule_violations.extend(result)
                    confidence -= 0.1 * len(result)
            
            return {
                "status": "passed" if not rule_violations else "failed",
                "rule_violations": rule_violations,
                "confidence": max(0.0, confidence),
                "applied_rules": ["amount_ranges", "date_ranges", "vendor_validity", "tax_calculations"]
            }
            
        except Exception as e:
            logger.error(f"Business rules validation failed: {str(e)}")
            return {
                "status": "error",
                "rule_violations": [str(e)],
                "confidence": 0.0,
                "applied_rules": []
            }
    
    async def _validate_completeness(self, data: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Validate completeness of extracted data"""
        missing_fields = []
        confidence = 1.0
        
        try:
            # Define required fields based on document type
            required_fields = self._get_required_fields(data, original_text)
            
            # Check for missing fields
            for field in required_fields:
                if field not in data or not data[field]:
                    missing_fields.append(field)
                    confidence -= 0.15
            
            # Check for data quality
            low_quality_fields = []
            for field, value in data.items():
                if isinstance(value, str) and len(value.strip()) < 2:
                    low_quality_fields.append(field)
                    confidence -= 0.1
            
            return {
                "status": "passed" if not missing_fields else "failed",
                "missing_fields": missing_fields,
                "low_quality_fields": low_quality_fields,
                "confidence": max(0.0, confidence),
                "required_fields": required_fields,
                "completeness_percentage": (len(data) - len(missing_fields)) / len(required_fields) * 100 if required_fields else 100
            }
            
        except Exception as e:
            logger.error(f"Completeness validation failed: {str(e)}")
            return {
                "status": "error",
                "missing_fields": [],
                "low_quality_fields": [],
                "confidence": 0.0,
                "required_fields": [],
                "completeness_percentage": 0
            }
    
    async def _validate_ocr_consistency(self, extracted_data: Dict[str, Any], ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate consistency with original OCR results"""
        inconsistencies = []
        confidence = 1.0
        
        try:
            # Combine OCR text
            ocr_text = " ".join([item.get("text", "") for item in ocr_results])
            
            # Check if extracted data matches OCR content
            for field, value in extracted_data.items():
                if isinstance(value, str) and value.strip():
                    if value.lower() not in ocr_text.lower():
                        inconsistencies.append(f"{field}: '{value}' not found in OCR text")
                        confidence -= 0.1
            
            return {
                "status": "passed" if not inconsistencies else "failed",
                "ocr_inconsistencies": inconsistencies,
                "confidence": max(0.0, confidence),
                "ocr_text_length": len(ocr_text),
                "extracted_fields": len(extracted_data)
            }
            
        except Exception as e:
            logger.error(f"OCR consistency validation failed: {str(e)}")
            return {
                "status": "error",
                "ocr_inconsistencies": [str(e)],
                "confidence": 0.0,
                "ocr_text_length": 0,
                "extracted_fields": 0
            }
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules"""
        return {
            "amount_ranges": {
                "invoice_total_min": 0.01,
                "invoice_total_max": 1000000,
                "tax_rate_min": 0.0,
                "tax_rate_max": 0.5
            },
            "date_ranges": {
                "max_future_days": 365,
                "max_past_days": 3650
            },
            "format_patterns": {
                "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                "phone": r'^\+?1?\d{9,15}$',
                "date": r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',
                "amount": r'^\$?\d+(?:,\d{3})*(?:\.\d{2})?$'
            }
        }
    
    async def _check_amount_consistency(self, data: Dict[str, Any]) -> List[str]:
        """Check amount consistency"""
        issues = []
        
        # Check if total matches sum of items
        if "items" in data and "total_amount" in data:
            items = data["items"]
            if isinstance(items, list):
                calculated_total = 0
                for item in items:
                    if isinstance(item, dict) and "total_price" in item:
                        amount = self._extract_numeric_amount(str(item["total_price"]))
                        calculated_total += amount
                
                stated_total = self._extract_numeric_amount(str(data["total_amount"]))
                if abs(calculated_total - stated_total) > 0.01:
                    issues.append(f"Total amount mismatch: calculated {calculated_total:.2f}, stated {stated_total:.2f}")
        
        # Check tax calculations
        if "subtotal" in data and "tax_amount" in data and "total_amount" in data:
            subtotal = self._extract_numeric_amount(str(data["subtotal"]))
            tax = self._extract_numeric_amount(str(data["tax_amount"]))
            total = self._extract_numeric_amount(str(data["total_amount"]))
            
            if abs(subtotal + tax - total) > 0.01:
                issues.append(f"Tax calculation mismatch: subtotal {subtotal:.2f} + tax {tax:.2f} ≠ total {total:.2f}")
        
        return issues
    
    async def _check_date_consistency(self, data: Dict[str, Any]) -> List[str]:
        """Check date consistency"""
        issues = []
        
        # Check date order
        dates = {}
        for field, value in data.items():
            if "date" in field.lower() and value:
                dates[field] = self._parse_date(str(value))
        
        # Check if invoice date is before due date
        if "invoice_date" in dates and "due_date" in dates:
            if dates["invoice_date"] > dates["due_date"]:
                issues.append("Invoice date is after due date")
        
        # Check for future dates
        from datetime import datetime
        today = datetime.now().date()
        
        for field, date in dates.items():
            if date and date > today:
                days_future = (date - today).days
                if days_future > 30:  # More than 30 days in future
                    issues.append(f"{field} is too far in future: {days_future} days")
        
        return issues
    
    async def _check_text_consistency(self, data: Dict[str, Any], original_text: str) -> List[str]:
        """Check text consistency with original"""
        issues = []
        
        for field, value in data.items():
            if isinstance(value, str) and len(value.strip()) > 0:
                # Check if value appears in original text (case-insensitive)
                if value.lower() not in original_text.lower():
                    issues.append(f"{field}: '{value}' not found in original document")
        
        return issues
    
    async def _check_field_dependencies(self, data: Dict[str, Any]) -> List[str]:
        """Check field dependencies"""
        issues = []
        
        # Example: If there's a total amount, there should be line items
        if "total_amount" in data and "items" not in data:
            issues.append("Total amount present but no line items found")
        
        # If there's tax, there should be subtotal
        if "tax_amount" in data and "subtotal" not in data:
            issues.append("Tax amount present but no subtotal found")
        
        return issues
    
    def _validate_email_format(self, email: str) -> tuple[bool, str]:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(pattern, email):
            return True, ""
        return False, "Invalid email format"
    
    def _validate_phone_format(self, phone: str) -> tuple[bool, str]:
        """Validate phone format"""
        import re
        # Remove common formatting characters
        cleaned = re.sub(r'[\s\-\(\)\+]', '', phone)
        if re.match(r'^\d{10,15}$', cleaned):
            return True, ""
        return False, "Invalid phone format"
    
    def _validate_date_format(self, date: str) -> tuple[bool, str]:
        """Validate date format"""
        import re
        # Check common date formats
        patterns = [
            r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',  # MM/DD/YYYY or DD/MM/YYYY
            r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$',     # YYYY/MM/DD
            r'^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}$'  # DD Mon YYYY
        ]
        
        for pattern in patterns:
            if re.match(pattern, date):
                return True, ""
        
        return False, "Invalid date format"
    
    def _validate_amount_format(self, amount: str) -> tuple[bool, str]:
        """Validate amount format"""
        import re
        pattern = r'^\$?\d+(?:,\d{3})*(?:\.\d{2})?$'
        if re.match(pattern, amount):
            return True, ""
        return False, "Invalid amount format"
    
    def _validate_invoice_format(self, invoice_num: str) -> tuple[bool, str]:
        """Validate invoice number format"""
        import re
        # Common invoice formats
        patterns = [
            r'^[A-Z]{2,3}\d{6,10}$',  # ABC123456
            r'^\d{8,12}$',             # 12345678
            r'^[A-Z]\d{2}[A-Z]\d{4}$'  # A12B3456
        ]
        
        for pattern in patterns:
            if re.match(pattern, invoice_num.upper()):
                return True, ""
        
        return False, "Unusual invoice number format"
    
    def _validate_tax_id_format(self, tax_id: str) -> tuple[bool, str]:
        """Validate tax ID format"""
        import re
        # Basic tax ID validation (simplified)
        if re.match(r'^\d{2}-\d{7}$', tax_id) or re.match(r'^\d{9}$', tax_id):
            return True, ""
        return False, "Invalid tax ID format"
    
    async def _check_amount_ranges(self, data: Dict[str, Any]) -> List[str]:
        """Check amount ranges against business rules"""
        issues = []
        
        for field, value in data.items():
            if "amount" in field.lower() and value:
                try:
                    numeric_value = self._extract_numeric_amount(str(value))
                    
                    # Check against defined ranges
                    if numeric_value < 0:
                        issues.append(f"{field}: negative amount")
                    elif numeric_value > 1000000:  # $1M threshold
                        issues.append(f"{field}: amount exceeds normal range (${numeric_value:,.2f})")
                    
                except ValueError:
                    issues.append(f"{field}: invalid amount format")
        
        return issues
    
    async def _check_date_ranges(self, data: Dict[str, Any]) -> List[str]:
        """Check date ranges"""
        issues = []
        from datetime import datetime, timedelta
        
        today = datetime.now().date()
        
        for field, value in data.items():
            if "date" in field.lower() and value:
                try:
                    parsed_date = self._parse_date(str(value))
                    if parsed_date:
                        # Check for dates too far in past
                        if parsed_date < today - timedelta(days=3650):  # 10 years
                            issues.append(f"{field}: date is more than 10 years in the past")
                        
                        # Check for dates too far in future
                        if parsed_date > today + timedelta(days=365):  # 1 year
                            issues.append(f"{field}: date is more than 1 year in the future")
                
                except ValueError:
                    issues.append(f"{field}: invalid date format")
        
        return issues
    
    async def _check_vendor_validity(self, data: Dict[str, Any]) -> List[str]:
        """Check vendor validity (placeholder)"""
        # In a real implementation, you would check against a vendor database
        issues = []
        
        if "vendor_name" in data and data["vendor_name"]:
            vendor = data["vendor_name"].strip()
            if len(vendor) < 2:
                issues.append("Vendor name too short")
            elif len(vendor) > 100:
                issues.append("Vendor name too long")
        
        return issues
    
    async def _check_tax_calculations(self, data: Dict[str, Any]) -> List[str]:
        """Check tax calculations"""
        issues = []
        
        if "tax_amount" in data and "subtotal" in data:
            try:
                tax = self._extract_numeric_amount(str(data["tax_amount"]))
                subtotal = self._extract_numeric_amount(str(data["subtotal"]))
                
                if subtotal > 0:
                    tax_rate = tax / subtotal
                    
                    # Check if tax rate is reasonable (0-50%)
                    if tax_rate < 0 or tax_rate > 0.5:
                        issues.append(f"Tax rate unreasonable: {tax_rate:.1%}")
            
            except (ValueError, ZeroDivisionError):
                issues.append("Invalid tax calculation")
        
        return issues
    
    def _get_required_fields(self, data: Dict[str, Any], original_text: str) -> List[str]:
        """Get required fields based on document type"""
        # Determine document type from data and text
        doc_type = self._detect_document_type(data, original_text)
        
        required_fields_map = {
            "invoice": ["vendor_name", "invoice_number", "invoice_date", "total_amount"],
            "contract": ["parties", "contract_date", "effective_date"],
            "form": ["form_fields"],
            "report": ["report_type", "executive_summary"],
            "general": ["summary"]
        }
        
        return required_fields_map.get(doc_type, required_fields_map["general"])
    
    def _detect_document_type(self, data: Dict[str, Any], original_text: str) -> str:
        """Detect document type from data and text"""
        # Check for invoice indicators
        invoice_indicators = ["invoice_number", "invoice_date", "total_amount", "vendor_name"]
        if any(indicator in data for indicator in invoice_indicators):
            return "invoice"
        
        # Check for contract indicators
        contract_indicators = ["parties", "contract_date", "effective_date", "termination"]
        if any(indicator in data for indicator in contract_indicators):
            return "contract"
        
        # Check text content
        text_lower = original_text.lower()
        if any(word in text_lower for word in ["invoice", "bill", "receipt"]):
            return "invoice"
        elif any(word in text_lower for word in ["contract", "agreement"]):
            return "contract"
        elif any(word in text_lower for word in ["form", "application"]):
            return "form"
        elif any(word in text_lower for word in ["report", "analysis"]):
            return "report"
        
        return "general"
    
    def _calculate_overall_score(self, validations: Dict[str, Any]) -> float:
        """Calculate overall validation score"""
        scores = []
        
        for validation_type, result in validations.items():
            if isinstance(result, dict) and "confidence" in result:
                scores.append(result["confidence"])
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _generate_recommendations(self, validations: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        for validation_type, result in validations.items():
            if isinstance(result, dict) and result.get("status") == "failed":
                if validation_type == "consistency":
                    recommendations.append("Review data for internal consistency")
                elif validation_type == "format":
                    recommendations.append("Correct format issues in extracted data")
                elif validation_type == "business_rules":
                    recommendations.append("Verify data against business requirements")
                elif validation_type == "completeness":
                    recommendations.append("Complete missing required fields")
                elif validation_type == "ocr_consistency":
                    recommendations.append("Verify extraction against original document")
        
        return recommendations
    
    def _extract_numeric_amount(self, amount_str: str) -> float:
        """Extract numeric value from amount string"""
        try:
            # Remove currency symbols and commas
            cleaned = re.sub(r'[^\d.]', '', amount_str)
            return float(cleaned) if cleaned else 0.0
        except:
            return 0.0
    
    def _parse_date(self, date_str: str):
        """Parse date string to datetime object"""
        from datetime import datetime
        
        formats = [
            "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
            "%m-%d-%Y", "%d-%m-%Y", "%Y-%m-%d",
            "%m/%d/%y", "%d/%m/%y", "%y/%m/%d"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue
        
        return None
    
    async def close(self):
        """Clean up resources"""
        if self.ernie_model:
            await self.ernie_model.close()