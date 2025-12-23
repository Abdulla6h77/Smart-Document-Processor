from typing import Dict, Any, Optional, List
import asyncio
import time
import json
from datetime import datetime
from loguru import logger
from pathlib import Path

# Change to absolute imports
from src.agents.base_agent import BaseAgent
from src.agents.ocr_agent import OCRAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.validation_agent import ValidationAgent
from src.utils.document_loader import DocumentLoader
from src.utils.logger import log_processing_metrics
class CoordinatorAgent(BaseAgent):
    """Complete Coordinator Agent with advanced workflow management"""
    
    def __init__(self, name: str, model_config: Dict[str, Any]):
        super().__init__(name, model_config)
        
        # Initialize components
        self.agents = {}
        self.document_loader = None
        self.setup_complete = False
        self.workflow_config = {}
        self.parallel_processing = model_config.get("parallel_processing", True)
        self.retry_attempts = model_config.get("retry_attempts", 3)
        self.timeout = model_config.get("timeout", 300)
        
        self.logger.info("Coordinator Agent initialized")
    
    async def setup(self, config: Dict[str, Any]):
        """Setup coordinator with full configuration"""
        try:
            self.logger.info("Setting up Coordinator Agent...")
            
            # Store workflow config
            self.workflow_config = config
            
            # Initialize document loader
            self.document_loader = DocumentLoader(config.get("processing", {}))
            
            # Initialize sub-agents with their configs
            agent_configs = config.get("agents", {})
            
            # OCR Agent
            ocr_config = agent_configs.get("ocr", {})
            self.agents["ocr"] = OCRAgent("OCRAgent", ocr_config)
            self.logger.info("OCR Agent initialized")
            
            # Analysis Agent
            analysis_config = agent_configs.get("analysis", {})
            self.agents["analysis"] = AnalysisAgent("AnalysisAgent", analysis_config)
            self.logger.info("Analysis Agent initialized")
            
            # Validation Agent
            validation_config = agent_configs.get("validation", {})
            self.agents["validation"] = ValidationAgent("ValidationAgent", validation_config)
            self.logger.info("Validation Agent initialized")
            
            self.setup_complete = True
            self.logger.info("Coordinator Agent setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Coordinator setup failed: {str(e)}")
            self.setup_complete = False
            raise
    
    async def _execute_task(self, task: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute coordinated document processing with advanced workflow"""
        if not self.setup_complete:
            raise RuntimeError("Coordinator not properly setup. Call setup() first.")
        
        start_time = datetime.now()
        
        try:
            document_path = task.get("document_path")
            extraction_type = task.get("extraction_type", "text")
            output_format = task.get("output_format", "json")
            analysis_type = task.get("analysis_type", "auto")
            workflow_type = task.get("workflow_type", "standard")
            
            if not document_path:
                raise ValueError("document_path is required")
            
            self.logger.info(f"Starting coordinated processing: {workflow_type} workflow")
            self.logger.info(f"Document: {Path(document_path).name}")
            self.logger.info(f"Extraction type: {extraction_type}")
            
            # Execute workflow based on type
            if workflow_type == "parallel":
                result = await self._execute_parallel_workflow(document_path, extraction_type, analysis_type, task)
            elif workflow_type == "streaming":
                result = await self._execute_streaming_workflow(document_path, extraction_type, analysis_type, task)
            else:
                result = await self._execute_standard_workflow(document_path, extraction_type, analysis_type, task)
            
            # Add comprehensive metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            result["metadata"] = {
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "document_path": document_path,
                "extraction_type": extraction_type,
                "output_format": output_format,
                "workflow_type": workflow_type,
                "agents_involved": ["ocr", "analysis", "validation"],
                "overall_confidence": result.get("overall_confidence", 0.8),
                "setup_complete": self.setup_complete
            }
            
            # Log processing metrics
            doc_type = result.get("analysis_results", {}).get("document_type", "unknown")
            log_processing_metrics(doc_type, processing_time, True)
            
            self.logger.info(f"Coordinated processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            log_processing_metrics("unknown", processing_time, False, str(e))
            
            self.logger.error(f"Coordinated processing failed: {str(e)}")
            raise
    
    async def _execute_standard_workflow(self, document_path: str, extraction_type: str, 
                                       analysis_type: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute standard sequential workflow"""
        self.logger.info("Executing standard workflow")
        
        # Step 1: Load and preprocess document
        load_result = await self._load_document(document_path)
        
        # Step 2: OCR extraction
        ocr_result = await self._perform_ocr(load_result, extraction_type)
        
        # Step 3: Analysis
        analysis_result = await self._perform_analysis(ocr_result, analysis_type, task)
        
        # Step 4: Validation
        validation_result = await self._perform_validation(analysis_result, ocr_result)
        
        # Step 5: Compile final results
        final_result = await self._compile_results(
            load_result, ocr_result, analysis_result, validation_result, task
        )
        
        # Calculate overall confidence
        final_result["overall_confidence"] = self._calculate_overall_confidence([
            ocr_result.get("confidence", 0.8),
            analysis_result.get("confidence_score", 0.8),
            validation_result.get("confidence", 0.8)
        ])
        
        return final_result
    
    async def _execute_parallel_workflow(self, document_path: str, extraction_type: str, 
                                       analysis_type: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parallel workflow for better performance"""
        self.logger.info("Executing parallel workflow")
        
        # Step 1: Load document
        load_result = await self._load_document(document_path)
        
        # Step 2: Run OCR and preprocessing in parallel
        ocr_task = self._perform_ocr(load_result, extraction_type)
        
        # For parallel workflow, we can start analysis while OCR is running
        # This is useful for large documents
        ocr_result = await ocr_task
        
        # Step 3: Analysis and validation (can be parallelized further)
        analysis_result = await self._perform_analysis(ocr_result, analysis_type, task)
        validation_result = await self._perform_validation(analysis_result, ocr_result)
        
        # Compile results
        final_result = await self._compile_results(
            load_result, ocr_result, analysis_result, validation_result, task
        )
        
        final_result["overall_confidence"] = self._calculate_overall_confidence([
            ocr_result.get("confidence", 0.8),
            analysis_result.get("confidence_score", 0.8),
            validation_result.get("confidence", 0.8)
        ])
        
        return final_result
    
    async def _execute_streaming_workflow(self, document_path: str, extraction_type: str, 
                                        analysis_type: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute streaming workflow for large documents"""
        self.logger.info("Executing streaming workflow")
        
        # For streaming, we process page by page and stream results
        # This is a simplified implementation
        load_result = await self._load_document(document_path)
        
        # Process each page separately and combine results
        page_results = []
        
        for i, image in enumerate(load_result["processed_images"]):
            self.logger.info(f"Processing page {i + 1}/{len(load_result['processed_images'])}")
            
            # Create single-page task
            single_page_task = {
                **task,
                "image": image,
                "document_path": f"page_{i+1}"
            }
            
            # Process single page
            page_result = await self._process_single_page(single_page_task, extraction_type, analysis_type)
            page_results.append(page_result)
        
        # Combine page results
        combined_result = await self._combine_page_results(page_results, load_result, task)
        
        return combined_result
    
    async def _process_single_page(self, task: Dict[str, Any], extraction_type: str, analysis_type: str) -> Dict[str, Any]:
        """Process a single page (for streaming workflow)"""
        # Simplified single page processing
        ocr_task = {
            "type": "ocr_extraction",
            "extraction_type": extraction_type,
            "image": task.get("image")
        }
        
        ocr_result = await self.agents["ocr"].process(ocr_task)
        
        analysis_task = {
            "type": "document_analysis",
            "extracted_text": ocr_result["result"]["extracted_content"],
            "analysis_type": analysis_type
        }
        
        analysis_result = await self.agents["analysis"].process(analysis_task)
        
        validation_task = {
            "type": "data_validation",
            "extracted_data": analysis_result["result"].get("detailed_analysis", {}),
            "original_text": ocr_result["result"]["extracted_content"]
        }
        
        validation_result = await self.agents["validation"].process(validation_task)
        
        return {
            "ocr": ocr_result,
            "analysis": analysis_result,
            "validation": validation_result
        }
    
    async def _load_document(self, document_path: str) -> Dict[str, Any]:
        """Load and preprocess document with comprehensive error handling"""
        try:
            self.logger.info(f"Loading document: {Path(document_path).name}")
            
            # Get document info
            doc_info = self.document_loader.get_document_info(document_path)
            
            # Load document images
            images = await self.document_loader.load_document(document_path)
            
            # Preprocess images
            processed_images = await self.document_loader.preprocess_images(images)
            
            return {
                "document_info": doc_info,
                "original_images": images,
                "processed_images": processed_images,
                "load_success": True,
                "image_count": len(images),
                "load_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Document loading failed: {str(e)}")
            raise
    
    async def _perform_ocr(self, load_result: Dict[str, Any], extraction_type: str) -> Dict[str, Any]:
        """Perform OCR extraction with progress tracking"""
        try:
            self.logger.info(f"Starting OCR extraction: {extraction_type}")
            
            ocr_agent = self.agents["ocr"]
            processed_images = load_result["processed_images"]
            
            # Create OCR task
            ocr_task = {
                "type": "ocr_extraction",
                "document_path": "batch_processing",
                "extraction_type": extraction_type,
                "preprocessing": True
            }
            
            # Process all images
            result = await ocr_agent.process(ocr_task)
            
            # Add OCR-specific metrics
            if result["success"]:
                ocr_data = result["result"]
                self.logger.info(f"OCR completed: {ocr_data.get('total_items', 0)} items extracted, "
                               f"confidence: {ocr_data.get('confidence', 0):.2f}")
            
            return result["result"] if result["success"] else {"error": result.get("error", {}).get("message", "OCR failed")}
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {str(e)}")
            raise
    
    async def _perform_analysis(self, ocr_result: Dict[str, Any], analysis_type: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform document analysis with type detection"""
        try:
            self.logger.info(f"Starting document analysis: {analysis_type}")
            
            analysis_agent = self.agents["analysis"]
            
            # Determine analysis type if auto
            if analysis_type == "auto":
                analysis_type = await self._detect_document_type(ocr_result.get("extracted_content", ""))
                self.logger.info(f"Auto-detected document type: {analysis_type}")
            
            # Create analysis task
            analysis_task = {
                "type": "document_analysis",
                "extracted_text": ocr_result.get("extracted_content", ""),
                "analysis_type": analysis_type,
                "document_type": task.get("document_type", "unknown"),
                "context": {
                    "ocr_confidence": ocr_result.get("confidence", 0),
                    "pages": ocr_result.get("pages_processed", 1),
                    "filename": task.get("original_filename", "unknown")
                }
            }
            
            result = await analysis_agent.process(analysis_task)
            
            if result["success"]:
                self.logger.info(f"Analysis completed: {analysis_type}")
            else:
                self.logger.warning(f"Analysis completed with issues: {result.get('error', {}).get('message', '')}")
            
            return result["result"] if result["success"] else {"error": result.get("error", {}).get("message", "Analysis failed")}
            
        except Exception as e:
            self.logger.error(f"Document analysis failed: {str(e)}")
            raise
    
    async def _perform_validation(self, analysis_result: Dict[str, Any], ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform validation with comprehensive checks"""
        try:
            self.logger.info("Starting validation")
            
            validation_agent = self.agents["validation"]
            
            validation_task = {
                "type": "data_validation",
                "extracted_data": analysis_result.get("detailed_analysis", {}),
                "validation_type": "comprehensive",
                "original_text": ocr_result.get("extracted_content", ""),
                "context": {
                    "ocr_results": ocr_result.get("extracted_items", []),
                    "analysis_type": analysis_result.get("analysis_type", "general"),
                    "document_type": analysis_result.get("document_type", "unknown")
                }
            }
            
            result = await validation_agent.process(validation_task)
            
            if result["success"]:
                validation_data = result["result"]
                self.logger.info(f"Validation completed: Score {validation_data.get('overall_score', 0):.2f}, "
                               f"Valid: {validation_data.get('is_valid', False)}")
            else:
                self.logger.warning("Validation completed with issues")
            
            return result["result"] if result["success"] else {"error": result.get("error", {}).get("message", "Validation failed")}
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise
    
    async def _detect_document_type(self, text: str) -> str:
        """Advanced document type detection"""
        try:
            if not text:
                return "general"
            
            text_lower = text.lower()
            
            # Comprehensive keyword matching
            document_types = {
                "invoice": ["invoice", "bill", "receipt", "payment", "due date", "invoice number", "total amount"],
                "contract": ["contract", "agreement", "terms and conditions", "party", "effective date", "termination"],
                "form": ["form", "application", "registration", "field", "signature", "date of birth"],
                "report": ["report", "analysis", "findings", "executive summary", "conclusion", "recommendations"],
                "correspondence": ["dear", "sincerely", "regards", "letter", "memo", "email"],
                "legal": ["legal", "court", "lawsuit", "attorney", "plaintiff", "defendant"],
                "financial": ["financial statement", "balance sheet", "income", "expenses", "profit", "loss"],
                "medical": ["patient", "diagnosis", "treatment", "prescription", "medical", "health"],
                "technical": ["specification", "manual", "technical", "requirements", "system", "architecture"]
            }
            
            # Score each document type
            scores = {}
            for doc_type, keywords in document_types.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                scores[doc_type] = score
            
            # Return the highest scoring type
            best_type = max(scores, key=scores.get)
            best_score = scores[best_type]
            
            # If no clear match, return general
            if best_score == 0:
                return "general"
            
            self.logger.info(f"Document type detection: {best_type} (score: {best_score})")
            return best_type
            
        except Exception as e:
            self.logger.error(f"Document type detection failed: {str(e)}")
            return "general"
    
    async def _compile_results(self, load_result: Dict[str, Any], ocr_result: Dict[str, Any], 
                              analysis_result: Dict[str, Any], validation_result: Dict[str, Any], 
                              task: Dict[str, Any]) -> Dict[str, Any]:
        """Compile comprehensive final results"""
        try:
            self.logger.info("Compiling final results")
            
            # Create detailed result structure
            final_result = {
                "processing_status": "completed",
                "success": True,
                "document_information": {
                    "filename": load_result["document_info"]["filename"],
                    "file_size": load_result["document_info"]["size_mb"],
                    "pages": load_result["image_count"],
                    "format": load_result["document_info"]["extension"],
                    "dimensions": load_result["document_info"].get("dimensions", {}),
                    "loaded_successfully": load_result["load_success"]
                },
                "extraction_results": {
                    "extracted_text": ocr_result.get("extracted_content", ""),
                    "extraction_confidence": ocr_result.get("confidence", 0),
                    "pages_processed": ocr_result.get("pages_processed", 0),
                    "processing_time": ocr_result.get("processing_time", 0),
                    "extraction_type": ocr_result.get("extraction_type", "text"),
                    "items_extracted": ocr_result.get("total_items", 0)
                },
                "analysis_results": {
                    "document_type": analysis_result.get("document_type", "unknown"),
                    "category": analysis_result.get("category", "general"),
                    "summary": analysis_result.get("summary", ""),
                    "entities": analysis_result.get("entities", {}),
                    "detailed_analysis": analysis_result.get("detailed_analysis", {}),
                    "analysis_confidence": analysis_result.get("confidence_score", 0),
                    "analysis_type": analysis_result.get("analysis_type", "general")
                },
                "validation_results": {
                    "is_valid": validation_result.get("is_valid", False),
                    "validation_score": validation_result.get("overall_score", 0),
                    "validation_status": validation_result.get("status", "unknown"),
                    "issues": validation_result.get("validations", {}),
                    "recommendations": validation_result.get("recommendations", []),
                    "missing_fields": validation_result.get("missing_fields", []),
                    "rule_violations": validation_result.get("rule_violations", [])
                },
                "text_statistics": analysis_result.get("text_stats", {}),
                "workflow_metadata": {
                    "agents_used": ["ocr", "analysis", "validation"],
                    "parallel_processing": self.parallel_processing,
                    "retry_attempts": self.retry_attempts,
                    "workflow_type": "standard"
                }
            }
            
            # Add quality metrics
            final_result["quality_metrics"] = self._calculate_quality_metrics(final_result)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Result compilation failed: {str(e)}")
            raise
    
    def _calculate_quality_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics"""
        try:
            ocr_confidence = result["extraction_results"]["extraction_confidence"]
            analysis_confidence = result["analysis_results"]["analysis_confidence"]
            validation_score = result["validation_results"]["validation_score"]
            is_valid = result["validation_results"]["is_valid"]
            
            # Overall quality score (weighted average)
            quality_score = (
                ocr_confidence * 0.3 +           # 30% OCR confidence
                analysis_confidence * 0.3 +      # 30% Analysis confidence  
                validation_score * 0.4           # 40% Validation score
            )
            
            # Determine quality level
            if quality_score >= 0.9:
                quality_level = "excellent"
            elif quality_score >= 0.8:
                quality_level = "good"
            elif quality_score >= 0.7:
                quality_level = "fair"
            else:
                quality_level = "poor"
            
            return {
                "overall_quality_score": quality_score,
                "quality_level": quality_level,
                "recommendations": self._generate_quality_recommendations(quality_score, is_valid),
                "confidence_breakdown": {
                    "ocr": ocr_confidence,
                    "analysis": analysis_confidence,
                    "validation": validation_score
                },
                "reliability_score": quality_score * 0.9 if is_valid else quality_score * 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Quality metrics calculation failed: {str(e)}")
            return {
                "overall_quality_score": 0.0,
                "quality_level": "unknown",
                "error": str(e)
            }
    
    def _generate_quality_recommendations(self, quality_score: float, is_valid: bool) -> List[str]:
        """Generate recommendations based on quality score"""
        recommendations = []
        
        if quality_score < 0.7:
            recommendations.append("Low confidence detected - consider manual review")
        
        if not is_valid:
            recommendations.append("Validation failed - check extracted data for errors")
        
        if quality_score < 0.8:
            recommendations.append("Consider improving document quality or resolution")
        
        if quality_score >= 0.9:
            recommendations.append("High quality extraction - results are reliable")
        
        return recommendations
    
    def _calculate_overall_confidence(self, confidence_scores: List[float]) -> float:
        """Calculate weighted overall confidence"""
        if not confidence_scores:
            return 0.0
        
        # Weighted average with emphasis on validation
        weights = [0.25, 0.35, 0.4]  # OCR, Analysis, Validation
        weighted_score = sum(score * weight for score, weight in zip(confidence_scores, weights))
        
        return min(weighted_score, 1.0)  # Cap at 1.0
    
    async def _combine_page_results(self, page_results: List[Dict[str, Any]], load_result: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from multiple pages (streaming workflow)"""
        # This is a simplified implementation
        # In a full implementation, you'd merge the results properly
        
        combined_text = []
        all_entities = []
        total_confidence = 0
        
        for page_result in page_results:
            if page_result["ocr"]["success"]:
                combined_text.append(page_result["ocr"]["result"]["extracted_content"])
                total_confidence += page_result["ocr"]["result"]["confidence"]
            
            if page_result["analysis"]["success"]:
                analysis_data = page_result["analysis"]["result"]
                if "entities" in analysis_data:
                    all_entities.extend(analysis_data["entities"])
        
        return {
            "extracted_content": " ".join(combined_text),
            "combined_entities": all_entities,
            "average_confidence": total_confidence / len(page_results) if page_results else 0,
            "pages_processed": len(page_results),
            "streaming_workflow": True
        }
    
    def get_all_agents_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all agents"""
        status = {
            "coordinator": self.get_status(),
            "setup_complete": self.setup_complete,
            "workflow_config": {
                "parallel_processing": self.parallel_processing,
                "retry_attempts": self.retry_attempts,
                "timeout": self.timeout
            }
        }
        
        # Add individual agent statuses
        for agent_name, agent in self.agents.items():
            status[agent_name] = agent.get_status()
        
        # Add overall system health
        all_agents_healthy = all(
            agent.get_status()["status"] != "error" for agent in self.agents.values()
        )
        status["system_health"] = {
            "all_agents_healthy": all_agents_healthy,
            "total_agents": len(self.agents),
            "active_agents": sum(1 for agent in self.agents.values() if agent.status == "completed")
        }
        
        return status
    
    async def close(self):
        """Close all agents and cleanup resources"""
        self.logger.info("Closing Coordinator Agent and all sub-agents...")
        
        # Close all sub-agents
        close_tasks = []
        for agent in self.agents.values():
            if hasattr(agent, 'close'):
                close_tasks.append(agent.close())
        
        if close_tasks:
            results = await asyncio.gather(*close_tasks, return_exceptions=True)
            for agent_name, result in zip(self.agents.keys(), results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error closing {agent_name}: {str(result)}")
                else:
                    self.logger.info(f"{agent_name} closed successfully")
        
        self.logger.info("All agents closed successfully")
        self.setup_complete = False
    
    # Additional utility methods for advanced workflows
    async def process_batch(self, document_paths: List[str], extraction_type: str = "text") -> List[Dict[str, Any]]:
        """Process multiple documents in batch"""
        tasks = []
        for doc_path in document_paths:
            task = {
                "type": "document_processing",
                "document_path": doc_path,
                "extraction_type": extraction_type,
                "workflow_type": "standard"
            }
            tasks.append(self.process(task))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "document_path": document_paths[i],
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = {
            "coordinator_stats": self.get_performance_metrics(),
            "agent_stats": {}
        }
        
        # Collect stats from all agents
        for agent_name, agent in self.agents.items():
            stats["agent_stats"][agent_name] = agent.get_performance_metrics()
        
        # Calculate overall statistics
        total_tasks = sum(agent["total_tasks_processed"] for agent in stats["agent_stats"].values())
        total_errors = sum(agent["error_count"] for agent in stats["agent_stats"].values())
        
        stats["overall"] = {
            "total_tasks_processed": total_tasks,
            "total_errors": total_errors,
            "overall_success_rate": (total_tasks - total_errors) / total_tasks if total_tasks > 0 else 0,
            "active_agents": len([agent for agent in self.agents.values() if agent.status != "error"])
        }
        
        return stats