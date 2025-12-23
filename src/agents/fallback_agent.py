"""
Fallback agent implementation when CAMEL-AI is not available
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger
import asyncio
import json
import uuid

class FallbackAgent:
    """Simple fallback agent that doesn't depend on CAMEL-AI"""
    
    def __init__(self, name: str, system_message: str = "", **kwargs):
        self.name = name
        self.system_message = system_message
        self.agent_id = str(uuid.uuid4())
        self.memory = []
        self.status = "idle"
        self.last_activity = None
        self.total_tasks_processed = 0
        self.error_count = 0
        self.config = kwargs
        
        self.logger = logger.bind(agent_name=name, agent_id=self.agent_id)
        self.logger.info(f"Fallback agent {name} initialized")
    
    async def process(self, task: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a task using fallback logic"""
        task_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            self.status = "processing"
            self.last_activity = start_time
            
            self.logger.info(f"Processing task {task_id}")
            
            # Simulate processing based on task type
            result = await self._simulate_processing(task, context)
            
            self.status = "completed"
            self.total_tasks_processed += 1
            
            return {
                "success": True,
                "task_id": task_id,
                "agent_name": self.name,
                "result": result,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "status": "completed"
            }
            
        except Exception as e:
            self.status = "error"
            self.error_count += 1
            
            return {
                "success": False,
                "task_id": task_id,
                "agent_name": self.name,
                "error": str(e),
                "status": "error"
            }
    
    async def _simulate_processing(self, task: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Simulate agent processing"""
        task_type = task.get("type", "unknown")
        
        # Simulate different processing based on task type
        if task_type == "document_analysis":
            return {
                "document_type": "invoice",
                "summary": "Document processed successfully",
                "confidence": 0.85,
                "entities": {
                    "amounts": ["$1,000.00"],
                    "dates": ["2024-01-15"],
                    "organizations": ["Test Company"]
                },
                "simulated": True
            }
        elif task_type == "ocr_extraction":
            return {
                "extracted_content": "Sample extracted text",
                "confidence": 0.90,
                "simulated": True
            }
        else:
            return {
                "result": f"Processed {task_type} task",
                "confidence": 0.80,
                "simulated": True
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_name": self.name,
            "status": self.status,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "metrics": {
                "total_tasks_processed": self.total_tasks_processed,
                "error_count": self.error_count,
                "success_rate": (self.total_tasks_processed - self.error_count) / max(self.total_tasks_processed, 1)
            }
        }