# src/agents/base_agent.py - Complete file with CAMEL-AI fix
from typing import Dict, Any, Optional, List, Union
import asyncio
import json
import uuid
from datetime import datetime
from loguru import logger
from abc import ABC, abstractmethod

# Try CAMEL first, with proper error handling
try:
    import camel
    from camel.agents import ChatAgent as BaseAgentClass
    USING_CAMEL = True
    logger.info("Using CAMEL-AI framework")
except ImportError:
    # Fallback implementation
    class BaseAgentClass:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get('name', 'Unknown')
            self.system_message = kwargs.get('system_message', '')
            self.config = kwargs.get('model_config', {})
            self.memory = []
            self.status = "idle"
            self.last_activity = None
            self.total_tasks_processed = 0
            self.error_count = 0
            self.success_rate = 1.0
            self.logger = logger.bind(agent_name=self.name)
        def get_status(self):
            return {
                "agent_name": self.name,
                "status": self.status,
                "last_activity": self.last_activity,
                "metrics": {
                    "total_tasks_processed": self.total_tasks_processed,
                    "error_count": self.error_count,
                    "success_rate": self.success_rate
                }
            }
    USING_CAMEL = False
    logger.info("Using fallback agent implementation")

class BaseAgent(ABC):
    """Base class for all agents with CAMEL-AI compatibility"""
    
    def __init__(self, name: str, model_config: Dict[str, Any]):
        self.name = name
        self.model_config = model_config
        self.agent_id = str(uuid.uuid4())
        self.memory = []
        self.status = "idle"
        self.last_activity = None
        self.total_tasks_processed = 0
        self.error_count = 0
        self.success_rate = 1.0
        self.config = model_config
        self.logger = logger.bind(agent_name=name, agent_id=self.agent_id)
        
        # Initialize base agent with proper error handling
        if USING_CAMEL:
            try:
                # CAMEL initialization - use their actual API
                # ChatAgent only accepts system_message and model_config, not 'name'
                self.base_agent = BaseAgentClass(
                    system_message=f"You are {name}, a specialized AI agent",
                    model_config=model_config
                )
            except Exception as e:
                logger.warning(f"CAMEL initialization failed: {e}, using fallback object")
                # Create a simple fallback object instead of trying ChatAgent again
                class FallbackAgent:
                    def __init__(self, name, system_message, model_config):
                        self.name = name
                        self.system_message = system_message
                        self.model_config = model_config
                        self.memory = []
                        self.status = "idle"
                self.base_agent = FallbackAgent(name, f"You are {name}, a specialized AI agent", model_config)
        else:
            # Fallback initialization - create a simple object
            class FallbackAgent:
                def __init__(self, name, system_message, model_config):
                    self.name = name
                    self.system_message = system_message
                    self.model_config = model_config
                    self.memory = []
                    self.status = "idle"
            self.base_agent = FallbackAgent(name, f"You are {name}, a specialized AI agent", model_config)
    
    async def process(self, task: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Main processing method with full error handling"""
        # #region agent log
        import json
        log_path = r"d:\erine_project\.cursor\debug.log"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "base_agent.py:83", "message": "BaseAgent.process entry", "data": {"agent_name": self.name, "task_type": task.get("type") if task else None}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
        task_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            self.status = "processing"
            self.last_activity = start_time
            self.logger.info(f"Starting task {task_id}")
            
            # Validate task
            if not self._validate_task(task):
                # #region agent log
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "F", "location": "base_agent.py:94", "message": "Task validation failed", "data": {"agent_name": self.name, "task": task}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                except: pass
                # #endregion
                raise ValueError(f"Invalid task format for {self.name}")
            
            # Execute task
            # #region agent log
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "base_agent.py:98", "message": "Before _execute_task", "data": {"agent_name": self.name}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            result = await self._execute_task(task, context)
            # #region agent log
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "base_agent.py:98", "message": "After _execute_task", "data": {"agent_name": self.name, "result_type": type(result).__name__ if result else None}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            
            # Validate result
            if not self._validate_result(result):
                raise ValueError(f"Invalid result format from {self.name}")
            
            # Update metrics
            self.total_tasks_processed += 1
            self.status = "completed"
            
            # Add to memory
            self._add_to_memory(task, result, task_id, start_time, datetime.now())
            
            self.logger.info(f"Task {task_id} completed successfully")
            
            # Return standardized result format
            return self._format_result(result, task_id, start_time, datetime.now())
            
        except Exception as e:
            # #region agent log
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "base_agent.py:116", "message": "Exception in BaseAgent.process", "data": {"agent_name": self.name, "exception_type": type(e).__name__, "exception_message": str(e), "exception_traceback": str(e.__traceback__) if hasattr(e, "__traceback__") else None}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            self.status = "error"
            self.error_count += 1
            self._update_success_rate()
            
            self.logger.error(f"Task {task_id} failed: {str(e)}")
            
            # Return error result
            return self._format_error_result(str(e), task_id, start_time, datetime.now())
    
    @abstractmethod
    async def _execute_task(self, task: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute the specific task - implemented by each agent"""
        pass
    
    def _validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate task format"""
        required_fields = ["type"]
        return all(field in task for field in required_fields)
    
    def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate result format"""
        return isinstance(result, dict)
    
    def _add_to_memory(self, task: Dict[str, Any], result: Dict[str, Any], task_id: str, 
                       start_time: datetime, end_time: datetime):
        """Add task and result to agent memory with full metadata"""
        memory_entry = {
            "task_id": task_id,
            "timestamp": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds()
            },
            "task": task,
            "result": result,
            "status": self.status,
            "agent_name": self.name,
            "agent_id": self.agent_id
        }
        
        self.memory.append(memory_entry)
        
        # Keep only last 1000 entries to prevent memory overflow
        if len(self.memory) > 1000:
            self.memory = self.memory[-1000:]
    
    def _sanitize_for_json(self, obj: Any) -> Any:
        """Recursively sanitize object to be JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items() if not callable(v) and not k.startswith('_')}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj if not callable(item)]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif hasattr(obj, '__dict__'):
            # For custom objects, try to convert to dict
            try:
                return self._sanitize_for_json(obj.__dict__)
            except:
                return str(obj)
        else:
            # For anything else, convert to string
            return str(obj)
    
    def _format_result(self, result: Dict[str, Any], task_id: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Format result with standardized metadata"""
        # Sanitize result to ensure JSON serializability
        sanitized_result = self._sanitize_for_json(result)
        
        return {
            "success": True,
            "task_id": task_id,
            "agent_name": self.name,
            "agent_id": self.agent_id,
            "timestamp": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds()
            },
            "status": self.status,
            "result": sanitized_result,
            "metrics": {
                "confidence": sanitized_result.get("confidence", 0.8) if isinstance(sanitized_result, dict) else 0.8,
                "processing_time": (end_time - start_time).total_seconds()
            }
        }
    
    def _format_error_result(self, error_message: str, task_id: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Format error result"""
        return {
            "success": False,
            "task_id": task_id,
            "agent_name": self.name,
            "agent_id": self.agent_id,
            "timestamp": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds()
            },
            "status": "error",
            "error": {
                "message": error_message,
                "type": "processing_error"
            },
            "result": None,
            "metrics": {
                "confidence": 0.0,
                "processing_time": (end_time - start_time).total_seconds()
            }
        }
    
    def _update_success_rate(self):
        """Update success rate based on recent performance"""
        if self.total_tasks_processed > 0:
            self.success_rate = (self.total_tasks_processed - self.error_count) / self.total_tasks_processed
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            "agent_name": self.name,
            "agent_id": self.agent_id,
            "status": self.status,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "metrics": {
                "total_tasks_processed": self.total_tasks_processed,
                "error_count": self.error_count,
                "success_rate": self.success_rate,
                "memory_size": len(self.memory)
            },
            "config": {
                "model": self.model_config.get("model", "unknown"),
                "temperature": self.model_config.get("temperature", 0.7)
            }
        }
    
    def get_memory(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent memory entries"""
        return self.memory[-limit:] if self.memory else []
    
    def clear_memory(self):
        """Clear agent memory"""
        self.memory = []
        self.logger.info("Agent memory cleared")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        if not self.memory:
            return {"error": "No tasks processed yet"}
        
        recent_tasks = self.memory[-100:]  # Last 100 tasks
        successful_tasks = [task for task in recent_tasks if task["status"] == "completed"]
        failed_tasks = [task for task in recent_tasks if task["status"] == "error"]
        
        avg_processing_time = sum(
            task["timestamp"]["duration"] for task in successful_tasks
        ) / len(successful_tasks) if successful_tasks else 0
        
        return {
            "agent_name": self.name,
            "period": "last_100_tasks",
            "total_tasks": len(recent_tasks),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": len(successful_tasks) / len(recent_tasks) if recent_tasks else 0,
            "average_processing_time": avg_processing_time,
            "recent_errors": [
                {
                    "task_id": task["task_id"],
                    "error": task["result"]["error"]["message"],
                    "timestamp": task["timestamp"]["start"]
                }
                for task in failed_tasks[-5:]  # Last 5 errors
            ]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            "agent_name": self.name,
            "status": self.status,
            "healthy": self.status != "error" and self.success_rate > 0.5,
            "success_rate": self.success_rate,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }

# Utility functions for all agents
def create_agent_task(task_type: str, data: Dict[str, Any], priority: str = "normal") -> Dict[str, Any]:
    """Create standardized task format"""
    return {
        "type": task_type,
        "data": data,
        "priority": priority,
        "created_at": datetime.now().isoformat(),
        "metadata": {
            "source": "user",
            "version": "1.0"
        }
    }

def validate_agent_config(config: Dict[str, Any], required_fields: List[str]) -> bool:
    """Validate agent configuration"""
    return all(field in config for field in required_fields)