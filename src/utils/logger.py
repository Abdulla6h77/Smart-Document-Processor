import sys
import os
from pathlib import Path
from loguru import logger
import json

def setup_logger(log_level: str = "INFO", log_file: str = "logs/app.log"):
    """Setup logging configuration"""
    
    # Remove default logger
    logger.remove()
    
    # Create logs directory
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)
    
    # Console logger with colored output
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File logger with rotation
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        serialize=False,
        backtrace=True,
        diagnose=True
    )
    
    # Error log file
    logger.add(
        "logs/errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
        level="ERROR",
        rotation="5 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # JSON log for structured logging
    logger.add(
        "logs/app.json",
        format="{message}",
        level=log_level,
        rotation="10 MB",
        retention="10 days",
        serialize=True,
        backtrace=False
    )
    
    logger.info("Logging system initialized")
    return logger

def log_agent_activity(agent_name: str, activity: str, details: dict = None):
    """Log agent-specific activities"""
    log_data = {
        "event_type": "agent_activity",
        "agent": agent_name,
        "activity": activity,
        "details": details or {},
        "timestamp": logger.catch
    }
    logger.info(f"AGENT_ACTIVITY: {json.dumps(log_data)}")

def log_processing_metrics(document_type: str, processing_time: float, success: bool, error: str = None):
    """Log document processing metrics"""
    metrics = {
        "event_type": "processing_metrics",
        "document_type": document_type,
        "processing_time": processing_time,
        "success": success,
        "error": error,
        "timestamp": logger.catch
    }
    logger.info(f"PROCESSING_METRICS: {json.dumps(metrics)}")

def log_api_call(endpoint: str, method: str, status_code: int, response_time: float):
    """Log API call metrics"""
    api_metrics = {
        "event_type": "api_call",
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "response_time": response_time,
        "timestamp": logger.catch
    }
    logger.info(f"API_CALL: {json.dumps(api_metrics)}")