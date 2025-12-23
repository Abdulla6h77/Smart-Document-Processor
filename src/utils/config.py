import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from loguru import logger

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        # Load environment variables first
        load_dotenv()
        
        # Find config file
        config_file = Path(config_path)
        if not config_file.exists():
            # Try to find in parent directory
            config_file = Path("src") / config_path
            if not config_file.exists():
                # Create default config if not found
                return create_default_config()
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables
        config = override_with_env(config)
        
        logger.info(f"Configuration loaded from {config_file}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        return create_default_config()

def override_with_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """Override config with environment variables"""
    if os.getenv("OPENROUTER_API_KEY"):
        config["agents"]["coordinator"]["openrouter_api_key"] = os.getenv("OPENROUTER_API_KEY")
        config["agents"]["coordinator"]["openrouter_base_url"] = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        config["agents"]["coordinator"]["ernie_model_name"] = os.getenv("ERNIE_MODEL_NAME", "baidu/ernie-4.0-turbo-8k")
    
    return config

def create_default_config() -> Dict[str, Any]:
     """Create default configuration with OpenRouter support"""
     return {
        "agents": {
            "coordinator": {
                "name": "DocumentCoordinator",
                "model": "ernie-4.5-turbo-32k",
                "temperature": 0.7,
                "max_tokens": 4000,
                "timeout": 30,
                # OpenRouter settings (new)
                "openrouter_api_key": None,
                "openrouter_base_url": "https://openrouter.ai/api/v1",
                "ernie_model_name": "baidu/ernie-4.0-turbo-8k"
            },
            "ocr": {
                "name": "OCRAgent",
                "model": "paddleocr-vl-0.9b",
                "languages": ["en", "ch"],
                "confidence_threshold": 0.8,
                "use_gpu": False
            },
            "analysis": {
                "name": "AnalysisAgent",
                "model": "ernie-4.5-turbo-8k",
                "temperature": 0.5,
                "max_tokens": 2000,
                "timeout": 20
            },
            "validation": {
                "name": "ValidationAgent",
                "model": "ernie-4.5-turbo-8k",
                "temperature": 0.3,
                "max_tokens": 1500,
                "timeout": 15
            }
        },
        "processing": {
            "supported_formats": ["pdf", "jpg", "png", "jpeg", "tiff", "bmp", "docx"],
            "max_file_size_mb": 50,
            "batch_size": 10,
            "timeout_seconds": 300
        },
        "output": {
            "formats": ["json", "markdown", "csv", "html"],
            "include_confidence_scores": True,
            "generate_summary": True
        },
        "debug": False
    }

def save_config(config: Dict[str, Any], config_path: str = "config.yaml"):
    """Save configuration to file"""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config: {str(e)}")

def get_agent_config(config: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    """Get configuration for specific agent"""
    return config.get("agents", {}).get(agent_name, {})