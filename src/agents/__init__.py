# agents/__init__.py - Change to absolute imports
from src.agents.base_agent import BaseAgent
from src.agents.coordinator_agent import CoordinatorAgent
from src.agents.ocr_agent import OCRAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.validation_agent import ValidationAgent

__all__ = [
    'BaseAgent',
    'CoordinatorAgent', 
    'OCRAgent',
    'AnalysisAgent',
    'ValidationAgent'
]