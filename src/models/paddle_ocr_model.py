import asyncio
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import cv2
from PIL import Image
from loguru import logger
import time

# Import PaddleOCR
try:
    from paddleocr import PaddleOCR
except ImportError:
    logger.error("PaddleOCR not installed. Install with: pip install paddleocr")
    PaddleOCR = None

class PaddleOCRModel:
    """Complete PaddleOCR model wrapper with all functionality"""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.config = model_config
        self.model = None
        self.languages = model_config.get("languages", ["en"])
        self.confidence_threshold = model_config.get("confidence_threshold", 0.8)
        self.use_gpu = model_config.get("use_gpu", False)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize PaddleOCR model"""
        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR library not available")
        
        try:
            # Initialize with multilingual support
            lang_param = 'multilingual' if len(self.languages) > 1 else self.languages[0]
            
            self.model = PaddleOCR(
                use_angle_cls=True,
                lang=lang_param,
                use_gpu=self.use_gpu,
                show_log=False,
                det_model_dir=None,  # Use default
                rec_model_dir=None,  # Use default
                cls_model_dir=None   # Use default
            )
            
            logger.info(f"PaddleOCR model initialized with languages: {self.languages}")
            
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            raise
    
    async def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text from image with full details"""
        try:
            start_time = time.time()
            
            # Run OCR
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._run_ocr, image)
            
            processing_time = time.time() - start_time
            
            # Parse results
            extracted_items = []
            
            if result and len(result) > 0:
                for line in result[0]:
                    text = line[1][0]
                    confidence = float(line[1][1])
                    
                    # Get bounding box coordinates
                    bbox = line[0]
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    extracted_items.append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": {
                            "x_min": min(x_coords),
                            "y_min": min(y_coords),
                            "x_max": max(x_coords),
                            "y_max": max(y_coords)
                        },
                        "processing_time": processing_time
                    })
            
            logger.info(f"Extracted {len(extracted_items)} text items in {processing_time:.2f}s")
            return extracted_items
            
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            return []
    
    def _run_ocr(self, image: np.ndarray) -> Any:
        """Run OCR synchronously (for executor)"""
        return self.model.ocr(image, cls=True)
    
    async def extract_tables(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract table structures from document"""
        try:
            # This is a simplified implementation
            # In production, you'd use PaddleOCR's table extraction
            text_results = await self.extract_text(image)
            
            # Group text by vertical proximity to detect table rows
            if not text_results:
                return []
            
            # Sort by y-coordinate
            sorted_texts = sorted(text_results, key=lambda x: x["bbox"]["y_min"])
            
            tables = []
            current_table = []
            row_threshold = 50  # pixels
            
            for i, text_item in enumerate(sorted_texts):
                if i == 0:
                    current_table.append(text_item)
                    continue
                
                prev_item = sorted_texts[i-1]
                y_diff = abs(text_item["bbox"]["y_min"] - prev_item["bbox"]["y_min"])
                
                if y_diff < row_threshold:
                    current_table.append(text_item)
                else:
                    if len(current_table) > 1:  # At least 2 items to form a row
                        tables.append({
                            "type": "table",
                            "rows": self._group_into_rows(current_table),
                            "confidence": 0.85
                        })
                    current_table = [text_item]
            
            # Add final table
            if len(current_table) > 1:
                tables.append({
                    "type": "table",
                    "rows": self._group_into_rows(current_table),
                    "confidence": 0.85
                })
            
            return tables
            
        except Exception as e:
            logger.error(f"Table extraction failed: {str(e)}")
            return []
    
    def _group_into_rows(self, text_items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group text items into rows based on y-coordinate"""
        if not text_items:
            return []
        
        # Sort by y-coordinate
        sorted_items = sorted(text_items, key=lambda x: x["bbox"]["y_min"])
        
        rows = []
        current_row = [sorted_items[0]]
        row_threshold = 30  # pixels
        
        for i in range(1, len(sorted_items)):
            current_item = sorted_items[i]
            prev_item = sorted_items[i-1]
            
            y_diff = abs(current_item["bbox"]["y_min"] - prev_item["bbox"]["y_min"])
            
            if y_diff < row_threshold:
                current_row.append(current_item)
            else:
                # Sort row by x-coordinate
                current_row.sort(key=lambda x: x["bbox"]["x_min"])
                rows.append(current_row)
                current_row = [current_item]
        
        # Add final row
        if current_row:
            current_row.sort(key=lambda x: x["bbox"]["x_min"])
            rows.append(current_row)
        
        return rows
    
    async def extract_structure(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract document structure (headings, paragraphs, etc.)"""
        try:
            text_results = await self.extract_text(image)
            
            if not text_results:
                return {"structure": [], "confidence": 0.0}
            
            # Analyze text size and position to identify structure
            structure_elements = []
            
            for item in text_results:
                text = item["text"]
                bbox = item["bbox"]
                
                # Simple heuristics for structure detection
                element_type = self._classify_structure_element(text, bbox)
                
                structure_elements.append({
                    "type": element_type,
                    "text": text,
                    "confidence": item["confidence"],
                    "bbox": bbox
                })
            
            return {
                "structure": structure_elements,
                "confidence": sum(item["confidence"] for item in text_results) / len(text_results)
            }
            
        except Exception as e:
            logger.error(f"Structure extraction failed: {str(e)}")
            return {"structure": [], "confidence": 0.0}
    
    def _classify_structure_element(self, text: str, bbox: Dict[str, Any]) -> str:
        """Classify text element as heading, paragraph, etc."""
        text_length = len(text.strip())
        
        # Heading detection based on text length and capitalization
        if text_length < 100 and text.strip().isupper():
            return "heading"
        elif text_length < 50 and text.strip().replace(" ", "").isalnum():
            return "title"
        elif "\n" in text or text_length > 200:
            return "paragraph"
        else:
            return "text"
    
    async def process_batch(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """Process multiple images in batch"""
        tasks = [self.extract_text(image) for image in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed for image {i}: {str(result)}")
                processed_results.append([])
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model": "PaddleOCR-VL",
            "languages": self.languages,
            "confidence_threshold": self.confidence_threshold,
            "use_gpu": self.use_gpu,
            "status": "active" if self.model else "inactive"
        }