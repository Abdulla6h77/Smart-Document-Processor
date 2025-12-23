from typing import Dict, Any, Optional, List
import asyncio
import time
from datetime import datetime
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from loguru import logger
# Import fix for async operations
import io  # Add this import at the top of the file

# Change to absolute import
from src.models.paddle_ocr_model import PaddleOCRModel
from src.agents.base_agent import BaseAgent

class OCRAgent(BaseAgent):
    """Complete OCR Agent with advanced image processing and text extraction"""
    
    def __init__(self, name: str, model_config: Dict[str, Any]):
        super().__init__(name, model_config)
        
        # Initialize PaddleOCR model
        self.ocr_model = PaddleOCRModel(model_config)
        
        # Configuration
        self.confidence_threshold = model_config.get("confidence_threshold", 0.8)
        self.supported_formats = model_config.get("supported_formats", ["pdf", "jpg", "png", "jpeg", "tiff", "bmp"])
        self.max_file_size = model_config.get("max_file_size_mb", 50) * 1024 * 1024
        self.preprocessing_enabled = model_config.get("preprocessing_enabled", True)
        self.extract_tables = model_config.get("extract_tables", True)
        self.extract_structure = model_config.get("extract_structure", True)
        
        # Performance settings
        self.batch_size = model_config.get("batch_size", 5)
        self.timeout = model_config.get("timeout", 300)
        
        self.logger.info(f"OCR Agent initialized with confidence threshold: {self.confidence_threshold}")
    
    async def _execute_task(self, task: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute OCR extraction task with full processing pipeline"""
        document_path = task.get("document_path")
        extraction_type = task.get("extraction_type", "text")
        image_data = task.get("image")  # For direct image data
        preprocess = task.get("preprocessing", self.preprocessing_enabled)
        
        if not document_path and image_data is None:
            raise ValueError("Either document_path or image data is required for OCR task")
        
        self.logger.info(f"Starting OCR extraction: {extraction_type}")
        
        # Load and preprocess images
        if image_data is not None:
            images = [image_data] if isinstance(image_data, np.ndarray) else image_data
            doc_info = {"filename": "direct_image", "extension": ".png", "size_mb": 0}
        else:
            images, doc_info = await self._load_and_preprocess_document(document_path, preprocess)
        
        # Perform OCR extraction based on type
        if extraction_type == "text":
            result = await self._extract_text_from_images(images)
        elif extraction_type == "table":
            result = await self._extract_tables_from_images(images)
        elif extraction_type == "structure":
            result = await self._extract_structure_from_images(images)
        elif extraction_type == "full":
            result = await self._extract_full_content(images)
        else:
            result = await self._extract_text_from_images(images)
        
        # Add document metadata
        result["document_info"] = doc_info
        result["extraction_type"] = extraction_type
        result["images_processed"] = len(images)
        result["preprocessing_applied"] = preprocess
        
        # Calculate overall confidence
        result["confidence"] = self._calculate_overall_confidence(result)
        
        self.logger.info(f"OCR extraction completed: {len(result.get('extracted_items', []))} items, confidence: {result['confidence']:.2f}")
        
        return result
    
    async def _load_and_preprocess_document(self, document_path: str, preprocess: bool) -> tuple[List[np.ndarray], Dict[str, Any]]:
        """Load document and preprocess images"""
        try:
            path_obj = Path(document_path)
            
            # Validate file
            if not path_obj.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")
            
            if path_obj.stat().st_size > self.max_file_size:
                raise ValueError(f"File size exceeds maximum allowed ({self.max_file_size // (1024*1024)}MB)")
            
            if path_obj.suffix.lower().lstrip('.') not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {path_obj.suffix}")
            
            # Get document info
            doc_info = {
                "filename": path_obj.name,
                "extension": path_obj.suffix.lower(),
                "size_bytes": path_obj.stat().st_size,
                "size_mb": round(path_obj.stat().st_size / (1024 * 1024), 2),
                "created": path_obj.stat().st_ctime,
                "modified": path_obj.stat().st_mtime
            }
            
            self.logger.info(f"Loading document: {path_obj.name} ({doc_info['size_mb']} MB)")
            
            # Load images based on file type
            if path_obj.suffix.lower() == '.pdf':
                images = await self._load_pdf(document_path)
            elif path_obj.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                images = await self._load_image(document_path)
            else:
                raise ValueError(f"File format not implemented: {path_obj.suffix}")
            
            # Add image dimensions to doc_info
            if images:
                height, width = images[0].shape[:2]
                doc_info["dimensions"] = {"width": width, "height": height, "pages": len(images)}
            
            # Preprocess if enabled
            if preprocess:
                processed_images = await self._preprocess_images(images)
                self.logger.info(f"Preprocessing completed: {len(images)} images enhanced")
            else:
                processed_images = images
            
            return processed_images, doc_info
            
        except Exception as e:
            self.logger.error(f"Document loading failed: {str(e)}")
            raise
    
    async def _load_pdf(self, pdf_path: str) -> List[np.ndarray]:
        """Load PDF and convert each page to image"""
        images = []
        
        try:
            import fitz  # PyMuPDF
            pdf_document = fitz.open(pdf_path)
            
            self.logger.info(f"PDF loaded: {pdf_document.page_count} pages")
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Render page at high quality
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR quality
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to numpy array
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                img_array = np.array(img)
                
                # Convert RGB to BGR for OpenCV compatibility
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                images.append(img_array)
                
                if page_num % 10 == 0:  # Log progress
                    self.logger.debug(f"Processed PDF page {page_num + 1}")
            
            pdf_document.close()
            return images
            
        except ImportError:
            self.logger.error("PyMuPDF not installed. Install with: pip install PyMuPDF")
            raise
        except Exception as e:
            self.logger.error(f"PDF loading failed: {str(e)}")
            raise
    
    async def _load_image(self, image_path: str) -> List[np.ndarray]:
        """Load single image file"""
        try:
            img = cv2.imread(image_path)
            
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            self.logger.info(f"Image loaded: {img.shape[1]}x{img.shape[0]} pixels")
            
            return [img]
            
        except Exception as e:
            self.logger.error(f"Image loading failed: {str(e)}")
            raise
    
    async def _preprocess_images(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Apply advanced preprocessing to improve OCR quality"""
        processed_images = []
        
        for i, img in enumerate(images):
            try:
                self.logger.debug(f"Preprocessing image {i + 1}")
                
                # Apply comprehensive preprocessing pipeline
                processed = await self._enhance_image(img)
                processed_images.append(processed)
                
            except Exception as e:
                self.logger.warning(f"Preprocessing failed for image {i + 1}: {str(e)}")
                processed_images.append(img)  # Use original if preprocessing fails
        
        return processed_images
    
    async def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply comprehensive image enhancement for better OCR"""
        img = image.copy()
        
        # Step 1: Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Step 2: Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Step 3: Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Step 4: Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Step 5: Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Step 6: Optional resize for very small images
        height, width = cleaned.shape
        if height < 1000 or width < 1000:
            scale_factor = min(2000 / height, 2000 / width)  # Max dimension 2000
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return cleaned
    
    async def _extract_text_from_images(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """Extract text from multiple images"""
        all_items = []
        total_processing_time = 0
        
        # Process images in batches for better performance
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            
            # Process batch
            batch_results = await self._process_batch(batch)
            
            for j, result in enumerate(batch_results):
                page_num = i + j + 1
                
                # Filter by confidence threshold
                filtered_items = [
                    item for item in result 
                    if item["confidence"] >= self.confidence_threshold
                ]
                
                # Add page information
                for item in filtered_items:
                    item["page"] = page_num
                
                all_items.extend(filtered_items)
                total_processing_time += sum(item.get("processing_time", 0) for item in result)
        
        # Combine text
        extracted_text = " ".join([item["text"] for item in all_items])
        
        return {
            "extracted_content": extracted_text,
            "extracted_items": all_items,
            "confidence_scores": [item["confidence"] for item in all_items],
            "total_items": len(all_items),
            "filtered_items": len([item for item in all_items if item["confidence"] >= self.confidence_threshold]),
            "processing_time": total_processing_time,
            "pages_processed": len(images),
            "confidence": sum(item["confidence"] for item in all_items) / len(all_items) if all_items else 0
        }
    
    async def _extract_tables_from_images(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """Extract tables from images"""
        all_tables = []
        total_processing_time = 0
        
        for i, image in enumerate(images):
            start_time = time.time()
            
            # Use PaddleOCR table extraction
            tables = await self.ocr_model.extract_tables(image)
            
            processing_time = time.time() - start_time
            
            # Add page information
            for table in tables:
                table["page"] = i + 1
                table["processing_time"] = processing_time
            
            all_tables.extend(tables)
            total_processing_time += processing_time
        
        return {
            "extracted_content": f"Found {len(all_tables)} tables across {len(images)} pages",
            "extracted_tables": all_tables,
            "total_tables": len(all_tables),
            "processing_time": total_processing_time,
            "pages_processed": len(images),
            "confidence": 0.85  # Default confidence for table extraction
        }
    
    async def _extract_structure_from_images(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """Extract document structure (headings, paragraphs, etc.)"""
        all_structure = []
        total_processing_time = 0
        
        for i, image in enumerate(images):
            start_time = time.time()
            
            # Extract structure
            structure_result = await self.ocr_model.extract_structure(image)
            
            processing_time = time.time() - start_time
            
            # Add page information
            structure_elements = structure_result.get("structure", [])
            for element in structure_elements:
                element["page"] = i + 1
            
            all_structure.extend(structure_elements)
            total_processing_time += processing_time
        
        return {
            "extracted_content": f"Extracted structure from {len(images)} pages",
            "structure_elements": all_structure,
            "total_elements": len(all_structure),
            "processing_time": total_processing_time,
            "pages_processed": len(images),
            "confidence": sum(item.get("confidence", 0.8) for item in all_structure) / len(all_structure) if all_structure else 0
        }
    
    async def _extract_full_content(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """Extract all types of content (text, tables, structure)"""
        # Run all extraction types in parallel
        text_task = self._extract_text_from_images(images)
        table_task = self._extract_tables_from_images(images)
        structure_task = self._extract_structure_from_images(images)
        
        text_result, table_result, structure_result = await asyncio.gather(
            text_task, table_task, structure_task
        )
        
        # Combine results
        return {
            "extracted_content": text_result["extracted_content"],
            "text_extraction": text_result,
            "table_extraction": table_result,
            "structure_extraction": structure_result,
            "total_elements": (
                text_result.get("total_items", 0) + 
                table_result.get("total_tables", 0) + 
                len(structure_result.get("structure_elements", []))
            ),
            "processing_time": (
                text_result.get("processing_time", 0) + 
                table_result.get("processing_time", 0) + 
                structure_result.get("processing_time", 0)
            ),
            "pages_processed": len(images),
            "confidence": self._calculate_combined_confidence([text_result, table_result, structure_result])
        }
    
    async def _process_batch(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """Process multiple images in batch"""
        tasks = [self.ocr_model.extract_text(image) for image in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch processing failed for image {i}: {str(result)}")
                processed_results.append([])
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _calculate_overall_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        confidence_scores = []
        
        # Get confidence from different sources
        if "confidence_scores" in result:
            scores = result["confidence_scores"]
            if scores:
                confidence_scores.append(sum(scores) / len(scores))
        
        if "confidence" in result:
            confidence_scores.append(result["confidence"])
        
        # Default confidence if no scores available
        if not confidence_scores:
            return 0.8
        
        return sum(confidence_scores) / len(confidence_scores)
    
    def _calculate_combined_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate combined confidence from multiple extraction types"""
        confidences = []
        for result in results:
            if "confidence" in result:
                confidences.append(result["confidence"])
        
        return sum(confidences) / len(confidences) if confidences else 0.8
    
    async def close(self):
        """Cleanup resources"""
        # Add any cleanup logic here
        self.logger.info("OCR Agent closed")

