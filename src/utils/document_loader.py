import asyncio
import fitz  # PyMuPDF
from PIL import Image
import io
from typing import List, Dict, Any, Optional
from pathlib import Path
import cv2
import numpy as np
from loguru import logger
import os

class DocumentLoader:
    """Complete document loading and preprocessing utility"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_formats = config.get("supported_formats", ["pdf", "jpg", "png", "jpeg", "tiff", "bmp", "docx"])
        self.max_file_size = config.get("max_file_size_mb", 50) * 1024 * 1024  # Convert to bytes
        self.quality_threshold = config.get("quality_threshold", 300)  # DPI
    
    async def load_document(self, file_path: str) -> List[np.ndarray]:
        """Load document and convert to list of images"""
        try:
            path_obj = Path(file_path)
            
            # Check file size
            if path_obj.stat().st_size > self.max_file_size:
                raise ValueError(f"File size exceeds maximum allowed ({self.max_file_size // (1024*1024)}MB)")
            
            # Check file format
            if path_obj.suffix.lower().lstrip('.') not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {path_obj.suffix}")
            
            logger.info(f"Loading document: {file_path}")
            
            # Load based on file type
            if path_obj.suffix.lower() == '.pdf':
                images = await self._load_pdf(file_path)
            elif path_obj.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                images = await self._load_image(file_path)
            elif path_obj.suffix.lower() == '.docx':
                images = await self._load_docx(file_path)
            else:
                raise ValueError(f"File format not implemented: {path_obj.suffix}")
            
            logger.info(f"Document loaded successfully: {len(images)} pages/images")
            return images
            
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {str(e)}")
            raise
    
    async def _load_pdf(self, pdf_path: str) -> List[np.ndarray]:
        """Load PDF and convert each page to image"""
        images = []
        
        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            
            # Convert each page
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Render page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to numpy array
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                img_array = np.array(img)
                
                # Convert RGB to BGR for OpenCV compatibility
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                images.append(img_array)
                
                logger.debug(f"PDF page {page_num + 1} converted to image")
            
            pdf_document.close()
            return images
            
        except Exception as e:
            logger.error(f"PDF loading failed: {str(e)}")
            raise
    
    async def _load_image(self, image_path: str) -> List[np.ndarray]:
        """Load single image file"""
        try:
            # Read image with OpenCV
            img = cv2.imread(image_path)
            
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Check image quality
            height, width = img.shape[:2]
            if height < 100 or width < 100:
                logger.warning(f"Image dimensions very small: {width}x{height}")
            
            return [img]
            
        except Exception as e:
            logger.error(f"Image loading failed: {str(e)}")
            raise
    
    async def _load_docx(self, docx_path: str) -> List[np.ndarray]:
        """Load DOCX file (convert to images)"""
        try:
            # For now, we'll use a simple approach
            # In production, you'd use python-docx and convert to PDF first
            
            # Placeholder implementation - convert to PDF then load
            # This would require additional libraries like docx2pdf
            
            logger.warning("DOCX loading not fully implemented, converting to PDF first...")
            
            # For now, raise not implemented
            raise NotImplementedError("DOCX loading requires additional setup")
            
        except Exception as e:
            logger.error(f"DOCX loading failed: {str(e)}")
            raise
    
    async def preprocess_images(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Apply preprocessing to improve OCR quality"""
        processed_images = []
        
        for i, img in enumerate(images):
            try:
                logger.debug(f"Preprocessing image {i + 1}")
                
                # Apply preprocessing pipeline
                processed = await self._enhance_image(img)
                processed_images.append(processed)
                
            except Exception as e:
                logger.warning(f"Preprocessing failed for image {i + 1}: {str(e)}")
                processed_images.append(img)  # Use original if preprocessing fails
        
        return processed_images
    
    async def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply image enhancement for better OCR"""
        # Create a copy to avoid modifying original
        img = image.copy()
        
        # Convert to grayscale if colored
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Optional: Resize if image is very small
        height, width = cleaned.shape
        if height < 1000 or width < 1000:
            scale_factor = 2.0
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return cleaned
    
    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about the document"""
        try:
            path_obj = Path(file_path)
            stat = path_obj.stat()
            
            info = {
                "filename": path_obj.name,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "extension": path_obj.suffix.lower(),
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "supported": path_obj.suffix.lower().lstrip('.') in self.supported_formats
            }
            
            # Additional info for images
            if info["extension"] in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                try:
                    with Image.open(file_path) as img:
                        info["dimensions"] = {
                            "width": img.width,
                            "height": img.height
                        }
                        info["mode"] = img.mode
                        info["format"] = img.format
                except Exception as e:
                    logger.warning(f"Could not get image info: {str(e)}")
            
            # Additional info for PDFs
            elif info["extension"] == '.pdf':
                try:
                    with fitz.open(file_path) as pdf:
                        info["pages"] = pdf.page_count
                        info["pdf_version"] = pdf.version
                        info["encrypted"] = pdf.is_encrypted
                except Exception as e:
                    logger.warning(f"Could not get PDF info: {str(e)}")
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get document info: {str(e)}")
            return {"error": str(e)}

# Utility functions
async def save_processed_images(images: List[np.ndarray], output_dir: str, base_name: str):
    """Save processed images for debugging"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    saved_paths = []
    
    for i, img in enumerate(images):
        file_name = f"{base_name}_page_{i+1}.png"
        file_path = output_path / file_name
        
        # Convert BGR to RGB for saving
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        cv2.imwrite(str(file_path), img_rgb)
        saved_paths.append(str(file_path))
    
    return saved_paths