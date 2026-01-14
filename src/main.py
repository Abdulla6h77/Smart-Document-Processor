import asyncio
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.coordinator_agent import CoordinatorAgent
from utils.config import load_config
from utils.logger import setup_logger

# Create FastAPI app
app = FastAPI(
    title="Smart Document Processor",
    description="Multi-agent document processing system using ERNIE and PaddleOCR",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
config = None
coordinator = None

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    global config, coordinator
    
    try:
        # Setup logging
        setup_logger()
        logger.info("Starting Smart Document Processor...")
        
        # Load configuration
        config = load_config()
        
        # Initialize coordinator agent
        coordinator_config = config["agents"]["coordinator"]
        coordinator = CoordinatorAgent(
            name=coordinator_config["name"],
            model_config=coordinator_config
        )
        
        # Setup coordinator with full config
        await coordinator.setup(config)
        
        # Create necessary directories
        Path("uploads").mkdir(exist_ok=True)
        Path("outputs").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        logger.info("Smart Document Processor started successfully!")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global coordinator
    
    try:
        logger.info("Shutting down Smart Document Processor...")
        
        if coordinator:
            await coordinator.close()
            
        logger.info("Shutdown completed successfully!")
        
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with UI"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Document Processor</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .upload-area { border: 2px dashed #ccc; border-radius: 10px; padding: 40px; text-align: center; margin: 30px 0; }
            .btn { background: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            .btn:hover { background: #0056b3; }
            select, input[type="file"] { margin: 10px; padding: 10px; font-size: 16px; }
            .status { margin-top: 20px; padding: 15px; border-radius: 5px; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .agent-status { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px; }
            .agent-card { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }
            .agent-name { font-weight: bold; color: #007bff; }
            .agent-status-text { color: #666; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Smart Document Processor</h1>
            <p style="text-align: center; color: #666;">Multi-Agent Document Processing with ERNIE & PaddleOCR</p>
            
            <div class="upload-area">
                <h3>Upload Document for Processing</h3>
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" name="file" id="fileInput" accept=".pdf,.jpg,.jpeg,.png,.tiff,.bmp,.docx" required>
                    <br><br>
                    <select name="extraction_type" id="extractionType">
                        <option value="text">Text Extraction</option>
                        <option value="table">Table Extraction</option>
                        <option value="structure">Structure Extraction</option>
                    </select>
                    <br><br>
                    <select name="output_format" id="outputFormat">
                        <option value="json">JSON Output</option>
                        <option value="markdown">Markdown Output</option>
                        <option value="html">HTML Output</option>
                        <option value="csv">CSV Output</option>
                    </select>
                    <br><br>
                    <button type="submit" class="btn">Process Document</button>
                </form>
            </div>
            
            <div id="status"></div>
            <div id="results"></div>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('fileInput');
                const extractionType = document.getElementById('extractionType').value;
                const outputFormat = document.getElementById('outputFormat').value;
                
                formData.append('file', fileInput.files[0]);
                formData.append('extraction_type', extractionType);
                formData.append('output_format', outputFormat);
                
                const statusDiv = document.getElementById('status');
                const resultsDiv = document.getElementById('results');
                
                statusDiv.innerHTML = '<div class="status">Processing document...</div>';
                resultsDiv.innerHTML = '';
                
                try {
                    const response = await fetch('/process-document', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        statusDiv.innerHTML = '<div class="status success">Document processed successfully!</div>';
                        resultsDiv.innerHTML = `<h3>Results:</h3><pre style="background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">${JSON.stringify(result, null, 2)}</pre>`;
                    } else {
                        statusDiv.innerHTML = `<div class="status error">Error: ${result.detail || 'Unknown error'}</div>`;
                    }
                } catch (error) {
                    statusDiv.innerHTML = `<div class="status error">Network error: ${error.message}</div>`;
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    extraction_type: str = Form("text"),
    output_format: str = Form("json"),
    analysis_type: str = Form("auto")
):
    """Process a document through the multi-agent system"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file extension
        allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.docx'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"File type not supported. Allowed: {allowed_extensions}")
        
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Received document: {file.filename} ({len(content)} bytes)")
        
        # Create task for coordinator
        task = {
            "type": "document_processing",
            "document_path": str(file_path),
            "extraction_type": extraction_type,
            "output_format": output_format,
            "analysis_type": analysis_type,
            "original_filename": file.filename
        }
        
        # Process through multi-agent system
        result = await coordinator.process(task)
        
        # Clean up uploaded file
        try:
            file_path.unlink()
        except:
            pass
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent-status")
async def get_agent_status():
    """Get status of all agents"""
    if not coordinator:
        return JSONResponse(content={"error": "System not initialized"}, status_code=503)
    
    try:
        status = coordinator.get_all_agents_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Failed to get agent status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "version": "1.0.0",
            "coordinator_ready": coordinator is not None and coordinator.setup_complete,
            "timestamp": datetime.now().isoformat()
        }
        return JSONResponse(content=health_status)
    except Exception as e:
        return JSONResponse(content={"status": "unhealthy", "error": str(e)}, status_code=500)

@app.get("/config")
async def get_config():
    """Get current configuration"""
    if not config:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Return safe config (without sensitive data)
    safe_config = {
        "agents": {
            "coordinator": {
                "name": config["agents"]["coordinator"]["name"],
                "model": config["agents"]["coordinator"]["model"]
            },
            "ocr": {
                "name": config["agents"]["ocr"]["name"],
                "languages": config["agents"]["ocr"]["languages"],
                "confidence_threshold": config["agents"]["ocr"]["confidence_threshold"]
            },
            "analysis": {
                "name": config["agents"]["analysis"]["name"],
                "model": config["agents"]["analysis"]["model"]
            },
            "validation": {
                "name": config["agents"]["validation"]["name"],
                "model": config["agents"]["validation"]["model"]
            }
        },
        "processing": {
            "supported_formats": config["processing"]["supported_formats"],
            "max_file_size_mb": config["processing"]["max_file_size_mb"]
        }
    }
    
    return JSONResponse(content=safe_config)

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )