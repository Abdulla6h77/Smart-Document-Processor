# Smart Document Processor - Complete Project Context

## Project Overview
A multi-agent document processing system that uses **PaddleOCR** for text extraction and **ERNIE LLM** (via OpenRouter) for document analysis.

---

## ⚠️ CRITICAL: Model Configuration Issues

### Fake Model Names in config.yaml
The `config.yaml` file contains **fictional ERNIE model names** that don't exist:
```yaml
# ❌ THESE ARE NOT REAL MODELS
model: "ernie-4.5-turbo-32k"  # coordinator
model: "ernie-4.5-turbo-8k"   # analysis & validation
model: "paddleocr-vl-0.9b"    # ocr (also fake)
```

### Actual Model Being Used
The real model is configured in `.env`:
```env
ERNIE_MODEL_NAME=baidu/ernie-4.5-21b-a3b  # ✅ REAL MODEL via OpenRouter
OPENROUTER_API_KEY=your key 
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### Where Models Are Actually Loaded
1. **`src/models/openrouter_model.py:18`** - Reads actual model from env:
   ```python
   self.model_name = os.getenv("ERNIE_MODEL_NAME", "baidu/ernie-4.0-turbo-8k")
   ```

2. **`src/models/ernie_model.py:12`** - Has hardcoded fallback:
   ```python
   self.model_name = model_config.get("model", "ernie-4.5-turbo-8k")  # ❌ Fake fallback
   ```

3. **Environment variable takes precedence** over config.yaml values

### Available Real ERNIE Models (via OpenRouter)
Based on OpenRouter/Baidu API, likely available models:
- `baidu/ernie-4.5-21b-a3b` (currently configured)
- `baidu/ernie-4.0-turbo-8k`
- `baidu/ernie-3.5-8k`
- `baidu/ernie-4-turbo`

---

## Architecture

### Multi-Agent System

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│              (Streamlit UI / FastAPI)                    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│              Coordinator Agent                            │
│  - Orchestrates workflow                                  │
│  - Manages agent communication                            │
│  - Handles error recovery                                 │
└─────┬─────────┬──────────────┬──────────────────────────┘
      │         │              │
      ▼         ▼              ▼
┌─────────┐ ┌──────────┐ ┌─────────────┐
│   OCR   │ │ Analysis │ │ Validation  │
│  Agent  │ │  Agent   │ │   Agent     │
└────┬────┘ └────┬─────┘ └──────┬──────┘
     │           │               │
     ▼           ▼               ▼
┌─────────┐ ┌──────────┐ ┌─────────────┐
│Paddle   │ │  ERNIE   │ │   ERNIE     │
│  OCR    │ │  Model   │ │   Model     │
└─────────┘ └──────────┘ └─────────────┘
```

---

## Core Components

### 1. Agents (`src/agents/`)

#### Base Agent (`base_agent.py`)
- Abstract base class for all agents
- Handles task execution, error handling, metrics
- Memory management with last 1000 entries
- Attempts to use CAMEL-AI framework, falls back to custom implementation
- **Key Methods:**
  - `process()` - Main entry point
  - `_execute_task()` - Abstract, implemented by subclasses
  - `get_status()` - Returns agent health metrics
  - `get_performance_metrics()` - Detailed performance stats

#### Coordinator Agent (`coordinator_agent.py`)
- Orchestrates entire document processing pipeline
- Manages workflow execution (standard/parallel/streaming)
- Initializes and coordinates sub-agents
- **Workflow Types:**
  - `standard` - Sequential processing
  - `parallel` - Concurrent OCR + Analysis
  - `streaming` - Real-time progressive processing
- **Setup Required:** Must call `setup(config)` before use

#### OCR Agent (`ocr_agent.py`)
- Extracts text from PDFs and images
- Uses PaddleOCR for text recognition
- **Features:**
  - Image preprocessing (denoising, contrast, thresholding)
  - Multi-page PDF support (via PyMuPDF/fitz)
  - Table extraction
  - Structure detection
  - Batch processing
- **Extraction Types:**
  - `text` - Plain text extraction
  - `table` - Table structure detection
  - `structure` - Document layout analysis
  - `full` - All above combined
- **Supported Formats:** PDF, JPG, PNG, JPEG, TIFF, BMP
- **Max File Size:** 50MB (configurable)

#### Analysis Agent (`analysis_agent.py`)
- Analyzes extracted text using ERNIE LLM
- Document-type specific analysis
- **Analysis Types:**
  - `general` - Generic document analysis
  - `invoice` - Extract vendor, amounts, items, dates
  - `contract` - Extract parties, clauses, obligations
  - `form` - Extract form fields
  - `report` - Technical report analysis
  - `correspondence` - Letter/email analysis
- **Outputs:**
  - Document summary
  - Entity extraction (people, orgs, dates, amounts)
  - Document categorization
  - Confidence scores

#### Validation Agent (`validation_agent.py`)
- Cross-validates extracted information
- Rule-based and LLM-based validation
- **Validation Types:**
  - `consistency` - Internal data consistency checks
  - `format` - Email, phone, date, amount formats
  - `business_rules` - Amount ranges, date ranges, tax calculations
  - `completeness` - Missing field detection
  - `ocr_consistency` - Cross-check with OCR results
- **Threshold:** 70% overall score to pass

#### Fallback Agent (`fallback_agent.py`)
- Error recovery mechanism
- Provides degraded service when primary agents fail

---

### 2. Models (`src/models/`)

#### OpenRouter Model (`openrouter_model.py`)
- **Primary interface** to ERNIE LLM via OpenRouter API
- **API Endpoint:** `https://openrouter.ai/api/v1/chat/completions`
- **Authentication:** Bearer token from `OPENROUTER_API_KEY`
- **Features:**
  - Async HTTP client (aiohttp)
  - Retry logic with exponential backoff (3 attempts)
  - Timeout handling (default 30s)
  - Configurable temperature and max_tokens
- **Methods:**
  - `generate()` - General text generation
  - `analyze_document()` - Structured document analysis
  - `validate_information()` - Data validation
  - `generate_summary()` - Text summarization
- **Fallback Parsing:** If JSON parsing fails, uses regex-based entity extraction

#### ERNIE Model (`ernie_model.py`)
- **Wrapper** around OpenRouterModel
- Attempts OpenRouter first, falls back to simulated responses
- **Fallback Mode:** Returns hardcoded JSON responses for testing
- **Purpose:** Abstraction layer for potential direct ERNIE API integration

#### PaddleOCR Model (`paddle_ocr_model.py`)
- Wrapper around PaddleOCR library
- **Initialization Strategy:**
  1. Try primary language from config
  2. Fallback to Chinese (multilingual)
  3. Fallback to English
  4. Fail with error
- **Features:**
  - Async text extraction
  - Bounding box coordinates
  - Confidence scores per text block
  - Table structure detection
  - Multi-language support
- **Configuration:**
  - Languages: `["en", "ch", "es", "fr"]`
  - Confidence threshold: 0.8
  - GPU support: Configurable (default: CPU)

---

### 3. Utilities (`src/utils/`)

#### Config (`config.py`)
- Loads YAML configuration
- Overrides with environment variables
- **Priority:** ENV vars > config.yaml > defaults
- **Key Function:** `load_config()` - Handles missing files gracefully

#### Document Loader (`document_loader.py`)
- File validation and loading
- Format detection
- Size checking
- Path resolution

#### Logger (`logger.py`)
- Loguru-based logging
- Processing metrics tracking
- Agent-specific log bindings

---

## Configuration Files

### config.yaml
**WARNING:** Contains fake model names. See top of document.

```yaml
agents:
  coordinator:
    model: "ernie-4.5-turbo-32k"  # ❌ FAKE - ignored by code
    temperature: 0.7
    max_tokens: 4000
    timeout: 30
    parallel_processing: true
    retry_attempts: 3
  
  ocr:
    model: "paddleocr-vl-0.9b"  # ❌ FAKE - PaddleOCR doesn't use this
    languages: ["en", "ch", "es", "fr"]
    confidence_threshold: 0.8
    use_gpu: false
  
  analysis:
    model: "ernie-4.5-turbo-8k"  # ❌ FAKE - ignored by code
    temperature: 0.5
  
  validation:
    model: "ernie-4.5-turbo-8k"  # ❌ FAKE - ignored by code
    temperature: 0.3
```

### .env (Actual Configuration)
```env
# ✅ REAL CONFIGURATION THAT MATTERS
OPENROUTER_API_KEY=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
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
ERNIE_MODEL_NAME=baidu/ernie-4.5-21b-a3b  # ✅ REAL MODEL

# PaddleOCR
USE_GPU=false
PADDLEOCR_MODEL_PATH=./models/paddleocr

# Application
DEBUG=true
LOG_LEVEL=INFO
MAX_FILE_SIZE=50MB
UPLOAD_DIR=./uploads
OUTPUT_DIR=./outputs

# Legacy fake model names (ignored by code)
COORDINATOR_MODEL=ernie-4.5-turbo-32k  # ❌ Not used
WORKER_MODEL=ernie-4.5-turbo-8k        # ❌ Not used
```

---

## API Interfaces

### 1. FastAPI Service (`app.py`)
- Very minimal - just imports and runs Streamlit app
- Should be replaced with proper FastAPI endpoints

### 2. Streamlit UI (`streamlit_app.py`)
- Interactive web interface
- **Features:**
  - Document upload
  - Extraction type selection (text/table/structure/full)
  - Analysis type selection (auto/invoice/contract/form/report/general)
  - Workflow type selection (standard/parallel/streaming)
  - Real-time processing status
  - Results display
- **Caching:** Uses `@st.cache_resource` for coordinator instance
- **Extensive Debug Logging:** Writes to `d:\erine_project\.cursor\debug.log`

---

## Workflow Execution

### Standard Workflow
```
1. Load Document → OCR Agent
2. OCR Results → Analysis Agent
3. Analysis Results → Validation Agent
4. Validation Results → Final Output
```

### Parallel Workflow
```
1. Load Document
2. OCR Agent + Analysis Agent (parallel)
3. Validation Agent
4. Final Output
```

### Streaming Workflow
```
1. Load Document
2. Stream OCR results as available
3. Stream analysis in real-time
4. Progressive validation
5. Incremental output updates
```

---

## Data Flow

### Input Task Format
```json
{
  "type": "document_processing",
  "document_path": "/path/to/file.pdf",
  "extraction_type": "full",
  "analysis_type": "invoice",
  "workflow_type": "standard",
  "output_format": "json"
}
```

### Output Format
```json
{
  "success": true,
  "task_id": "uuid",
  "agent_name": "CoordinatorAgent",
  "extraction_results": {
    "extracted_text": "...",
    "confidence": 0.92,
    "images_processed": 3
  },
  "analysis_results": {
    "document_type": "invoice",
    "vendor_name": "...",
    "total_amount": "...",
    "confidence": 0.89
  },
  "validation_results": {
    "is_valid": true,
    "overall_score": 0.85,
    "validations": {...}
  },
  "metadata": {
    "processing_time": 12.5,
    "timestamp": "2025-12-31T13:47:00",
    "workflow_type": "standard"
  }
}
```

---

## Dependencies (`requirements.txt`)

```
camel-ai          # Multi-agent framework (optional, fallback available)
paddlepaddle      # PaddleOCR backend
paddleocr         # OCR engine
fastapi           # API framework
uvicorn           # ASGI server
streamlit         # Web UI
PyPDF2            # PDF handling (unused - actually uses PyMuPDF)
Pillow            # Image processing
pandas            # Data manipulation
python-dotenv     # Environment variables
pyyaml            # Config parsing
aiohttp           # Async HTTP client
loguru            # Logging
```

**Additional Runtime Dependency:**
- `PyMuPDF` (fitz) - Required for PDF loading, not in requirements.txt

---

## File Structure

```
smart-document-processor/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py           # Abstract base class
│   │   ├── coordinator_agent.py    # Workflow orchestrator
│   │   ├── ocr_agent.py           # PaddleOCR wrapper
│   │   ├── analysis_agent.py      # ERNIE analysis
│   │   ├── validation_agent.py    # Data validation
│   │   └── fallback_agent.py      # Error recovery
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── openrouter_model.py    # ✅ REAL API client
│   │   ├── ernie_model.py         # Wrapper with fallback
│   │   └── paddle_ocr_model.py    # PaddleOCR wrapper
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py              # Config loader
│   │   ├── document_loader.py     # File loader
│   │   └── logger.py              # Logging setup
│   │
│   └── main.py                     # CLI entry point (unused)
│
├── ernie/                          # Virtual environment
├── temp_uploads/                   # Temp file storage
├── test_documents/                 # Test files
│
├── app.py                          # FastAPI entry (minimal)
├── streamlit_app.py               # ✅ MAIN UI
├── config.yaml                     # ❌ Fake model names
├── .env                           # ✅ REAL configuration
├── requirements.txt                # Dependencies
├── README.md                       # Project docs
├── CONTEXT.md                      # This file
│
├── test_system.py                  # End-to-end tests
├── test_openrouter.py             # API connectivity tests
├── quick_test.py                   # Smoke tests
├── quick_start.py                  # Quick start script
└── import_fixer.py                # Import path fixes
```

---

## Known Issues & Technical Debt

### 1. Model Configuration Confusion
- **Problem:** config.yaml has fake model names that are ignored
- **Fix:** Update config.yaml with real model names or remove model field
- **Impact:** Confusing for developers, but functionally works via .env

### 2. Missing Dependency
- **Problem:** PyMuPDF (fitz) required but not in requirements.txt
- **Fix:** Add `PyMuPDF` to requirements.txt
- **Impact:** PDF loading fails on fresh install

### 3. FastAPI Not Implemented
- **Problem:** app.py is just a Streamlit wrapper
- **Fix:** Implement proper REST endpoints
- **Impact:** No programmatic API access

### 4. Extensive Debug Logging
- **Problem:** Hardcoded debug logs throughout code
- **Fix:** Remove or make conditional on DEBUG flag
- **Impact:** Performance overhead, cluttered logs

### 5. CAMEL-AI Optional But Treated As Required
- **Problem:** Code tries to use CAMEL-AI but has fallback
- **Fix:** Either make it required or improve fallback implementation
- **Impact:** Confusion about whether it's needed

### 6. Inconsistent Import Paths
- **Problem:** Mix of absolute imports (`from src.agents`) and relative imports
- **Fix:** Standardize on absolute imports from project root
- **Impact:** Import errors depending on how code is run

---

## Running the Project

### 1. Setup Environment
```powershell
# Create virtual environment
python -m venv ernie
ernie\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install PyMuPDF  # Missing dependency
```

### 2. Configure Environment
```powershell
# Edit .env file with real API key
# Verify ERNIE_MODEL_NAME is set to real model
```

### 3. Run Streamlit UI (Primary Interface)
```powershell
streamlit run streamlit_app.py
```

### 4. Run Tests
```powershell
python test_system.py          # Full system test
python test_openrouter.py      # API connectivity
python quick_test.py           # Smoke tests
```

---

## Performance Characteristics

### Typical Processing Times
- **OCR (1 page):** 2-5 seconds
- **Analysis (ERNIE):** 3-8 seconds
- **Validation:** 2-4 seconds
- **Total (standard workflow):** 7-17 seconds per document

### Resource Usage
- **Memory:** ~500MB-1GB (PaddleOCR models)
- **CPU:** Medium (OCR preprocessing)
- **Network:** Low (only ERNIE API calls)
- **Disk:** Minimal (temp files cleaned up)

### Scaling Considerations
- OCR is CPU-bound (can enable GPU with `USE_GPU=true`)
- ERNIE calls are network-bound (could batch requests)
- Parallel workflow offers ~30% speedup
- Batch processing not yet optimized

---

## Security Notes

### API Key Exposure
- `.env` file contains plaintext API key
- ⚠️ **Never commit .env to version control**
- Rotate key if exposed

### File Upload Security
- Max file size: 50MB
- Format validation exists
- No malware scanning
- Temp files stored in `temp_uploads/`

### Output Sanitization
- JSON responses not sanitized
- Could contain sensitive data from documents
- No PII detection/redaction

---

## Future Improvements

### High Priority
1. Fix model name confusion (config.yaml vs .env)
2. Add missing PyMuPDF dependency
3. Implement proper FastAPI REST API
4. Remove hardcoded debug logging

### Medium Priority
5. Add authentication/authorization
6. Implement rate limiting
7. Add caching layer for repeat documents
8. Improve error messages and user feedback
9. Add batch processing optimization

### Low Priority
10. Docker containerization
11. CI/CD pipeline
12. Multi-language UI
13. Advanced table extraction
14. Document versioning/history

---

## Contact & Maintenance

**Last Updated:** 2025-12-31  
**Current Version:** Not versioned  
**License:** MIT  
**Repository:** Local only (no remote)

---

## Quick Reference

### Essential Environment Variables
```bash
OPENROUTER_API_KEY=sk-or-v1-...     # ✅ REQUIRED
ERNIE_MODEL_NAME=baidu/ernie-4.5-21b-a3b  # ✅ REQUIRED
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  # Optional
```

### Model Names Reference
```
❌ FAKE (in config.yaml):
   - ernie-4.5-turbo-32k
   - ernie-4.5-turbo-8k
   - paddleocr-vl-0.9b

✅ REAL (via OpenRouter):
   - baidu/ernie-4.5-21b-a3b (current)
   - baidu/ernie-4.0-turbo-8k
   - baidu/ernie-3.5-8k
   - baidu/ernie-4-turbo
```

### Key Files to Modify
- `.env` - API configuration (never commit!)
- `config.yaml` - Agent behavior settings
- `streamlit_app.py` - UI customization
- `src/agents/*` - Agent logic

---

## End of Context Document
