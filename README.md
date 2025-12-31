# Smart Document Processor

A **multi-agent document processing pipeline** that combines **OCR (PaddleOCR)** with **ERNIE-based LLM analysis** via **OpenRouter**, exposed through both **CLI tools** and **service endpoints** (**FastAPI** and **Streamlit**).

---

## 🚀 Features

* 📄 **OCR with PaddleOCR** for PDFs and images
* 🧠 **ERNIE (via OpenRouter)** for document analysis and validation
* ⚙️ **FastAPI service** for document submission, processing, and health checks
* 🖥️ **Streamlit UI** for interactive demos
* 🤖 **Modular multi-agent architecture**:

  * OCR Agent
  * Analysis Agent
  * Validation Agent
  * Coordinator Agent
  * Fallback Agent
* 🔧 **Configurable setup** via `config.yaml` and environment variables

---

## 🏗️ Architecture Overview

```text
User / Client
     │
     ▼
FastAPI / Streamlit
     │
     ▼
Coordinator Agent
 ┌───────┬──────────┬───────────┐
 ▼       ▼          ▼           ▼
OCR   Analysis   Validation   Fallback
Agent   Agent       Agent       Agent
     │
     ▼
Structured Output / JSON
```

---

## ⚡ Quick Start

### 1️⃣ Create & Activate Virtual Environment (Windows PowerShell)

```powershell
python -m venv ernie
ernie\Scripts\Activate.ps1
```

---

### 2️⃣ Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3️⃣ Environment Variables

Create a `.env` file (**do not commit this file**):

```env
OPENROUTER_API_KEY=your-api-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
ERNIE_MODEL_NAME=baidu/ernie-4.0-turbo-8k
```

---

### 4️⃣ Run FastAPI Service

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Once running:

* API Docs: `http://localhost:8000/docs`
* Health Check: `http://localhost:8000/health`

---

### 5️⃣ Run Tests

```powershell
python test_system.py
python test_openrouter.py
```

---

### 6️⃣ Run Streamlit Demo

```powershell
streamlit run streamlit_app.py
```

---

## 📁 Project Structure

```text
.
├── src/
│   ├── agents/          # OCR, Analysis, Validation, Coordinator, Fallback agents
│   ├── models/          # PaddleOCR and OpenRouter / ERNIE wrappers
│   ├── utils/           # Config loading, logging, document utilities
│
├── app.py               # FastAPI entrypoint
├── streamlit_app.py     # Streamlit UI demo
├── config.yaml          # Default configuration
├── requirements.txt     # Pinned dependencies
│
├── test_openrouter.py   # OpenRouter connectivity tests
├── test_system.py       # End-to-end system tests
├── quick_test.py        # Lightweight smoke tests
└── README.md            # Project documentation
```

---

## ⚙️ Configuration

* **`config.yaml`**

  * Holds default system configuration
  * Loaded by `src/utils/config.py`

* **Environment Variables**

  * Override values in `config.yaml`
  * Required for OpenRouter authentication

> ⚠️ **Never commit `.env` files** — keep credentials secure.

---

## 📦 Dependency Notes

* **PaddleOCR / PaddlePaddle**

  * Versions are pinned in `requirements.txt`
  * Ensure your **Python version and OS** are supported

* **camel-ai**

  * Optional dependency
  * May increase installation time
  * Safe to remove if not required

After installation, verify dependency health:

```powershell
pip check
```

---

## 🤝 Contributing

1. Create a feature branch
2. Make changes with clear commits
3. Add or update tests where applicable
4. Ensure `.env` remains untracked
5. Run tests and `pip check` before submitting

Pull requests are welcome 🚀

---

## 📜 License

MIT License

---

## 🧩 Future Improvements (Optional)

* Async agent orchestration
* Document-level caching
* Multi-language OCR + LLM routing
* Authentication & rate limiting
* Docker & CI/CD pipeline

---

**Built for scalable, production-ready document intelligence.**
