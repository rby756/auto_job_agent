# Auto Job Agent

## 🧠 Project: AI-Powered Resume Matcher & Job Application Assistant

This repository implements an **intelligent job matching system** that processes resumes using OCR, matches them with job descriptions using RAG (Retrieval-Augmented Generation), and provides a user-friendly interface for job recommendations. The system is built with a modular architecture supporting both API and web interfaces.

---

## 🚀 Key Features

### 1. 📄 Resume Processing
- **OCR Extraction**: Automatically extracts text from PDF and image resumes
- **Structured Parsing**: Identifies key sections like skills, experience, and education

### 2. 🔍 Intelligent Job Matching
- **Semantic Search**: Uses RAG (Retrieval-Augmented Generation) for accurate job matching
- **Similarity Scoring**: Ranks jobs based on relevance to resume content
- **Configurable Thresholds**: Set minimum match scores for recommendations

### 3. 🖥️ User Interfaces
- **Streamlit Web App**: Interactive dashboard for resume upload and job recommendations
- **RESTful API**: Programmatic access to all features
- **Responsive Design**: Works on desktop and mobile devices

### 4. 🛠️ Developer Friendly
- **Modular Architecture**: Easy to extend and customize
- **Type Hints**: Full Python type support
- **Comprehensive Logging**: Built-in logging for debugging and monitoring

### 5. 📊 Data Management
- **Vector Store**: FAISS-based storage for efficient similarity search
- **File Processing**: Handles PDFs, DOCX, and image formats
- **Configurable Storage**: Customize storage locations for processed files

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌───────────────────────┐
│  Resume Upload  │ ──> │  OCR Processing  │ ──> │  Text Extraction &    │
│  (PDF/Image)    │     │  (Tesseract)     │     │  Preprocessing       │
└─────────────────┘     └─────────────────┘     └───────────┬───────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌─────────────────┐     ┌───────────────────────┐
│  Job            │ ──> │  Vector         │ <── │  Text Embedding       │
│  Descriptions   │     │  Store (FAISS)  │     │  (BGE Model)          │
└─────────────────┘     └─────────────────┘     └───────────┬───────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌─────────────────┐     ┌───────────────────────┐
│  User           │ <── │  Streamlit UI   │ <── │  Similarity Search    │
│  Interaction    │     │  & API          │     │  & Ranking            │
└─────────────────┘     └─────────────────┘     └───────────────────────┘
```

---

## 📁 Project Structure

```
auto_job_agent/
├── api_app/                 # FastAPI application
│   ├── main.py              # FastAPI app entry point
│   └── routers/             # API route handlers
│       ├── jobs.py          # Job matching endpoints
│       └── ocr.py           # OCR processing endpoints
├── auto_job_apply/          # Core application code
│   ├── ocr/                 # OCR processing components
│   └── rag/                 # RAG pipeline components
├── data/
│   ├── processed_resumes/   # Extracted text from resumes
│   ├── resumes/             # Sample resume files
│   ├── sample_jds/          # Sample job descriptions
│   └── vector_store/        # FAISS index and metadata
├── tests/                   # Test files
└── streamlit_app.py         # Streamlit web interface
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- Tesseract OCR (`sudo apt install tesseract-ocr` on Ubuntu/Debian)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rby756/auto_job_agent.git
   cd auto_job_agent
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv_job_agent
   source venv_job_agent/bin/activate  # On Windows: venv_job_agent\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements-api.txt
   pip install -r requirements-rag.txt
   pip install streamlit
   ```

4. **Start the API server**
   ```bash
   uvicorn api_app.main:app --reload
   ```

5. **Start the Streamlit UI** (in a new terminal)
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Access the application**
   - Web UI: http://localhost:8501
   - API Docs: http://localhost:8000/docs

---

## 🐳 Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access services**
   - Streamlit UI: http://localhost:8501
   - FastAPI Docs: http://localhost:8000/docs
   - API Base URL: http://localhost:8000

3. **Stop the services**
   ```bash
   docker-compose down
   ```

---

## 🚀 API Endpoints

### OCR Processing
- `POST /api/ocr/process` - Process a resume file (PDF/Image)
  ```bash
  curl -X POST -F "file=@resume.pdf" "http://localhost:8000/api/ocr/process?lang=eng"
  ```

### Job Matching
- `POST /api/jobs/match` - Find matching jobs for resume text
  ```json
  {
    "resume_text": "...",
    "top_k": 5,
    "score_threshold": 0.3
  }
  ```

### Job Ingestion
- `POST /api/jobs/ingest` - Add job descriptions to the vector store
  ```bash
  curl -X POST "http://localhost:8000/api/jobs/ingest?directory=/path/to/job/descriptions"
  ```

## 🛠️ Configuration

Configure the application using environment variables:

```env
# API Settings
HOST=0.0.0.0
PORT=8000

# File Paths
PROCESSED_RESUMES_DIR=./data/processed_resumes
VECTOR_STORE_PATH=./data/vector_store/job_descriptions.pkl

# OCR Settings
OCR_LANG=eng
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_rag.py -v

# Run with coverage report
pytest --cov=auto_job_apply tests/
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text extraction
- [Sentence Transformers](https://www.sbert.net/) for text embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Streamlit](https://streamlit.io/) for the web interface
- [FastAPI](https://fastapi.tiangolo.com/) for the API server
