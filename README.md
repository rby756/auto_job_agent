# README.md

## 🧠 Project: AI Agents for Auto Job Application

This repository implements an **end-to-end AI system** that automatically applies to jobs by parsing job descriptions (JDs), matching resumes using RAG + LLM, and submitting applications via browser automation. The project is built with an MLOps-first architecture and supports monitoring, feedback loops, and self-hosting.

---

## 🛠️ Use Cases

### 1. 🔍 Job Matching & Scoring
- Automatically matches job descriptions with resumes
- Uses RAG (Retrieval Augmented Generation) and LLMs for scoring and explanations

### 2. 🤖 Auto Application Submission
- Fills out job application forms from predefined templates
- Supports automation via Selenium/Playwright

### 3. 🧠 AI Agent Orchestration
- Runs multi-agent pipeline using LangGraph-style planner
- Modular decision logic for "apply" or "skip" with human-in-the-loop

### 4. 📊 Monitoring + Logging
- Logs token usage, latency, and errors from LLMs
- Tracks success/failure of agent decisions and API routes

### 5. 🧪 Resume Screening SaaS
- Can be adapted for internal HR teams or startups to automate screening
- Easily integrable via FastAPI

### 6. 🧰 Resume Insights & Improvement
- Gives candidates actionable insights on why a resume didn't match
- Optionally links to external resume improvement tools

---

## 🏗️ Architecture

```
Input JD + Resume
      │
      ▼
[JD Parser] ─▶ [RAG Resume Matcher] ─▶ [LangGraph Planner] ─▶ [Form Auto-Filler]
                                  │
                        [Streamlit UI / FastAPI API]
                                  ▼
                         Monitoring & Feedback Loop
```

---

## 📁 Key Directories

| Folder                  | Description |
|-------------------------|-------------|
| `auto_job_apply/`       | Core package containing agents, backend, RAG, config, monitoring |
| `api_app/`              | FastAPI entry point |
| `ui/`                   | Streamlit UI and feedback components |
| `data/`                 | Sample resumes, JDs, form templates |
| `monitoring/`           | MLflow, LLM token logger, Prometheus metrics |
| `infra/`                | Grafana and Prometheus setup |
| `deploy/`               | Docker Compose and Nginx config for deployment |
| `tests/`                | Unit tests |
| `notebooks/`            | Experiments and visualizations |
| `pipelines/`            | ML training and deployment pipelines |

---

## ⚙️ Setup Instructions

```bash
# 1. Clone the repo
$ git clone https://github.com/your-username/auto-job-agents.git && cd auto-job-agents

# 2. Create environment
$ python3 -m venv venv && source venv/bin/activate

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Configure
$ cp .env.example .env && edit values (LLM API keys, DB paths)

# 5. Run locally
$ streamlit run ui/streamlit_app.py
# or
$ uvicorn api_app.main:app --reload
```

---

## 🐳 Docker & Deployment

```bash
# Build and run the full stack
$ docker-compose up --build
```

- Access UI: `http://localhost:8501`
- Access API: `http://localhost:8000/docs`
- Grafana: `http://localhost:3000`

---

## 📊 Monitoring

- MLflow for evaluation tracking
- Prometheus for metrics
- Grafana for dashboard visualization
- LLM latency/token usage logging

---

## 🔐 Authentication

- Basic Auth/JWT for all API routes
- Define credentials in `.env`

---

## ✅ Features

- LangGraph-style multi-agent orchestration
- JD and resume matching with RAG + LLM
- Form auto-fill via browser automation
- Human-in-the-loop approval before apply
- CI/CD pipeline and Docker-ready
- Observability and real-time logging

---

## 🧪 Tests

```bash
pytest tests/
```

---

## 🧠 Author

- Reby Varghese (AI Engineer)
- [GitHub](https://github.com/rby756) | [LinkedIn](https://linkedin.com/in/reby-varghese-0ab68218b)

---

## 📄 License

This project is open-source and available under the MIT License.
