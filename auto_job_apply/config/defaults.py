# auto_job_apply/config/defaults.py
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"

# OCR Configuration
OCR_CONFIG = {
    "tesseract": {
        "cmd": "/usr/bin/tesseract",  # Default Linux path
        "lang": "eng",
        "oem": 3,  # LSTM OCR Engine
        "psm": 6,  # Assume a single uniform block of text
        "config": "--oem 3 --psm 6 -c preserve_interword_spaces=1"
    },
    "preprocessing": {
        "dpi": 300,
        "blur_kernel": (3, 3),
        "adaptive_thresh": {
            "max_value": 255,
            "method": "gaussian",
            "block_size": 11,
            "c": 2
        },
        "morphology": {
            "kernel_size": (2, 2),
            "iterations": 1
        }
    },
    "temp": {
        "dir": str(DATA_DIR / "temp"),
        "prefix": "ocr_temp_",
        "suffix": ".png",
        "keep_temp_files": False
    }
}

# RAG Configuration
RAG_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "vector_db": {
        "path": str(DATA_DIR / "vector_store"),
        "index_name": "job_descriptions"
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(DATA_DIR / "logs" / "app.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "standard",
            "level": "DEBUG"
        }
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
}

# API Keys (should be loaded from environment variables in production)
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "huggingface": os.getenv("HF_API_KEY", ""),
    "aws": {
        "access_key_id": os.getenv("AWS_ACCESS_KEY_ID", ""),
        "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        "region": os.getenv("AWS_DEFAULT_REGION", "us-west-2")
    }
}

# File Paths
FILE_PATHS = {
    "resumes": {
        "input": str(DATA_DIR / "resumes"),
        "processed": str(DATA_DIR / "processed_resumes"),
        "output": str(DATA_DIR / "output")
    },
    "jobs": {
        "input": str(DATA_DIR / "job_descriptions"),
        "processed": str(DATA_DIR / "processed_jobs")
    },
    "models": {
        "embedding": str(DATA_DIR / "models" / "embedding"),
        "classifier": str(DATA_DIR / "models" / "classifier")
    }
}

# Create necessary directories
for path_group in FILE_PATHS.values():
    for path in path_group.values():
        os.makedirs(path, exist_ok=True)
os.makedirs(OCR_CONFIG["temp"]["dir"], exist_ok=True)