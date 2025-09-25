import os
from pathlib import Path

# App Configuration
APP_TITLE = "FinComBot - Compliance Assistant"
APP_ICON = "🏦"
VERSION = "1.0.0"

# File paths
DATA_DIR = Path("Data")
EMBEDDINGS_FILE = DATA_DIR / "SEC5_embeddings.pkl"
FAISS_INDEX_FILE = DATA_DIR / "SEC5_faiss.index"
CLEANED_TEXT_FILE = DATA_DIR / "SEC5 - OPENING OF ACCOUNTS (004)_cleaned.txt"

# Model Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Search Configuration
DEFAULT_K = 5
MAX_K = 10
DEFAULT_MIN_SIMILARITY = 0.1

# UI Configuration
MAX_CHUNK_PREVIEW = 400
MAX_SEARCH_HISTORY = 50

# Sample Queries for User Guidance
SAMPLE_QUERIES = [
    "What documents are required to open a church account?",
    "How do we verify foreign national customers?",
    "What are the KYC requirements for minors?",
    "What documents are needed for CDD?",
    "How to handle suspicious transactions?",
    "What are the account opening requirements for companies?",
    "What is the process for Enhanced Due Diligence?",
    "How do we verify scrap metal dealers?",
    "What are the requirements for joint accounts?",
    "How to process account closure requests?"
]

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Security (for future deployment)
ALLOWED_USERS = []  # Empty means all users allowed
SESSION_TIMEOUT = 3600  # 1 hour in seconds