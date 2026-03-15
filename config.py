"""Paths and settings for job-rag. Secrets from .env only."""
import os
from dotenv import load_dotenv

load_dotenv()

# Resume root: parent of project dir + "resume"
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESUME_ROOT = os.path.abspath(os.path.join(_PROJECT_DIR, "..", "resume"))
DB_PATH = os.path.join(_PROJECT_DIR, "db")
NEW_JD_DIR = os.path.join(_PROJECT_DIR, "new_jobs")  # drop new JD PDFs here
RESUME_INDEX_PATH = os.path.join(_PROJECT_DIR, "resume_index.json")
# Reference table of all PDFs + classification (path, folder, likely_type)
RESUME_REFERENCE_PATH = os.path.join(_PROJECT_DIR, "data", "resume_reference.csv")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
