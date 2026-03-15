I am building a RAG-based resume tailoring pipeline called job-rag. Here is the full context:
Goal
Automatically tailor resumes for new job applications by finding the most similar past job I've applied to, retrieving the resume I used for that job, and using Claude API to identify what changes are needed for the new role.
Project Structure
job-rag/
├── .cursorrules
├── .env                    # API keys, gitignored
├── .env.example            # template, committed to git
├── .gitignore
├── README.md
├── requirements.txt
├── config.py               # all paths and settings
├── find_resumes.py         # DONE - scans resume folders
├── build_db.py             # TO BUILD - embeds PDFs into ChromaDB
├── tailor.py               # TO BUILD - query + Claude tailoring
└── db/                     # ChromaDB stores here, gitignored
Resume Folder Structure
~/resume/                   # outcome = "applied"
├── 0_slow/                 # outcome = "slow_response"
├── rejected/               # outcome = "rejected"
├── interviewed/            # outcome = "interviewed"
└── interviewing/           # outcome = "interviewing"
File Naming Pattern
Resume PDFs follow this pattern:
{4-6 digits}.?{optional digits}_{CompanyName}.pdf
Examples: 123456_BioNTech.pdf, 1234.5_GSK.pdf
The job_id is extracted from the filename prefix e.g. 123456_BioNTech
find_resumes.py (ALREADY BUILT)

Scans all 5 folders
Matches files using regex pattern
Extracts job_id from filename
Infers outcome from folder name
Outputs resume_index.json with fields:

job_id — e.g. "123456_BioNTech"
filename — e.g. "123456_BioNTech.pdf"
path — absolute path to file
folder — which subfolder
outcome — applied/rejected/interviewed etc.
extension — .pdf or .docx



build_db.py (TO BUILD NEXT)
Should do the following:

Read resume_index.json
For each entry, extract text from the PDF using pypdf2
Embed the text using ChromaDB's default embedding (sentence-transformers)
Store in persistent ChromaDB at ./db with metadata:

job_id
path
outcome
folder
company (extracted from job_id after underscore)


Skip files already in the database (incremental updates)
Print progress as it runs

tailor.py (TO BUILD AFTER)
Should do the following:

Accept a new job description as input (paste as string or read from file)
Embed the new job description
Query ChromaDB for top 3 most similar past jobs
Display the matches with similarity scores
Take the top match, retrieve its resume path
Read the matched resume text
Call Claude API (claude-sonnet-4-20250514) with:

The matched past job description
The matched resume
The new job description
Instructions to return JSON: {"add": [], "remove": [], "modify": []}


Print the suggested changes clearly

config.py
pythonimport os
from dotenv import load_dotenv
load_dotenv()

RESUME_ROOT = os.path.expanduser("~/resume")
DB_PATH = "./db"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
```

## .env.example
```
ANTHROPIC_API_KEY=your-key-here
```

## .gitignore
```
.env
db/
__pycache__/
venv/
*.pyc
resume_index.json
```

## requirements.txt
```
chromadb
sentence-transformers
anthropic
pypdf2
python-docx
python-dotenv
Tech Stack

Python 3.11, conda environment named job-rag
ChromaDB — local persistent vector database
sentence-transformers — default embedding model (all-MiniLM-L6-v2)
Anthropic Claude API — for resume diff and tailoring suggestions
pypdf2 — PDF text extraction

Rules

Never hardcode paths or API keys
Always use config.py for settings
Always use .env for secrets
Keep code simple and readable
Print progress so I can see what's happening
Handle missing files gracefully with try/except

Current Status

conda environment job-rag created and activated
packages installed
find_resumes.py is ready
Please now build build_db.py first, then tailor.py