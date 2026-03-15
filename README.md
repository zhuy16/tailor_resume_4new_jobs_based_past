# tailor_resume_4new_jobs_based_past

![version](https://img.shields.io/badge/version-0.1.0-blue)
![python](https://img.shields.io/badge/python-3.11+-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A local RAG pipeline that tailors your resume for new job applications by learning from your own past applications.

**Core idea:** your best reference for any new job is a resume you already wrote for something similar. This tool finds that match automatically, then uses Claude to rewrite it for the new role.

---

## How it works

```
Past applications (PDFs)  ──►  ChromaDB (local vector store)
                                       │
New job description PDF   ──►  similarity search  ──►  top 3 matches
                                       │
                               matched past resume (DOCX)
                                       │
                               Claude rewrites it  ──►  tailored_<job>.docx
```

1. **Scan** your resume archive → classify every file, build reference CSVs
2. **Embed** all past job description PDFs into a local ChromaDB
3. **Drop** a new JD PDF into `new_jobs/` and run `tailor.py` → get a tailored DOCX in seconds

---

## Quickstart

```bash
# 1. Install
conda create -n job-rag python=3.11 && conda activate job-rag
pip install -r requirements.txt

# 2. Configure
cp .env.example .env          # add your Anthropic API key
# edit config.py → set RESUME_ROOT to your resume archive path

# 3. Build the database (once)
python find_resumes.py
python build_db.py

# 4. Tailor a resume — drop a JD PDF into new_jobs/, then:
python tailor.py
```

Output: `new_jobs/tailored_<job-name>.docx`

📖 **[Full setup & usage guide →](docs/setup_and_usage.md)**

---

## Tech stack

- [ChromaDB](https://www.trychroma.com/) — local vector store (no server needed)
- [Anthropic Claude](https://www.anthropic.com/) — resume rewriting
- [python-docx](https://python-docx.readthedocs.io/) — DOCX generation
- [PyPDF2](https://pypdf2.readthedocs.io/) — PDF text extraction

---

## Project structure

```
.
├── config.py              paths and settings (no secrets)
├── find_resumes.py        scan RESUME_ROOT, classify files, write data/ CSVs
├── build_db.py            embed job description PDFs into ChromaDB
├── tailor.py              RAG query + Claude rewrite → DOCX
├── requirements.txt
├── .env.example           copy to .env, add API key
├── new_jobs/              drop new JD PDFs here; tailored DOCXs saved here
├── data/                  generated CSVs (gitignored — personal paths)
├── db/                    ChromaDB vector store (gitignored)
├── docs/
│   └── setup_and_usage.md full guide
└── analysis/              optional embedding visualisation scripts
```

---

## Privacy

All data stays local. The only external call is to the Anthropic API (JD + resume text only). Personal CSVs, the vector DB, and all resume files are gitignored.

