# Detailed Setup & Usage Guide

## Requirements

- Python 3.11+
- Conda (or any virtualenv)
- An [Anthropic API key](https://console.anthropic.com/)
- Your resume archive folder (see structure below)

---

## Installation

```bash
conda create -n job-rag python=3.11
conda activate job-rag
pip install -r requirements.txt
```

---

## Configuration

### 1. API key
```bash
cp .env.example .env
# Open .env and set:
# ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Resume root path
Open `config.py` and set `RESUME_ROOT` to the absolute path of your resume archive:

```python
RESUME_ROOT = "/Users/yourname/Documents/resume"
```

### Resume folder structure expected
Each job application lives in its own subfolder containing at minimum the job
description PDF and the resume DOCX/PDF you submitted.

```
resume/
├── CompanyA_2025/
│   ├── CompanyA_job_description.pdf
│   └── resume_CompanyA.docx
├── 0_slow/                   # "slow response" outcome bucket
│   └── CompanyB_2024/
├── rejected/
├── interviewed/
└── interviewing/
```

Outcome is inferred from the top-level bucket folder name:
| Folder | Outcome |
|---|---|
| (root) | applied |
| `0_slow/` | slow\_response |
| `rejected/` | rejected |
| `interviewed/` | interviewed |
| `interviewing/` | interviewing |

---

## One-time database build

Run these two steps once (and again whenever you add new past applications):

```bash
# Step 1 – scan and classify every file in RESUME_ROOT
python find_resumes.py
# Outputs: data/resume_reference.csv, data/job_descriptions_pdf_list.csv, etc.

# Step 2 – embed all job description PDFs into ChromaDB
python build_db.py
# Outputs: db/  (local ChromaDB vector store)
```

`build_db.py` is incremental — re-running it only embeds new files, so it is
safe to run after each new application.

---

## Tailoring a resume for a new job

### Quickest path
1. Save the new job description as a PDF
2. Drop it into `new_jobs/`
3. Run:

```bash
conda run -n job-rag python tailor.py
```

### What happens
| Step | Detail |
|---|---|
| 1 | Picks the newest PDF in `new_jobs/` |
| 2 | Embeds it and queries ChromaDB for the 3 most similar past jobs |
| 3 | Prints similarity scores and outcomes for each match |
| 4 | Finds the resume (prefers DOCX) you used for the top match |
| 5 | Calls Claude to rewrite a complete tailored resume |
| 6 | Saves `new_jobs/tailored_<jd-name>.docx` — **review and edit before sending** |

### CLI options
```bash
# Point at a specific JD file instead of auto-detecting
python tailor.py --jd path/to/job.pdf

# Show top 5 similar past jobs instead of 3
python tailor.py --top-k 5
```

### Multiple JDs at once
Drop multiple PDFs into `new_jobs/` — the script loops over all of them and
produces one DOCX per JD in a single run.

---

## Output DOCX format

Sections are always in this order:
**Name → Contact → SUMMARY → SKILLS → EXPERIENCE → EDUCATION**

| Style | Usage |
|---|---|
| `Heading 1` (14 pt bold blue + rule) | Name and ALL-CAPS section headers |
| `Heading 2` (10.5 pt bold) | `Title \| Company \| Start–End` |
| `List Bullet` | Achievement bullets (max 6/role, max 50 words each) |
| `Normal` | Contact info, date ranges |

`**text**` markers in Claude's output become real bold runs (metrics, skills).

Because the DOCX uses standard Word styles, you can paste individual sections
into your own template and they will adopt your template's formatting.

---

## Analysis scripts (optional)

`analysis/` contains standalone visualisation scripts (UMAP, PCA, TF-IDF) for
exploring your embedding space. They require additional packages beyond
`requirements.txt` (e.g. `umap-learn`, `matplotlib`, `scikit-learn`).

---

## Privacy notes

- Everything stays local: your CSV indexes, ChromaDB, and resume files are all
  gitignored and never leave your machine.
- The only external call is to the Anthropic API, which receives the text of
  the JDs and resumes being compared. No filenames or personal metadata are sent.
