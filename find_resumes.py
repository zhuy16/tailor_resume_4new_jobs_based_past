"""
Scan resume folder for all PDFs and DOCXs, classify each as job_description / resume / cover_letter / combined,
and save a reference table (CSV) with path, job_folder, outcome, filename, extension, likely_type.
Run from project root. Uses config.py for RESUME_ROOT and RESUME_REFERENCE_PATH.
"""
import csv
import os
from pathlib import Path

from PyPDF2 import PdfReader
from docx import Document as DocxDocument

import config

# Outcome folders under RESUME_ROOT (first path component)
OUTCOME_FOLDERS = {"0_slow", "rejected", "interviewed", "interviewing"}
OUTCOME_DEFAULT = "applied"

# Filename (lowercase) keywords -> likely_type
FILENAME_HINTS = {
    "job_description": [
        "jd", "job_desc", "job description", "description", "posting", "position",
        "role_desc", "job_spec", "jobspec", "job_posting", "job ad", "jobad",
    ],
    "resume": [
        "resume", "cv", "cv_resume", "résumé", "bio", "_r_", "_resume",
    ],
    "cover_letter": [
        "cover", "cover_letter", "letter", "cl_", "_cl.", "cover letter",
    ],
    "combined": [
        "cover_resume", "resume_cover", "combined", "resume_and_cover", "application",
    ],
}

# Content (first ~3000 chars): phrases that suggest document type
CONTENT_JD = [
    "responsibilities", "qualifications", "requirements", "about the role",
    "we are looking for", "experience required", "applicant", "apply now",
    "job description", "position summary", "key responsibilities",
    "required qualifications", "preferred qualifications", "benefits",
]
CONTENT_COVER = [
    "dear hiring", "dear recruiter", "i am writing to apply", "i am excited to apply",
    "sincerely", "yours faithfully", "best regards", "thank you for considering",
]
CONTENT_RESUME = [
    "experience", "education", "skills", "summary", "objective",
    "work experience", "professional experience", "employment",
]

# Filenames that are not job descriptions (interview prep, recruiter notes, Gmail printouts, etc.)
NOT_JOB_DESCRIPTION_FILENAMES = {
    "🔹 Interview Outline.docx",
    "🔹 Why HHMI.docx",
    "🗓summary of recruiter talk.docx",
    "FinalRoundPreparation.docx",
    "JobTalk_AZ_V0.pdf",
    "JobTalk_AZ_V2.pdf",
    "SandboxAQ Job Openings.pdf",
    "Miltenyi_job_presentation_V1.pdf",
    "Acknowledged.docx",
    "What You.docx",
    "recruiter_interview.docx",
    "week0125_Process Weekly Certification.pdf",
    "research_notes.pdf",
    "SamStartUp_Smart Questions to Ask Sam.docx",
    "Suggested Jobs _ ZipRecruiter.pdf",
    "Excellent.docx",
    "🧭 1_for_recruiter.docx",
    "250725_softwareEngineering_best_practices.pdf",
    "AI Agents_ How to Build Autonomous Workflows _ Encord.pdf",
}

# Filenames that are job descriptions (override regardless of current classification)
FORCE_JOB_DESCRIPTION_FILENAMES = {
    "Iambic Therapeutics_ Insights _ LinkedIn.pdf",
    "Senior Scientist in Rockville, Maryland, United States at EMD Group.pdf",
    "Tampus_BioinformaticsScientistCareers.pdf",
    "Incyte_Associate Director, Computational Biology and Data Science, Translational Medicine in Wilmington, Delaware _ Incyte Corporation.pdf",
    # Manually reclassified 2026-03-10 (were in pdf_resumes_list, clearly job postings)
    "19493 KK Sr. Biomedical Data Scientist BAMF (1) (1).pdf",
    "Output Biosciences.pdf",
    "CAREERS AT NVIDIA.pdf",
    "Director, Next-Generation Sequencing for Biologics Engineering at AstraZeneca.pdf",
    "Job Detail - Veracyte.pdf",
    "Sr Computational Biologist - Boston, Massachusetts, United States.pdf",
    "Computational Biologist II at Immunai.pdf",
    "Data Scientist, Computational Biology - Silver Spring, MD - Indeed.com.pdf",
    "Eli_lily.pdf",
    "SandboxAQ_BioinformaticResearchEngineer.pdf",
    "DataEngineer-bioinformatics.pdf",
    "Bioinformatics Scientist.pdf",
    "Sr. Bioinformatics Scientist at Guardant Health in Palo Alto, California REF5877X.pdf",
    "Senior Scientist \u2013 Research Computational Biology (ARIA) Jobs at Amgen in United States - Remote.pdf",
    "Openings \u2014 Verge Genomics.pdf",
    "Job Openings - GeneDx\u00ae.pdf",
    "myGwork \u00a6 LGBTQ+ Jobs \u00a6 Senior Principal Scientist, Biological Network Analytics \U0001f3f3\ufe0f_\U0001f308.pdf",
    "Senior Research Scientist.pdf",
    "Principal AI Researcher at Synthesize Bio.pdf",
    "Computational Biologics Design Senior Scientist at AstraZeneca.pdf",
    # Manually reclassified 2026-03-10 (were in pdf_resumes_list, clearly job postings)
    "19493 KK Sr. Biomedical Data Scientist BAMF (1) (1).pdf",
    "Output Biosciences.pdf",
    "CAREERS AT NVIDIA.pdf",
    "Director, Next-Generation Sequencing for Biologics Engineering at AstraZeneca.pdf",
    "Job Detail - Veracyte.pdf",
    "Sr Computational Biologist - Boston, Massachusetts, United States.pdf",
    "Computational Biologist II at Immunai.pdf",
    "Data Scientist, Computational Biology - Silver Spring, MD - Indeed.com.pdf",
    "Eli_lily.pdf",
    "SandboxAQ_BioinformaticResearchEngineer.pdf",
    "DataEngineer-bioinformatics.pdf",
    "Bioinformatics Scientist.pdf",
    "Sr. Bioinformatics Scientist at Guardant Health in Palo Alto, California REF5877X.pdf",
    "Senior Scientist \u2013 Research Computational Biology (ARIA) Jobs at Amgen in United States - Remote.pdf",
    "Openings \u2014 Verge Genomics.pdf",
    "Job Openings - GeneDx\u00ae.pdf",
    "myGwork \u00a6 LGBTQ+ Jobs \u00a6 Senior Principal Scientist, Biological Network Analytics \U0001f3f3\ufe0f_\U0001f308.pdf",
    "Senior Research Scientist.pdf",
    "Principal AI Researcher at Synthesize Bio.pdf",
    "Computational Biologics Design Senior Scientist at AstraZeneca.pdf",
}


def _classify_from_filename(filename: str) -> str | None:
    """Return likely_type if filename strongly suggests one, else None."""
    lower = filename.lower().replace("-", " ").replace(".", " ")
    for likely_type, keywords in FILENAME_HINTS.items():
        if any(kw in lower for kw in keywords):
            return likely_type
    return None


def _extract_text(path: str, extension: str, max_chars: int = 3000) -> str:
    """Extract text from PDF or DOCX for content-based classification."""
    try:
        if extension == "pdf":
            reader = PdfReader(path)
            text = ""
            for p in reader.pages:
                part = (p.extract_text() or "").strip()
                text += part + " "
                if len(text) >= max_chars:
                    break
            return text[:max_chars].lower()
        if extension == "docx":
            doc = DocxDocument(path)
            text = "\n".join(p.text for p in doc.paragraphs).lower()
            return text[:max_chars]
    except Exception:
        pass
    return ""


def _classify_from_content(path: str, extension: str, max_chars: int = 3000) -> str:
    """Extract text from file and return best guess: job_description, resume, cover_letter, combined, unknown."""
    text = _extract_text(path, extension, max_chars)
    if not text:
        return "unknown"

    scores = {"job_description": 0, "cover_letter": 0, "resume": 0}
    for phrase in CONTENT_JD:
        if phrase in text:
            scores["job_description"] += 1
    for phrase in CONTENT_COVER:
        if phrase in text:
            scores["cover_letter"] += 1
    for phrase in CONTENT_RESUME:
        if phrase in text:
            scores["resume"] += 1

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "unknown"
    if scores["job_description"] >= 2 and scores["resume"] >= 2:
        return "combined"
    return best


def infer_outcome(rel_path: str) -> str:
    """Infer outcome from path under RESUME_ROOT. First component if in known set, else applied."""
    parts = Path(rel_path).parts
    if parts and parts[0] in OUTCOME_FOLDERS:
        return parts[0]
    return OUTCOME_DEFAULT


def job_folder_from_path(rel_path: str) -> str:
    """Return a short label for the application folder (for grouping)."""
    parts = Path(rel_path).parts
    if len(parts) >= 2 and parts[0] in OUTCOME_FOLDERS:
        return f"{parts[0]}/{parts[1]}"
    if parts:
        return parts[0]
    return "."


def main():
    root = Path(config.RESUME_ROOT)
    if not root.is_dir():
        print(f"Resume root not found: {root}")
        return

    rows = []
    for dirpath, _dirnames, filenames in os.walk(root):
        # Skip .venv and __pycache__
        if ".venv" in dirpath or "__pycache__" in dirpath:
            continue
        for name in filenames:
            if name.startswith("~$"):
                continue
            lower = name.lower()
            if lower.endswith(".pdf"):
                ext = "pdf"
            elif lower.endswith(".docx"):
                ext = "docx"
            else:
                continue
            path = os.path.join(dirpath, name)
            try:
                rel = os.path.relpath(path, root)
            except ValueError:
                rel = path
            outcome = infer_outcome(rel)
            job_folder = job_folder_from_path(rel)
            from_filename = _classify_from_filename(name)
            if from_filename:
                likely_type = from_filename
            else:
                likely_type = _classify_from_content(path, ext)
            # Override: these are not job descriptions (interview/recruiter notes, Gmail printouts, etc.)
            if likely_type == "job_description" and (
                name in NOT_JOB_DESCRIPTION_FILENAMES or name.startswith("Gmail")
            ):
                likely_type = "unknown"
            # Override: these are job descriptions (explicit list or resume PDFs not starting with digits)
            if name in FORCE_JOB_DESCRIPTION_FILENAMES:
                likely_type = "job_description"
            elif (
                likely_type == "resume"
                and ext == "pdf"
                and name
                and not name[0].isdigit()
                and any(x in name for x in (" _ ", "LinkedIn", "Careers"))
            ):
                likely_type = "job_description"
            # Resume PDFs with both "resume" and "cover" in filename -> combined
            if (
                likely_type == "resume"
                and ext == "pdf"
                and "resume" in name.lower()
                and "cover" in name.lower()
            ):
                likely_type = "combined"
            # Resume PDF with "letter" and "resume" in filename (e.g. letter_resume) -> combined
            if (
                likely_type == "resume"
                and ext == "pdf"
                and "resume" in name.lower()
                and "letter" in name.lower()
            ):
                likely_type = "combined"
            # Resume PDF with "cover" in filename but no "resume" -> cover_letter (cover-only misclassified)
            if (
                likely_type == "resume"
                and ext == "pdf"
                and "cover" in name.lower()
                and "resume" not in name.lower()
            ):
                likely_type = "cover_letter"
            # Pure cover_letter PDF misclassified as resume (filename has cover_letter)
            if (
                likely_type == "resume"
                and ext == "pdf"
                and "cover_letter" in name.lower()
            ):
                likely_type = "cover_letter"
            # Resume PDFs that are actually job listings / application confirmations -> job_description
            if (
                likely_type == "resume"
                and ext == "pdf"
                and (
                    "job application for" in name.lower()
                    or "jobs (now hiring)" in name.lower()
                    or "ziprecruiter" in name.lower()
                )
            ):
                likely_type = "job_description"
            # Combined PDFs that are actually LinkedIn job pages (no resume/cover in name) -> job_description
            if (
                likely_type == "combined"
                and ext == "pdf"
                and "linkedin" in name.lower()
                and " _ " in name
                and "resume" not in name.lower()
                and "cover" not in name.lower()
            ):
                likely_type = "job_description"
            rows.append({
                "path": os.path.abspath(path),
                "job_folder": job_folder,
                "outcome": outcome,
                "filename": name,
                "extension": ext,
                "likely_type": likely_type,
            })

    rows.sort(key=lambda r: (r["job_folder"], r["filename"]))
    out_path = config.RESUME_REFERENCE_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "job_folder", "outcome", "filename", "extension", "likely_type"])
        w.writeheader()
        w.writerows(rows)

    print(f"Found {len(rows)} files. Reference table saved to: {out_path}")
    # Count by likely_type and extension (e.g. resume pdf vs resume docx)
    by_type_ext = {}
    for r in rows:
        key = (r["likely_type"], r["extension"])
        by_type_ext[key] = by_type_ext.get(key, 0) + 1
    # Summary: group by likely_type, then show pdf/docx breakdown
    by_type = {}
    for (t, ext), count in by_type_ext.items():
        by_type[t] = by_type.get(t, 0) + count
    for t in sorted(by_type.keys(), key=lambda x: -by_type[x]):
        total = by_type[t]
        pdf_count = by_type_ext.get((t, "pdf"), 0)
        docx_count = by_type_ext.get((t, "docx"), 0)
        print(f"  {t}: {total} (pdf: {pdf_count}, docx: {docx_count})")


if __name__ == "__main__":
    main()
