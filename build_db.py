"""
Embed job description PDFs into a persistent ChromaDB.
Reads job_descriptions_pdf_list.csv, extracts text from each PDF,
and upserts into ChromaDB with metadata (job_folder, outcome, company).
Run: python build_db.py
"""
import csv
import os
import re

import chromadb
from PyPDF2 import PdfReader

import config


def extract_text(path: str, max_chars: int = 8000) -> str:
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + " "
            if len(text) >= max_chars:
                break
        return text[:max_chars].strip()
    except Exception as e:
        print(f"  [warn] Could not read {path}: {e}")
        return ""


def company_from_folder(job_folder: str) -> str:
    """Extract company name from job_folder like '0_slow/0818_BAMF' -> 'BAMF'."""
    last = job_folder.split("/")[-1]
    # Strip leading date prefix (digits and dots)
    name = re.sub(r"^\d[\d.]*_?", "", last)
    return name if name else last


def main():
    jd_csv = os.path.join(config._PROJECT_DIR, "data", "job_descriptions_pdf_list.csv")
    if not os.path.exists(jd_csv):
        print(f"JD list not found: {jd_csv}")
        return

    # Read the master reference to get job_folder and outcome per filename
    ref_csv = config.RESUME_REFERENCE_PATH
    meta_by_path = {}
    if os.path.exists(ref_csv):
        with open(ref_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                meta_by_path[row["path"]] = {
                    "job_folder": row["job_folder"],
                    "outcome": row["outcome"],
                }

    # Load JD list
    with open(jd_csv, newline="", encoding="utf-8") as f:
        jd_rows = list(csv.DictReader(f))
    print(f"JD list: {len(jd_rows)} entries")

    # Connect to persistent ChromaDB
    os.makedirs(config.DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=config.DB_PATH)
    collection = client.get_or_create_collection(
        name="job_descriptions",
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get(include=[])["ids"])
    print(f"Already in DB: {len(existing_ids)} documents")

    added = skipped = failed = 0
    for i, row in enumerate(jd_rows, 1):
        path = row["path"]
        filename = row["filename"]
        doc_id = path  # use full path as stable unique ID

        if doc_id in existing_ids:
            skipped += 1
            continue

        print(f"[{i}/{len(jd_rows)}] Embedding: {filename}")
        text = extract_text(path)
        if not text:
            print(f"  [skip] No text extracted.")
            failed += 1
            continue

        meta = meta_by_path.get(path, {})
        job_folder = meta.get("job_folder", "")
        outcome = meta.get("outcome", "applied")
        company = company_from_folder(job_folder)

        collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[{
                "filename": filename,
                "path": path,
                "job_folder": job_folder,
                "outcome": outcome,
                "company": company,
            }],
        )
        added += 1

    print(f"\nDone. Added: {added} | Skipped (already in DB): {skipped} | Failed: {failed}")
    print(f"Total in DB: {collection.count()}")


if __name__ == "__main__":
    main()
