"""
UMAP plot of job-description embeddings in ChromaDB.

Colour modes (press keys 1-6 to switch live):
  1 = outcome         (applied / rejected / interviewed / interviewing)
  2 = seniority       (senior / principal / director / staff / associate / other)
  3 = remote          (remote / hybrid / on-site / unknown)
  4 = domain          (single-cell / structural / clinical / cell-therapy / AI-ML / other)
  5 = role type       (scientist / engineer / data-scientist / bioinformatics / other)
  6 = special terms   (CI/CD / HL7 / translational / spatial / multiomics / other)

Hover over any point to see company + filename.
A static PNG is saved to data/review/umap_embeddings.png.

Run: conda run -n job-rag python explore_embeddings.py
"""
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
import matplotlib
import subprocess
for _backend in ("MacOSX", "TkAgg", "Qt5Agg", "Agg"):
    try:
        matplotlib.use(_backend)
        import matplotlib.pyplot as plt
        plt.figure(); plt.close()
        _CAN_SHOW = (_backend != "Agg")
        print(f"matplotlib backend: {_backend}")
        break
    except Exception:
        pass
import matplotlib.patches as mpatches
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

import config

# ── helpers ──────────────────────────────────────────────────────────────────

def detect(text: str, patterns: list[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def tag_seniority(text: str) -> str:
    if detect(text, [r"\bdirector\b", r"\bvp\b", r"\bhead of\b"]):
        return "director"
    if detect(text, [r"\bprincipal\b", r"\bstaff\b"]):
        return "principal/staff"
    if detect(text, [r"\bsenior\b", r"\bsr\.?\b"]):
        return "senior"
    if detect(text, [r"\bassociate\b", r"\bjunior\b"]):
        return "associate"
    if detect(text, [r"\blead\b"]):
        return "lead"
    return "other"


def tag_remote(text: str) -> str:
    if detect(text, [r"\bremote\b"]):
        return "remote"
    if detect(text, [r"\bhybrid\b"]):
        return "hybrid"
    if detect(text, [r"\bon.?site\b", r"\bin.?person\b"]):
        return "on-site"
    return "unknown"


def tag_domain(text: str) -> str:
    if detect(text, [r"single.?cell", r"scrna", r"scatac"]):
        return "single-cell"
    if detect(text, [r"cell.?therap", r"car.?t\b", r"adoptive"]):
        return "cell-therapy"
    if detect(text, [r"structural bio", r"cryo.?em", r"protein struct"]):
        return "structural"
    if detect(text, [r"\bclinical\b", r"\bhl7\b", r"\bfhir\b", r"\behr\b"]):
        return "clinical"
    if detect(text, [r"\bai\b", r"\bllm\b", r"machine.?learn", r"deep.?learn", r"neural"]):
        return "AI/ML"
    if detect(text, [r"spatial.?omic", r"spatial transcr"]):
        return "spatial"
    return "other"


def tag_role(text: str) -> str:
    if detect(text, [r"bioinformat"]):
        return "bioinformatics"
    if detect(text, [r"data.?scientist", r"data scientist"]):
        return "data-scientist"
    if detect(text, [r"computational bio"]):
        return "comp-bio"
    if detect(text, [r"software.?engineer", r"ml.?engineer", r"data.?engineer"]):
        return "engineer"
    if detect(text, [r"\bscientist\b"]):
        return "scientist"
    return "other"


def tag_special(text: str) -> str:
    if detect(text, [r"ci[/_]cd", r"devops", r"docker", r"kubernetes"]):
        return "CI/CD"
    if detect(text, [r"\bhl7\b", r"\bfhir\b"]):
        return "HL7/FHIR"
    if detect(text, [r"translational"]):
        return "translational"
    if detect(text, [r"multi.?omic", r"proteom", r"metabolom"]):
        return "multi-omics"
    if detect(text, [r"spatial"]):
        return "spatial"
    return "other"


# colour palettes
PALETTES = {
    "outcome": {
        "interviewed":   "#2196F3",
        "interviewing":  "#9C27B0",
        "rejected":      "#F44336",
        "applied":       "#4CAF50",
        "0_slow":        "#FF9800",
        "other":         "#9E9E9E",
    },
    "seniority": {
        "director":       "#D32F2F",
        "principal/staff":"#E64A19",
        "senior":         "#1976D2",
        "lead":           "#7B1FA2",
        "associate":      "#388E3C",
        "other":          "#9E9E9E",
    },
    "remote": {
        "remote":  "#2196F3",
        "hybrid":  "#FF9800",
        "on-site": "#F44336",
        "unknown": "#9E9E9E",
    },
    "domain": {
        # matplotlib tab10 — max hue spacing for structural/spatial/AI-ML/clinical
        "single-cell":  "#1F77B4",  # tab-blue      (~210°)
        "cell-therapy": "#2CA02C",  # tab-green     (~120°)
        "structural":   "#9467BD",  # tab-purple    (~270°)
        "clinical":     "#FF7F0E",  # tab-orange    (~ 30°)
        "AI/ML":        "#D62728",  # tab-red       (~  0°)
        "spatial":      "#17BECF",  # tab-cyan      (~186°)
        "other":        "#7F7F7F",  # medium gray
    },
    "role": {
        "bioinformatics": "#1976D2",
        "data-scientist": "#388E3C",
        "comp-bio":       "#E91E63",
        "engineer":       "#FF9800",
        "scientist":      "#9C27B0",
        "other":          "#9E9E9E",
    },
    "special": {
        "CI/CD":        "#F44336",
        "HL7/FHIR":     "#009688",
        "translational":"#3F51B5",
        "multi-omics":  "#E91E63",
        "spatial":      "#795548",
        "other":        "#9E9E9E",
    },
}

# Companies to always highlight with a star + label
# key = substring to match in filename (case-insensitive), value = (display label, color)
HIGHLIGHT_COMPANIES = {
    "astrazeneca": ("AZ",       "#1565C0"),
    "amgen":       ("Amgen",    "#6A1B9A"),
    "miltenyi":    ("Miltenyi", "#2E7D32"),
    "uthr":        ("UTHR",     "#C62828"),
    "united ther": ("UTHR",     "#C62828"),
    "sandbox":     ("SandboxAQ","#EF6C00"),
    "guardant":    ("Guardant", "#00695C"),
    "natera":      ("Natera",   "#4527A0"),
}

COMPANY_GROUP_PALETTE = {
    "AZ":        "#1565C0",
    "Amgen":     "#6A1B9A",
    "Miltenyi":  "#2E7D32",
    "UTHR":      "#C62828",
    "SandboxAQ": "#EF6C00",
    "Guardant":  "#00695C",
    "Natera":    "#4527A0",
    "other":     "#BDBDBD",
}
PALETTES["company"] = COMPANY_GROUP_PALETTE

MODE_KEYS = ["outcome", "seniority", "remote", "domain", "role", "special", "company"]
KEY_MAP   = {"1": "outcome", "2": "seniority", "3": "remote",
             "4": "domain",  "5": "role",       "6": "special", "7": "company"}

# ── load embeddings ───────────────────────────────────────────────────────────
client = chromadb.PersistentClient(path=config.DB_PATH)
collection = client.get_collection("job_descriptions")
result = collection.get(include=["embeddings", "metadatas", "documents"])

embeddings = np.array(result["embeddings"])
metadatas  = result["metadatas"]
documents  = result["documents"]
print(f"Loaded {len(embeddings)} embeddings, dim={embeddings.shape[1]}")

# ── exclude outlier companies ────────────────────────────────────────────────
EXCLUDE_COMPANIES = {"guardant", "natera", "johnson"}
keep = [i for i, m in enumerate(metadatas)
        if not any(ex in m.get("filename", "").lower() for ex in EXCLUDE_COMPANIES)]
embeddings = embeddings[keep]
metadatas  = [metadatas[i]  for i in keep]
documents  = [documents[i]  for i in keep]
print(f"After excluding {EXCLUDE_COMPANIES}: {len(embeddings)} documents remaining.")

# ── tag every document ───────────────────────────────────────────────────────
filenames = [m.get("filename", "") for m in metadatas]
companies = [m.get("company",  "") for m in metadatas]
outcomes  = [m.get("outcome",  "applied") for m in metadatas]

# combine filename + document text for keyword detection
combined = [f + " " + (d or "") for f, d in zip(filenames, documents)]

tags = {
    "outcome":   [outcomes[i] for i in range(len(metadatas))],
    "seniority": [tag_seniority(t) for t in combined],
    "remote":    [tag_remote(t)    for t in combined],
    "domain":    [tag_domain(t)    for t in combined],
    "role":      [tag_role(t)      for t in combined],
    "special":   [tag_special(t)   for t in combined],
}
labels = [f"{c}  |  {f}" for c, f in zip(companies, filenames)]

# ── embeddings → PCA(50) → UMAP(2D) ─────────────────────────────────────────
print("Running PCA(50) on embeddings...")
pca50 = PCA(n_components=50, random_state=42)
emb_pca = pca50.fit_transform(embeddings)
print(f"PCA(50) explained variance: {pca50.explained_variance_ratio_.sum():.1%}")

print("Running UMAP on PCA-50 space (this takes ~10-20 s)...")
reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
coords = reducer.fit_transform(emb_pca)
print("UMAP done.")

# ── plot ─────────────────────────────────────────────────────────────────────
current_mode = ["outcome"]   # mutable container so closure can update it

fig, ax = plt.subplots(figsize=(13, 9))
plt.subplots_adjust(bottom=0.08)


def get_colors(mode):
    palette = PALETTES[mode]
    return [palette.get(t, palette.get("other", "#9E9E9E")) for t in tags[mode]]


sc = ax.scatter(coords[:, 0], coords[:, 1],
                c=get_colors("outcome"), s=35, alpha=0.8,
                linewidths=0.3, edgecolors="white")

ax.set_title("Job Description Embeddings  —  all-MiniLM-L6-v2 → PCA(50) → UMAP(2D)  (press 1-7 to recolour)", fontsize=12)
ax.set_xlabel("UMAP-1  →  clinical/oncology")
ax.set_ylabel("UMAP-2  →  pharma/drug-discovery")

# ── Star markers for final-round interview jobs ───────────────────────────────
# Each star lands on the exact document point (not a centroid).
FINAL_ROUND_JOBS = [
    ("Senior Scientist, Cell Therapy Discovery at AstraZeneca", "AZ"),
    ("Scientist II - Computational Biology _PC 892 _ Miltenyi",  "MT"),
    ("Senior Scientist \u2013 Research Computational Biology (ARIA) Jobs at Amgen", "AM"),
    ("Data Scientist, Computational Biology - Silver Spring",    "UT"),
]

for fname_substr, job_label in FINAL_ROUND_JOBS:
    idx = next((i for i, m in enumerate(metadatas)
                if fname_substr.lower() in m.get("filename", "").lower()), None)
    if idx is None:
        print(f"WARNING: final-round job not found: {fname_substr}")
        continue
    x, y = coords[idx, 0], coords[idx, 1]
    ax.plot(x, y, "*", markersize=18, color="#888888", alpha=0.42,
            markeredgecolor="#555555", markeredgewidth=1.2, zorder=5)
    ax.annotate(job_label, xy=(x, y), xytext=(10, 0),
                textcoords="offset points",
                fontsize=8, color="#999999", alpha=0.85, fontweight="bold",
                va="center", ha="left", zorder=6)


def refresh(mode):
    colors = get_colors(mode)
    sc.set_facecolors(colors)
    palette = PALETTES[mode]
    # rebuild legend
    for p in ax.patches:
        p.remove()
    handles = [mpatches.Patch(color=c, label=k) for k, c in palette.items()]
    ax.legend(handles=handles, title=mode.capitalize(), loc="upper right", fontsize=8)
    ax.set_title(f"UMAP — coloured by {mode}  (press 1-6 to switch)", fontsize=12)
    fig.canvas.draw_idle()


refresh("outcome")

# hover
annot = ax.annotate("", xy=(0, 0), xytext=(12, 12),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9),
                    fontsize=8)
annot.set_visible(False)


def on_hover(event):
    if event.inaxes != ax:
        return
    cont, ind = sc.contains(event)
    if cont:
        idx = ind["ind"][0]
        mode = current_mode[0]
        tag  = tags[mode][idx]
        annot.xy = (coords[idx, 0], coords[idx, 1])
        annot.set_text(f"{labels[idx]}\n[{mode}: {tag}]")
        annot.set_visible(True)
        fig.canvas.draw_idle()
    elif annot.get_visible():
        annot.set_visible(False)
        fig.canvas.draw_idle()


def on_key(event):
    if event.key in KEY_MAP:
        mode = KEY_MAP[event.key]
        current_mode[0] = mode
        refresh(mode)
        # save PNG for current mode
        out = os.path.join(config._PROJECT_DIR, "data", "review", f"umap_{mode}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")


fig.canvas.mpl_connect("motion_notify_event", on_hover)
fig.canvas.mpl_connect("key_press_event",     on_key)

# save default PNG
out_png = os.path.join(config._PROJECT_DIR, "data", "review", "umap_embeddings.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved static plot: {out_png}")
print("Keys: 1=outcome  2=seniority  3=remote  4=domain  5=role  6=special-terms")

subprocess.Popen(["open", out_png])

if _CAN_SHOW:
    plt.tight_layout()
    plt.show()
else:
    print("Interactive window not available (Agg backend). PNG opened in Preview.")
    print("For interactive mode run:  conda activate job-rag && python analysis/explore_embeddings.py")
