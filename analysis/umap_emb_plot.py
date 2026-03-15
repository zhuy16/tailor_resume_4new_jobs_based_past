"""
2-D plot of job descriptions: all-MiniLM-L6-v2 embeddings → PCA(50) → UMAP(2D).
Two-panel layout: left = domain, right = outcome.
Star markers show the four final-round interview jobs.
Axis-word summary box via Spearman correlation of TF-IDF features vs UMAP coords.
Static PNG saved to data/review/umap_emb.png.

Run: conda run -n job-rag python analysis/umap_emb_plot.py
"""
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess

import chromadb
import matplotlib
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
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

import config

# ── helpers ───────────────────────────────────────────────────────────────────

def detect(text: str, patterns: list) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def tag_seniority(text: str) -> str:
    if detect(text, [r"\bdirector\b", r"\bvp\b", r"\bhead of\b"]):
        return "director"
    if detect(text, [r"\bprincipal\b", r"\bstaff\b"]):
        return "principal/staff"
    if detect(text, [r"\bsenior\b", r"\bsr\.?\b"]):
        return "senior"
    if detect(text, [r"\blead\b"]):
        return "lead"
    if detect(text, [r"\bassociate\b", r"\bjunior\b"]):
        return "associate"
    return "other"


def tag_remote(text: str) -> str:
    if detect(text, [r"\bremote\b"]):
        return "remote"
    if detect(text, [r"\bhybrid\b"]):
        return "hybrid"
    if detect(text, [r"\bon.?site\b", r"\bin.?office\b"]):
        return "on-site"
    return "unknown"


# Domain keyword patterns — each list entry is one regex; all hits are counted.
_DOMAIN_PATTERNS = {
    "single-cell":   [r"single.?cell", r"scrna", r"scatac", r"single cell rna",
                      r"10x genomics", r"seurat", r"scanpy"],
    "cell-therapy":  [r"cell.?therap", r"car.?t", r"adoptive", r"tcr.?t",
                      r"tumor.?infiltrat", r"nk cell"],
    "structural":    [r"structural bio", r"cryo.?em", r"protein struct",
                      r"molecular dock", r"alphafold", r"rosetta"],
    "clinical":      [r"\bclinical\b", r"\bhl7\b", r"\bfhir\b", r"\behr\b",
                      r"clinical trial", r"clinical data", r"clincal ops"],
    "AI/ML":         [r"machine.?learn", r"deep.?learn", r"\bllm\b", r"\bgpt\b",
                      r"neural net", r"\bai\b", r"\bml\b", r"transformer",
                      r"reinforcement", r"\bmlops\b"],
    "spatial":       [r"spatial.?omic", r"spatial transcr", r"visium",
                      r"slide.?seq", r"merfish", r"spatial gene"],
    "production":    [r"\bmanufactur", r"\bgmp\b", r"\bcmc\b", r"process dev",
                      r"scale.?up", r"\bcdmo\b", r"\bproduction\b",
                      r"tech transfer", r"\bcgmp\b"],
    "RWE/healthcare":[r"\brwe\b", r"\brwd\b", r"real.?world", r"\bhealthcare\b",
                      r"health system", r"\bpayer\b", r"\bheor\b",
                      r"\bepidemi", r"claims data", r"electronic health"],
    "R&D/discovery": [r"\br&d\b", r"\bdiscovery\b", r"research.*develop",
                      r"\bpreclinical\b", r"\btranslational\b",
                      r"early.?stage", r"basic research"],
}

def tag_domain(text: str) -> str:
    """Assign domain by total keyword-hit count across all patterns.
    Each regex match anywhere in the text scores +1 for that category.
    Ties broken by the order of _DOMAIN_PATTERNS (dict insertion order).
    Falls back to 'other' if no keywords match at all.
    """
    t = text.lower()
    scores = {}
    for domain, patterns in _DOMAIN_PATTERNS.items():
        scores[domain] = sum(len(re.findall(p, t)) for p in patterns)
    best_score = max(scores.values())
    if best_score == 0:
        return "other"
    # return first domain with the highest score (ties → insertion order)
    return next(d for d, s in scores.items() if s == best_score)


def tag_role(text: str) -> str:
    if detect(text, [r"bioinformat"]):
        return "bioinformatics"
    if detect(text, [r"data.?scien"]):
        return "data-scientist"
    if detect(text, [r"computational.?bio", r"comp.?bio"]):
        return "comp-bio"
    if detect(text, [r"engineer"]):
        return "engineer"
    if detect(text, [r"scientist"]):
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


# ── colour palettes ───────────────────────────────────────────────────────────
HIGHLIGHT_COMPANIES = {
    "astrazeneca": ("AZ",        "#1565C0"),
    "amgen":       ("Amgen",     "#6A1B9A"),
    "miltenyi":    ("Miltenyi",  "#2E7D32"),
    "uthr":        ("UTHR",      "#C62828"),
    "united ther": ("UTHR",      "#C62828"),
    "sandbox":     ("SandboxAQ", "#EF6C00"),
}

PALETTES = {
    "outcome": {
        "interviewed":   "#0173B2",
        "interviewing":  "#CC78BC",
        "rejected":      "#D55E00",
        "no response":   "#BBBBBB",
        "other":         "#949494",
    },
    "seniority": {
        "director":        "#D32F2F",
        "principal/staff": "#E64A19",
        "senior":          "#1976D2",
        "lead":            "#7B1FA2",
        "associate":       "#388E3C",
        "other":           "#9E9E9E",
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
        "cell-therapy": "#D62728",  # tab-red       (~  0°)
        "structural":   "#9467BD",  # tab-purple    (~270°)
        "clinical":     "#FF7F0E",  # tab-orange    (~ 30°)
        "AI/ML":        "#2CA02C",  # tab-green     (~120°)
        "spatial":      "#17BECF",  # tab-cyan      (~186°)
        "production":   "#8C564B",  # tab-brown     (~ 15°)
        "RWE/healthcare":"#BCBD22",  # tab-olive     (~ 65°)
        "R&D/discovery":"#E377C2",  # tab-pink      (~320°)
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
        "CI/CD":         "#F44336",
        "HL7/FHIR":      "#009688",
        "translational": "#3F51B5",
        "multi-omics":   "#E91E63",
        "spatial":       "#795548",
        "other":         "#9E9E9E",
    },
    "company": {
        "AZ":        "#1565C0",
        "Amgen":     "#6A1B9A",
        "Miltenyi":  "#2E7D32",
        "UTHR":      "#C62828",
        "SandboxAQ": "#EF6C00",
        "other":     "#BDBDBD",
    },
}

MODE_KEYS = ["outcome", "seniority", "remote", "domain", "role", "special", "company"]
KEY_MAP   = {"1": "outcome", "2": "seniority", "3": "remote",
             "4": "domain",  "5": "role",       "6": "special", "7": "company"}

# ── load embeddings from ChromaDB ─────────────────────────────────────────────
print("Loading embeddings from ChromaDB...")
client = chromadb.PersistentClient(path=config.DB_PATH)
collection = client.get_collection("job_descriptions")
result = collection.get(include=["embeddings", "metadatas", "documents"])

embeddings = np.array(result["embeddings"])
metadatas  = result["metadatas"]
documents  = result["documents"]
print(f"Loaded {len(metadatas)} documents, dim={embeddings.shape[1]}")

# ── exclude outlier companies ─────────────────────────────────────────────────
EXCLUDE_COMPANIES = {"guardant", "natera", "johnson"}
keep = [i for i, m in enumerate(metadatas)
        if not any(ex in m.get("filename", "").lower() for ex in EXCLUDE_COMPANIES)]
embeddings = embeddings[keep]
metadatas  = [metadatas[i] for i in keep]
documents  = [documents[i] for i in keep]
print(f"After excluding {EXCLUDE_COMPANIES}: {len(metadatas)} documents remaining.")

# ── prepare text for tagging ──────────────────────────────────────────────────
filenames = [m.get("filename", "") for m in metadatas]
companies = [m.get("company",  "") for m in metadatas]
outcomes  = [m.get("outcome",  "applied") for m in metadatas]
combined  = [f + " " + (d or "") for f, d in zip(filenames, documents)]

# ── embeddings → PCA(50) → UMAP(2D) ──────────────────────────────────────────
print("Running PCA(50) on embeddings...")
pca = PCA(n_components=50, random_state=42)
emb_pca = pca.fit_transform(embeddings)
print(f"PCA(50) explained variance: {pca.explained_variance_ratio_.sum():.1%}")

print("Running UMAP on PCA-50 space (this takes ~15-30 s)...")
reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
coords = reducer.fit_transform(emb_pca)
print("UMAP done.")

# ── TF-IDF for axis-word annotation only ─────────────────────────────────────
print("Building TF-IDF matrix (for axis-word annotation)...")
_vec = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    stop_words="english",
    min_df=3,
    max_df=0.85,
    sublinear_tf=True,
)
_tfidf = _vec.fit_transform([d or "" for d in documents]).toarray()
_vocab = np.array(_vec.get_feature_names_out())
print(f"TF-IDF matrix: {_tfidf.shape}")

# ── tag every document ────────────────────────────────────────────────────────
tags = {
    "outcome":   outcomes,
    "seniority": [tag_seniority(t) for t in combined],
    "remote":    [tag_remote(t)    for t in combined],
    "domain":    [tag_domain(t)    for t in combined],
    "role":      [tag_role(t)      for t in combined],
    "special":   [tag_special(t)   for t in combined],
}

def tag_company(fname: str) -> str:
    fl = fname.lower()
    for kw, (label, _) in HIGHLIGHT_COMPANIES.items():
        if kw in fl:
            return label
    return "other"

tags["company"] = [tag_company(f) for f in filenames]
labels = [f"{c}  |  {f}" for c, f in zip(companies, filenames)]

# merge applied / 0_slow → "no response"
tags["outcome"] = [
    "no response" if o in ("applied", "0_slow") else o
    for o in tags["outcome"]
]

# ── data-driven axis words via Spearman correlation ───────────────────────────
print("Computing Spearman correlations with UMAP axes...")
_r1 = np.array([spearmanr(_tfidf[:, j], coords[:, 0]).statistic for j in range(len(_vocab))])
_r2 = np.array([spearmanr(_tfidf[:, j], coords[:, 1]).statistic for j in range(len(_vocab))])

TOP_AXIS = 12
_ax1_pos = _vocab[np.argsort(_r1)[-TOP_AXIS:][::-1]].tolist()
_ax1_neg = _vocab[np.argsort(_r1)[:TOP_AXIS]].tolist()
_ax2_pos = _vocab[np.argsort(_r2)[-TOP_AXIS:][::-1]].tolist()
_ax2_neg = _vocab[np.argsort(_r2)[:TOP_AXIS]].tolist()
print("UMAP 1  (+):", _ax1_pos[:6])
print("UMAP 1  (-):", _ax1_neg[:6])
print("UMAP 2  (+):", _ax2_pos[:6])
print("UMAP 2  (-):", _ax2_neg[:6])


def get_colors(mode):
    palette = PALETTES[mode]
    return [palette.get(t, palette.get("other", "#949494")) for t in tags[mode]]


# ── two-panel figure ──────────────────────────────────────────────────────────
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(22, 9))
fig.patch.set_facecolor("white")
plt.subplots_adjust(wspace=0.12, top=0.88)

fig.suptitle("Job Description Embedding Space",
             fontsize=16, fontweight="bold", y=0.97)
fig.text(0.5, 0.91,
         f"all-MiniLM-L6-v2  \u2192  PCA(50)  \u2192  UMAP(2D)   \u2502   {len(metadatas)} job descriptions",
         ha="center", fontsize=10, color="#666666", style="italic")


def _style_ax(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#888888", labelsize=10)


def scatter_on(ax, mode, panel_title):
    _style_ax(ax)
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=get_colors(mode), s=55, alpha=0.82,
        linewidths=0.5, edgecolors="white", zorder=2,
    )
    ax.set_title(panel_title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("UMAP 1", fontsize=12, labelpad=6)
    ax.set_ylabel("UMAP 2", fontsize=12, labelpad=6)
    palette = PALETTES[mode]
    leg = ax.legend(
        handles=[mpatches.Patch(facecolor=c, label=k) for k, c in palette.items()],
        title=mode.capitalize(), title_fontsize=10, fontsize=9,
        loc="upper right", framealpha=0.9, edgecolor="#DDDDDD",
    )
    leg.get_title().set_fontweight("bold")
    return sc


sc_l = scatter_on(ax_l, "domain",  "Colored by Domain")
sc_r = scatter_on(ax_r, "outcome", "Colored by Outcome")

# ── axis-word summary box (bottom-left of left panel) ────────────────────────
_axis_summary = (
    f"\u2190 {_ax1_neg[0]}, {_ax1_neg[1]}   {_ax1_pos[0]}, {_ax1_pos[1]} \u2192\n"
    f"\u2193 {_ax2_neg[0]}, {_ax2_neg[1]}   {_ax2_pos[0]}, {_ax2_pos[1]} \u2191"
)
ax_l.text(0.01, 0.01, _axis_summary,
          transform=ax_l.transAxes, fontsize=8, color="#555555",
          va="bottom", ha="left",
          bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#CCCCCC", alpha=0.90))

# ── final-round interview stars ───────────────────────────────────────────────
FINAL_ROUND_JOBS = [
    ("Senior Scientist, Cell Therapy Discovery at AstraZeneca", "AZ"),
    ("Scientist II - Computational Biology _PC 892 _ Miltenyi",  "MT"),
    ("Senior Scientist \u2013 Research Computational Biology (ARIA) Jobs at Amgen", "AM"),
    ("Data Scientist, Computational Biology - Silver Spring",    "UT"),
]


def draw_stars(ax):
    for fname_substr, job_label in FINAL_ROUND_JOBS:
        idx = next((i for i, m in enumerate(metadatas)
                    if fname_substr.lower() in m.get("filename", "").lower()), None)
        if idx is None:
            print(f"WARNING: final-round job not found: {fname_substr}")
            continue
        x, y = coords[idx, 0], coords[idx, 1]
        ax.plot(x, y, "*", markersize=18, color="#888888", alpha=0.42,
                markeredgecolor="#555555", markeredgewidth=1.2, zorder=5)
        ax.annotate(
            job_label,
            xy=(x, y), xytext=(10, 0), textcoords="offset points",
            fontsize=8, color="#999999", alpha=0.70, fontweight="bold",
            va="center", ha="left", zorder=6,
        )


draw_stars(ax_l)
draw_stars(ax_r)

# ── hover tooltip (both panels) ───────────────────────────────────────────────
annot_l = ax_l.annotate("", xy=(0, 0), xytext=(12, 12),
                         textcoords="offset points", fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9))
annot_r = ax_r.annotate("", xy=(0, 0), xytext=(12, 12),
                         textcoords="offset points", fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9))
annot_l.set_visible(False)
annot_r.set_visible(False)


def on_hover(event):
    for sc, annot, ax, mode in [(sc_l, annot_l, ax_l, "domain"),
                                 (sc_r, annot_r, ax_r, "outcome")]:
        if event.inaxes != ax:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            continue
        cont, ind = sc.contains(event)
        if cont:
            i = ind["ind"][0]
            tag = tags[mode][i]
            annot.xy = (coords[i, 0], coords[i, 1])
            annot.set_text(f"{labels[i]}\n[{mode}: {tag}]")
            annot.set_visible(True)
            fig.canvas.draw_idle()
        elif annot.get_visible():
            annot.set_visible(False)
            fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", on_hover)

# ── save & open ───────────────────────────────────────────────────────────────
out_png = os.path.join(config._PROJECT_DIR, "data", "review", "umap_emb_freq.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved static plot: {out_png}")

subprocess.Popen(["open", out_png])

if _CAN_SHOW:
    plt.show()
else:
    print("Interactive window not available (Agg backend). PNG opened in Preview.")
    print("For interactive mode run:  conda activate job-rag && python analysis/umap_emb_plot.py")
