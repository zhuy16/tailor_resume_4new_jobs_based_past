"""
2-D plot of job descriptions in TF-IDF vocabulary space (independent of embeddings).

Instead of using sentence-transformer embeddings, this script builds a TF-IDF matrix
directly from the document text and then runs t-SNE on it — so the layout reflects
raw keyword co-occurrence, not semantic similarity.

Colour modes (press keys 1-7 to switch live):
  1 = outcome         (applied / rejected / interviewed / interviewing)
  2 = seniority       (senior / principal / director / staff / associate / other)
  3 = remote          (remote / hybrid / on-site / unknown)
  4 = domain          (single-cell / structural / clinical / cell-therapy / AI-ML / spatial / other)
  5 = role type       (scientist / engineer / data-scientist / bioinformatics / comp-bio / other)
  6 = special terms   (CI/CD / HL7 / translational / multi-omics / spatial / other)
  7 = company group   (AZ / Amgen / Miltenyi / UTHR / SandboxAQ / Guardant / Natera / other)

Hover over any point to see company + filename.
Star markers show centroids for key companies.
A static PNG is saved to data/review/tfidf_tsne.png.

Run: conda run -n job-rag python analysis/tfidf_plot.py
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
        plt.figure(); plt.close()   # probe: fails if backend can't render
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
from sklearn.manifold import TSNE

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


# ── colour palettes ───────────────────────────────────────────────────────────

HIGHLIGHT_COMPANIES = {
    "astrazeneca": ("AZ",        "#1565C0"),
    "amgen":       ("Amgen",     "#6A1B9A"),
    "miltenyi":    ("Miltenyi",  "#2E7D32"),
    "uthr":        ("UTHR",      "#C62828"),
    "united ther": ("UTHR",      "#C62828"),
    "sandbox":     ("SandboxAQ", "#EF6C00"),
    "guardant":    ("Guardant",  "#00695C"),
    "natera":      ("Natera",    "#4527A0"),
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
        "single-cell":  "#E91E63",
        "cell-therapy": "#9C27B0",
        "structural":   "#3F51B5",
        "clinical":     "#009688",
        "AI/ML":        "#FF5722",
        "spatial":      "#795548",
        "other":        "#9E9E9E",
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
    "company": COMPANY_GROUP_PALETTE,
}

MODE_KEYS = ["outcome", "seniority", "remote", "domain", "role", "special", "company"]
KEY_MAP   = {"1": "outcome", "2": "seniority", "3": "remote",
             "4": "domain",  "5": "role",       "6": "special", "7": "company"}


# ── load documents from ChromaDB ──────────────────────────────────────────────
print("Loading documents from ChromaDB...")
client = chromadb.PersistentClient(path=config.DB_PATH)
collection = client.get_collection("job_descriptions")
result = collection.get(include=["metadatas", "documents"])

metadatas  = result["metadatas"]
documents  = result["documents"]
print(f"Loaded {len(metadatas)} documents")

# ── exclude outlier companies ─────────────────────────────────────────────────
EXCLUDE_COMPANIES = {"guardant", "natera", "johnson"}
keep = [i for i, m in enumerate(metadatas)
        if not any(ex in m.get("filename", "").lower() for ex in EXCLUDE_COMPANIES)]
metadatas  = [metadatas[i] for i in keep]
documents  = [documents[i] for i in keep]
print(f"After excluding {EXCLUDE_COMPANIES}: {len(metadatas)} documents remaining.")

# ── prepare text for tagging ──────────────────────────────────────────────────
filenames = [m.get("filename", "") for m in metadatas]
companies = [m.get("company",  "") for m in metadatas]
outcomes  = [m.get("outcome",  "applied") for m in metadatas]
combined  = [f + " " + (d or "") for f, d in zip(filenames, documents)]

# ── TF-IDF → PCA(50) → t-SNE(2) ─────────────────────────────────────────
print("Building TF-IDF matrix...")
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

N_PCA = 50
print(f"Running PCA ({N_PCA} components) on TF-IDF...")
pca = PCA(n_components=N_PCA, random_state=42)
pca_coords = pca.fit_transform(_tfidf)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

print("Running t-SNE on PCA-50 space (this takes ~10-20 s)...")
reducer = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    metric="cosine",
    random_state=42,
)
coords = reducer.fit_transform(pca_coords)
print("t-SNE done.")



# ── tag every document ────────────────────────────────────────────────────────
tags = {
    "outcome":   outcomes,
    "seniority": [tag_seniority(t) for t in combined],
    "remote":    [tag_remote(t)    for t in combined],
    "domain":    [tag_domain(t)    for t in combined],
    "role":      [tag_role(t)      for t in combined],
    "special":   [tag_special(t)   for t in combined],
}

# Tag company group for each doc
def tag_company(fname: str) -> str:
    fl = fname.lower()
    for kw, (label, _) in HIGHLIGHT_COMPANIES.items():
        if kw in fl:
            return label
    return "other"

tags["company"] = [tag_company(f) for f in filenames]
labels = [f"{c}  |  {f}" for c, f in zip(companies, filenames)]

# ── data-driven axis words via Spearman correlation ───────────────────────────
# Reuse the same TF-IDF matrix built above.
print("Computing Spearman correlations with t-SNE axes...")
_r1 = np.array([spearmanr(_tfidf[:, j], coords[:, 0]).statistic for j in range(len(_vocab))])
_r2 = np.array([spearmanr(_tfidf[:, j], coords[:, 1]).statistic for j in range(len(_vocab))])

TOP_AXIS = 12  # candidate pool per axis end (top-N by Spearman)
_ax1_pos = _vocab[np.argsort(_r1)[-TOP_AXIS:][::-1]].tolist()
_ax1_neg = _vocab[np.argsort(_r1)[:TOP_AXIS]].tolist()
_ax2_pos = _vocab[np.argsort(_r2)[-TOP_AXIS:][::-1]].tolist()
_ax2_neg = _vocab[np.argsort(_r2)[:TOP_AXIS]].tolist()
print("t-SNE 1  (+):", _ax1_pos)
print("t-SNE 1  (-):", _ax1_neg)
print("t-SNE 2  (+):", _ax2_pos)
print("t-SNE 2  (-):", _ax2_neg)

# ── palette overrides — matplotlib tab10 for wide hue separation ─────────────
# structural/spatial/AI-ML/clinical each placed at a distinct hue region:
#   purple (~270°) / cyan (~186°) / red (~0°) / orange (~30°)
PALETTES["domain"] = {
    "single-cell":  "#1F77B4",  # tab-blue      (~210°)
    "cell-therapy": "#2CA02C",  # tab-green     (~120°)
    "structural":   "#9467BD",  # tab-purple    (~270°)
    "clinical":     "#FF7F0E",  # tab-orange    (~ 30°)
    "AI/ML":        "#D62728",  # tab-red       (~  0°)
    "spatial":      "#17BECF",  # tab-cyan      (~186°)
    "other":        "#7F7F7F",  # medium gray
}
PALETTES["outcome"] = {
    "interviewed":   "#0173B2",
    "interviewing":  "#CC78BC",
    "rejected":      "#D55E00",
    "no response":   "#BBBBBB",
    "other":         "#949494",
}


# merge applied / 0_slow → "no response"
tags["outcome"] = [
    "no response" if o in ("applied", "0_slow") else o
    for o in tags["outcome"]
]


def get_colors(mode):
    palette = PALETTES[mode]
    return [palette.get(t, palette.get("other", "#949494")) for t in tags[mode]]


# ── two-panel figure ──────────────────────────────────────────────────────────
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(22, 9))
fig.patch.set_facecolor("white")
plt.subplots_adjust(wspace=0.12, top=0.88)

# Bold main title + italic methodology subtitle
fig.suptitle("Job Description Embedding Space",
             fontsize=16, fontweight="bold", y=0.97)
fig.text(0.5, 0.91,
         "all-MiniLM-L6-v2  \u2192  PCA(50)  \u2192  UMAP(2D)   \u2502   201 job descriptions",
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
    ax.set_xlabel("t-SNE 1", fontsize=12, labelpad=6)
    ax.set_ylabel("t-SNE 2", fontsize=12, labelpad=6)
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

# ── axis-word summary box (bottom-left corner of left panel) ─────────────────
_axis_summary = (
    f"\u2190 {_ax1_neg[0]}, {_ax1_neg[1]}   {_ax1_pos[0]}, {_ax1_pos[1]} \u2192\n"
    f"\u2193 {_ax2_neg[0]}, {_ax2_neg[1]}   {_ax2_pos[0]}, {_ax2_pos[1]} \u2191"
)
ax_l.text(0.01, 0.01, _axis_summary,
          transform=ax_l.transAxes, fontsize=8, color="#555555",
          va="bottom", ha="left",
          bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#CCCCCC", alpha=0.90))

# ── final-round interview stars (transparent gray + leader-line labels) ───────
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
        # Label immediately to the right of the star, in pixel offset space
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
out_png = os.path.join(config._PROJECT_DIR, "data", "review", "tfidf_tsne.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved static plot: {out_png}")

subprocess.Popen(["open", out_png])

if _CAN_SHOW:
    plt.show()
else:
    print("Interactive window not available (Agg backend). PNG opened in Preview.")
    print("For interactive mode run:  conda activate job-rag && python analysis/tfidf_plot.py")
