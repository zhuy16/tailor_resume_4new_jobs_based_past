"""
2-D PCA plot of job descriptions via TF-IDF → PCA(2).

Pipeline:
  1. TF-IDF (1-2 grams, custom stop words)  — sparse keyword space
  2. TruncatedSVD(20) oversampled, top-2 by explained variance  — 2-D PCA

PC1/PC2 axis labels show the top driving TF-IDF words.
Outlier companies (Guardant, Natera, J&J) excluded from fitting.

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
A static PNG is saved to data/review/pca_tfidf.png.

Run: conda run -n job-rag python analysis/pca_plot.py
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
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

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
        # Tableau-10-inspired — each group at a distinct hue region
        "single-cell":  "#E15759",  # warm red       (~  0°)
        "cell-therapy": "#59A14F",  # mid green      (~130°)
        "structural":   "#4E79A7",  # steel blue     (~210°)
        "clinical":     "#F28E2B",  # amber orange   (~ 35°)
        "AI/ML":        "#B07AA1",  # muted purple   (~285°)
        "spatial":      "#76B7B2",  # teal/turquoise (~185°)
        "other":        "#BAB0AC",  # warm gray
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

metadatas = result["metadatas"]
documents = result["documents"]
print(f"Loaded {len(documents)} documents.")

filenames = [m.get("filename", "") for m in metadatas]
companies = [m.get("company",  "") for m in metadatas]
outcomes  = [m.get("outcome",  "applied") for m in metadatas]

# Filter out companies whose boilerplate dominates PCA axes.
EXCLUDE_COMPANIES = {"guardant", "natera", "johnson"}
keep = [i for i, f in enumerate(filenames)
        if not any(ex in f.lower() for ex in EXCLUDE_COMPANIES)]
metadatas = [metadatas[i] for i in keep]
filenames = [filenames[i] for i in keep]
companies = [companies[i] for i in keep]
outcomes  = [outcomes[i]  for i in keep]
documents = [documents[i] for i in keep]
print(f"After excluding {EXCLUDE_COMPANIES}: {len(filenames)} documents remaining.")

docs_clean = [d if d else "" for d in documents]
combined   = [f + " " + d for f, d in zip(filenames, docs_clean)]

# ── TF-IDF → PCA(2) ──────────────────────────────────────────────────────────────

_CUSTOM_STOP = {
    "natera", "natera com", "natera employees", "natera is",
    "guardant", "guardant health",
    "johnson", "janssen", "spring house",
    "astrazeneca", "astra", "zeneca",
    "amgen", "miltenyi", "miltenyi biotec", "sandboxaq", "sandbox",
    "uthr", "united therapeutics",
    "cyber", "crimes", "cyber crimes", "criminal",
    "bonus", "applicable bonus", "does include", "include benefits",
    "position range", "salary", "compensation", "pay range",
    "equal opportunity", "eeo", "affirmative action", "disability",
    "veteran", "badge", "badge veteran", "badge has",
    "secretary", "meta data", "data meta",
    "ll", "ve", "including", "eg", "ie",
}
_STOP_WORDS = list(ENGLISH_STOP_WORDS.union(_CUSTOM_STOP))

print("Building TF-IDF matrix...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.90,
    sublinear_tf=True,
    strip_accents="unicode",
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b",
    stop_words=_STOP_WORDS,
)
tfidf_matrix = vectorizer.fit_transform(combined)
feature_names = np.array(vectorizer.get_feature_names_out())
print(f"TF-IDF shape: {tfidf_matrix.shape}  ({tfidf_matrix.shape[1]} features)")

# TruncatedSVD(20) oversampled — reorder by variance so PC1 > PC2
N_OVERSAMPLE = 20
print(f"Running TruncatedSVD({N_OVERSAMPLE}), selecting top-2 by explained variance...")
svd = TruncatedSVD(n_components=N_OVERSAMPLE, random_state=42)
coords_full = svd.fit_transform(tfidf_matrix)

order_idx = np.argsort(svd.explained_variance_ratio_)[::-1]
coords_sorted     = coords_full[:, order_idx]
components_sorted = svd.components_[order_idx]
evr_sorted        = svd.explained_variance_ratio_[order_idx]

coords   = coords_sorted[:, :2]
explained = evr_sorted[:2]
print(f"Explained variance: PC1={explained[0]:.1%}  PC2={explained[1]:.1%}  "
      f"(total all {N_OVERSAMPLE} comps: {evr_sorted.sum():.1%})")

word_loadings = components_sorted[:2]   # (2, n_vocab)

N_AXIS_WORDS = 6

def top_words_for_loading(loading_vec, n=N_AXIS_WORDS):
    order = np.argsort(loading_vec)
    neg = feature_names[order[:n]].tolist()
    pos = feature_names[order[-n:][::-1]].tolist()
    return neg, pos

pc1_neg, pc1_pos = top_words_for_loading(word_loadings[0])
pc2_neg, pc2_pos = top_words_for_loading(word_loadings[1])

x_label = (f"PC1 ({explained[0]:.1%})   ←  {' · '.join(pc1_neg)}"
           f"   |   {' · '.join(pc1_pos)}  →")
y_label = (f"PC2 ({explained[1]:.1%})   ↓  {' · '.join(pc2_neg)}"
           f"   |   {' · '.join(pc2_pos)}  ↑")

print(f"\nPC1 negative end: {pc1_neg}")
print(f"PC1 positive end: {pc1_pos}")
print(f"PC2 negative end: {pc2_neg}")
print(f"PC2 positive end: {pc2_pos}\n")

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

# ── biplot arrows: top-loading words as vectors ───────────────────────────────
# Show the N_BIPLOT words with the largest L2 norm in PC1-PC2 space.
N_BIPLOT = 12
components_2d = word_loadings   # (2, n_vocab)
norms = np.linalg.norm(components_2d.T, axis=1)
top_idx = np.argsort(norms)[-N_BIPLOT:]

scale = 0.3 * max(coords[:, 0].max() - coords[:, 0].min(),
                  coords[:, 1].max() - coords[:, 1].min())
arrow_xy = components_2d.T[top_idx] / (norms[top_idx].max()) * scale
arrow_words = feature_names[top_idx]

# ── plot ─────────────────────────────────────────────────────────────────────
current_mode = ["outcome"]
show_biplot  = [True]

fig, ax = plt.subplots(figsize=(14, 9))
plt.subplots_adjust(bottom=0.08)


def get_colors(mode):
    palette = PALETTES[mode]
    return [palette.get(t, palette.get("other", "#9E9E9E")) for t in tags[mode]]


sc = ax.scatter(coords[:, 0], coords[:, 1],
                c=get_colors("outcome"), s=40, alpha=0.75,
                linewidths=0.3, edgecolors="white", zorder=3)

ax.set_title("TF-IDF → LSA(50) → PCA(2)  (1-7 recolour | b toggle biplot arrows)",
             fontsize=11)
ax.set_xlabel(x_label, fontsize=8)
ax.set_ylabel(y_label, fontsize=8)

# ── star markers for key companies ───────────────────────────────────────────
for keyword, (display_label, color) in HIGHLIGHT_COMPANIES.items():
    idxs = [i for i, m in enumerate(metadatas)
            if keyword in m.get("filename", "").lower()
            or keyword in m.get("job_folder", "").lower()]
    if not idxs:
        continue
    pts = coords[idxs]
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    ax.scatter(cx, cy, marker="*", s=380, color=color,
               zorder=10, edgecolors="black", linewidths=0.6)
    ax.text(cx, cy, f" {display_label}", fontsize=8, fontweight="bold",
            color=color, zorder=11,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, linewidth=0))

# ── biplot arrows (word loadings) ─────────────────────────────────────────────
biplot_artists = []

def draw_biplot():
    for a in biplot_artists:
        a.remove()
    biplot_artists.clear()

    cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
    for (dx, dy), word in zip(arrow_xy, arrow_words):
        arr = ax.annotate(
            "", xy=(cx + dx, cy + dy), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9),
            zorder=6,
        )
        txt = ax.text(cx + dx * 1.08, cy + dy * 1.08, word,
                      fontsize=6.5, color="#333333", ha="center", va="center",
                      zorder=7)
        biplot_artists.extend([arr, txt])
    fig.canvas.draw_idle()


def clear_biplot():
    for a in biplot_artists:
        a.remove()
    biplot_artists.clear()
    fig.canvas.draw_idle()


draw_biplot()


def refresh(mode):
    colors = get_colors(mode)
    sc.set_facecolors(colors)
    palette = PALETTES[mode]
    ax.legend(
        handles=[mpatches.Patch(color=c, label=k) for k, c in palette.items()],
        title=mode.capitalize(), loc="upper right", fontsize=8,
    )
    ax.set_title(
        f"TF-IDF → LSA(50) → PCA(2) — coloured by {mode}  (1-7 recolour | b toggle biplot arrows)",
        fontsize=11,
    )
    fig.canvas.draw_idle()


refresh("outcome")

# ── hover tooltip ─────────────────────────────────────────────────────────────
annot = ax.annotate(
    "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
    bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9), fontsize=8,
)
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
        out = os.path.join(config._PROJECT_DIR, "data", "review", f"pca_{mode}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    elif event.key == "b":
        if show_biplot[0]:
            clear_biplot()
            show_biplot[0] = False
        else:
            draw_biplot()
            show_biplot[0] = True


fig.canvas.mpl_connect("motion_notify_event", on_hover)
fig.canvas.mpl_connect("key_press_event",     on_key)

# ── save default PNG and show ─────────────────────────────────────────────────
out_png = os.path.join(config._PROJECT_DIR, "data", "review", "pca_tfidf.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved static plot: {out_png}")
print("Keys: 1=outcome  2=seniority  3=remote  4=domain  5=role  6=special  7=company  b=toggle biplot arrows")

# Always open the PNG in Preview so the user sees it even without a live window.
subprocess.Popen(["open", out_png])

if _CAN_SHOW:
    plt.tight_layout()
    plt.show()
else:
    print("Interactive window not available (Agg backend). PNG opened in Preview.")
    print("For interactive mode run:  conda activate job-rag && python analysis/pca_plot.py")
