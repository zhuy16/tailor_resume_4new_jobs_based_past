"""
Explain UMAP axes: which words/phrases drive the spreading?

Method:
  1. Load embeddings + documents from ChromaDB.
  2. Run UMAP to get 2-D coords (same as explore_embeddings.py).
  3. Build a TF-IDF matrix over the documents.
  4. Correlate each TF-IDF feature with UMAP-1 and UMAP-2 (Spearman).
  5. Print and plot top-N words for each axis direction.

Run: conda run -n job-rag python explain_umap.py
"""
import os
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
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

import config

TOP_N = 25   # words to show per axis direction

# ── load ─────────────────────────────────────────────────────────────────────
client = chromadb.PersistentClient(path=config.DB_PATH)
col    = client.get_collection("job_descriptions")
result = col.get(include=["embeddings", "metadatas", "documents"])

embeddings = np.array(result["embeddings"])
documents  = result["documents"]
metadatas  = result["metadatas"]
filenames  = [m.get("filename", "") for m in metadatas]
print(f"Loaded {len(embeddings)} documents, dim={embeddings.shape[1]}")

# ── UMAP ─────────────────────────────────────────────────────────────────────
print("Running UMAP...")
reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
               metric="cosine", random_state=42)
coords = reducer.fit_transform(embeddings)   # (n, 2)
print("UMAP done.")

# ── TF-IDF ───────────────────────────────────────────────────────────────────
print("Building TF-IDF matrix...")
# include 1- and 2-grams, ignore very common/rare words
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    stop_words="english",
    min_df=3,          # must appear in ≥3 documents
    max_df=0.85,       # ignore terms in >85% of docs (too generic)
    sublinear_tf=True,
)
tfidf = vectorizer.fit_transform(documents).toarray()  # (n, vocab)
vocab = np.array(vectorizer.get_feature_names_out())
print(f"Vocabulary size: {len(vocab)}")

# ── Correlate TF-IDF features with each UMAP axis ────────────────────────────
print("Correlating TF-IDF features with UMAP axes (this takes ~10 s)...")
r1 = np.array([spearmanr(tfidf[:, j], coords[:, 0]).statistic for j in range(len(vocab))])
r2 = np.array([spearmanr(tfidf[:, j], coords[:, 1]).statistic for j in range(len(vocab))])

def top_words(corr_array, n=TOP_N):
    pos_idx = np.argsort(corr_array)[-n:][::-1]
    neg_idx = np.argsort(corr_array)[:n]
    return (
        list(zip(vocab[pos_idx],  corr_array[pos_idx])),
        list(zip(vocab[neg_idx], corr_array[neg_idx])),
    )

pos1, neg1 = top_words(r1)
pos2, neg2 = top_words(r2)

# ── Print results ─────────────────────────────────────────────────────────────
def print_axis(label, pos, neg):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  → HIGH end (positive direction):")
    for w, r in pos:
        print(f"      {w:<35}  r={r:+.3f}")
    print(f"  → LOW end (negative direction):")
    for w, r in neg:
        print(f"      {w:<35}  r={r:+.3f}")

print_axis("UMAP-1 (left ← → right)", pos1, neg1)
print_axis("UMAP-2 (bottom ↓ ↑ top)", pos2, neg2)

# ── Plot: two horizontal bar charts ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("Top TF-IDF terms correlated with each UMAP axis\n"
             "(Spearman r — what words drive the spreading)", fontsize=13)

def bar_plot(ax, word_score_list, title, color):
    words  = [w for w, _ in word_score_list]
    scores = [s for _, s in word_score_list]
    y = range(len(words))
    ax.barh(y, scores, color=color, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(words, fontsize=9)
    ax.set_xlabel("Spearman r")
    ax.set_title(title, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.invert_yaxis()

bar_plot(axes[0][0], pos1[:TOP_N], "UMAP-1  →  HIGH (positive)", "#1976D2")
bar_plot(axes[0][1], neg1[:TOP_N], "UMAP-1  ←  LOW (negative)",  "#D32F2F")
bar_plot(axes[1][0], pos2[:TOP_N], "UMAP-2  ↑  HIGH (positive)", "#388E3C")
bar_plot(axes[1][1], neg2[:TOP_N], "UMAP-2  ↓  LOW (negative)",  "#F57C00")

plt.tight_layout()

out_png = os.path.join(config._PROJECT_DIR, "data", "review", "umap_axis_words.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_png}")

subprocess.Popen(["open", out_png])

if _CAN_SHOW:
    plt.show()
else:
    print("Interactive window not available (Agg backend). PNG opened in Preview.")
    print("For interactive mode run:  conda activate job-rag && python analysis/explain_umap.py")
