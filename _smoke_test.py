import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
import config

client = chromadb.PersistentClient(path=config.DB_PATH)
collection = client.get_collection("job_descriptions")
result = collection.get(include=["metadatas", "documents"])
metadatas = result["metadatas"]
documents = result["documents"]
docs_clean = [d if d else "" for d in documents]
filenames = [m.get("filename", "") for m in metadatas]
combined = [f + " " + d for f, d in zip(filenames, docs_clean)]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True,
    strip_accents="unicode", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b",
)
tfidf_matrix = vectorizer.fit_transform(combined)
print(f"TF-IDF shape: {tfidf_matrix.shape}")

reducer = UMAP(n_components=2, n_neighbors=12, min_dist=0.1, metric="cosine", random_state=42)
coords = reducer.fit_transform(tfidf_matrix)
print(f"UMAP coords shape: {coords.shape}")
print("SMOKE TEST PASSED")
