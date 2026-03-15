import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chromadb, numpy as np, warnings
warnings.filterwarnings("ignore")
from umap import UMAP
import config

client = chromadb.PersistentClient(path=config.DB_PATH)
col = client.get_collection("job_descriptions")
result = col.get(include=["embeddings","metadatas"])
embeddings = np.array(result["embeddings"])
metadatas  = result["metadatas"]

reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
coords = reducer.fit_transform(embeddings)

keywords = ["astrazeneca","amgen","miltenyi","uthr","united therapeutics","sandboxaq","guardant","natera"]
print("%-4s  %7s  %7s  %s" % ("idx","UMAP1","UMAP2","filename"))
print("-"*80)
for i, m in enumerate(metadatas):
    fn     = m.get("filename","").lower()
    folder = m.get("job_folder","").lower()
    if any(k in fn or k in folder for k in keywords):
        print("%-4d  %7.2f  %7.2f  %s" % (i, coords[i,0], coords[i,1], m["filename"]))
