# If needed:
# !pip install -U sentence-transformers

import os, unicodedata, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#finding health topics


# ---------- pick assignments / base df ----------
if "results" in locals() and "bertopic_topic" in results.columns:
    assign_df = results.copy()
    assign_vec = assign_df["bertopic_topic"].to_numpy()
else:
    assign_df = df.copy()
    assign_df["bertopic_topic"] = topics  # assumes `topics` exists
    assign_vec = assign_df["bertopic_topic"].to_numpy()

topics_info = topic_model.get_topic_info()
topic_ids = [t for t in topics_info["Topic"].tolist() if t != -1]

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

# ========= A) KEYWORD method (prefix stems) =========
HEALTH_KEYWORDS = {
    "salute","sanita","ospedal","medic", "infermier",
    "vaccin","pandem","epidemi","malattia","pazient","terapia","farmac",
    "ssn","asl","sanitari","prevenzion","diagnosi","ricover",
    "prenotazion","sanitaria","covid","tumor","oncolog","psichiatr","psicolog",
    "obesita","materni","nascit","servizio","sanitario_nazionale", "green_pass", "virus"
}

def topic_has_health_words(tid, top_k=20):
    words = [w for w,_ in topic_model.get_topic(tid)[:top_k]]
    norm = [strip_accents(w.lower()) for w in words]
    for w in norm:
        for kw in HEALTH_KEYWORDS:
            if w.startswith(kw):
                return True
    return False

health_by_kw = [tid for tid in topic_ids if topic_has_health_words(tid, top_k=20)]

# ========= B) SEMANTIC method with UMBERTO (CPU ok) =========
st_model = SentenceTransformer("musixmatch/umberto-commoncrawl-cased-v1")

query_text = "sanità salute ospedale medico paziente medicina vaccino epidemia prevenzione diagnosi terapia servizio sanitario nazionale pandemia malattia farmaco ricovero tumore virus"
q_vec = st_model.encode(query_text, normalize_embeddings=True)

topic_order, topic_phrases = [], []
for tid in topic_ids:
    top_words = [w for w,_ in topic_model.get_topic(tid)[:10]]
    phrase = " ".join(top_words)            # no 'query:'/'passage:' prefixes needed for UmBERTo
    topic_order.append(tid)
    topic_phrases.append(phrase)

topic_vecs = st_model.encode(topic_phrases, normalize_embeddings=True)
sims = cosine_similarity(topic_vecs, q_vec.reshape(1, -1)).ravel()

# Robust cutoff: keep topics above the max of an absolute floor and a percentile
ABS_FLOOR = 0.35
PCTL = 85
thr = max(ABS_FLOOR, float(np.percentile(sims, PCTL)))
health_by_sem = [topic_order[i] for i, s in enumerate(sims) if s >= thr]

# ========= Combine & label =========
health_topics_union = sorted(set(health_by_kw) | set(health_by_sem))
health_topics_intersection = sorted(set(health_by_kw) & set(health_by_sem))

# Per-topic table (for auditing)
rows = []
for tid in topic_ids:
    words = [w for w,_ in topic_model.get_topic(tid)[:10]]
    sim = float(sims[topic_order.index(tid)])
    rows.append({
        "topic_id": tid,
        "top_words": ", ".join(words),
        "umberto_sim": sim,
        "health_by_keywords": tid in health_by_kw,
        "health_by_semantic": tid in health_by_sem,
        "health_union": tid in health_topics_union,
        "health_intersection": tid in health_topics_intersection
    })
topic_labels_df = pd.DataFrame(rows).sort_values("umberto_sim", ascending=False)

# Per-document flags (use UNION by default; switch to intersection for stricter labeling)
assign_df["is_health_by_keywords"] = pd.Series(assign_vec).isin(health_by_kw)
assign_df["is_health_by_semantic_umberto"] = pd.Series(assign_vec).isin(health_by_sem)
assign_df["is_health_union"] = pd.Series(assign_vec).isin(health_topics_union)
assign_df["is_health_intersection"] = pd.Series(assign_vec).isin(health_topics_intersection)

# Save
os.makedirs(OUT_DIR, exist_ok=True)
topic_labels_df.to_csv(os.path.join(OUT_DIR, "health_topics_table_umberto.csv"), index=False, encoding="utf-8")
assign_df.to_csv(os.path.join(OUT_DIR, "doc_assignments_with_health_umberto.csv"), index=False, encoding="utf-8")

print("Health topics (union):", health_topics_union)
print("Health topics (intersection):", health_topics_intersection)
print(f"Saved per-topic table → {os.path.join(OUT_DIR, 'health_topics_table_umberto.csv')}")
print(f"Saved per-document flags → {os.path.join(OUT_DIR, 'doc_assignments_with_health_umberto.csv')}")


#creating a new dataset for the health-related speeches, i.e. health intersection = TRUE
import pandas as pd

# Path to your saved dataset (update if needed)
csv_path_health = "C:/Users/Sara/Documents/ESS/20886 - Foundations of Social Sciences I/health-discourse-it/Results/doc_assignments_with_health_umberto.csv"

# Load dataset
df = pd.read_csv(csv_path_health)

# Ensure the column exists and filter
if "is_health_intersection" in df.columns:
    df_health = df[df["is_health_intersection"] == True].copy()
else:
    raise KeyError("The column 'health_intersection' was not found in the dataset.")

# Inspect results
print(f"Total observations before filtering: {len(df)}")
print(f"Total health-related observations: {len(df_health)}")

# Save filtered dataset
output_path = "C:/Users/Sara/Documents/ESS/20886 - Foundations of Social Sciences I/health-discourse-it/Results final/subset_health_topics.csv"
df_health.to_csv(output_path, index=False)
print(f"Filtered dataset saved to {output_path}")
