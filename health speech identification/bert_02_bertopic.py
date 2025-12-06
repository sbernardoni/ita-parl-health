EMB_PATH = "C:/insert path here/e5_base_embeddings.npy"

CSV_PATH = "C:/insert path here/camera_2013_2022_textcleaned_simple.csv"
try:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(CSV_PATH, encoding="latin-1")

texts = df[TEXT_COL].astype(str).tolist()
emb = np.load(EMB_PATH)
if emb.shape[0] != len(texts):
    raise ValueError(f"Embeddings ({emb.shape[0]}) ≠ texts ({len(texts)}).")

emb = normalize(emb)

print(f"Loaded embeddings: {emb.shape}")
print(f"Loaded CSV: {len(df)} rows")

#including centroid-based reassignment to have no outliers

# Config

TEXT_COL = "text_clean"
STOP_LANG = "italian"
NGRAM_RANGE = (1, 2)
MIN_DF, MAX_DF = 5, 0.95

UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
HDBSCAN_MIN_CLUSTER_SIZE = 20
HDBSCAN_MIN_SAMPLES = None

OUT_DIR = "C:/insert path here"
os.makedirs(OUT_DIR, exist_ok=True)


# Load data and embeddings

df = pd.read_csv(CSV_PATH)
if TEXT_COL not in df.columns:
    raise ValueError(f"Column '{TEXT_COL}' not found. Found: {list(df.columns)}")

texts = df[TEXT_COL].astype(str).tolist()
emb = np.load(EMB_PATH)
if emb.shape[0] != len(texts):
    raise ValueError(f"Embeddings ({emb.shape[0]}) ≠ texts ({len(texts)}).")

emb = normalize(emb)

# BERTopic setup

vectorizer = CountVectorizer(
    stop_words=italian_stopwords, 
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    max_df=MAX_DF
)

umap_model = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    n_components=UMAP_N_COMPONENTS,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    language=None,
    calculate_probabilities=True,
    verbose=True
)


# Fit BERTopic using embeddings

topics, probs = topic_model.fit_transform(texts, embeddings=emb)


# Topic reassignment (reduce_outliers + centroid fallback)

print(f"Initial outlier rate: {(np.array(topics) == -1).mean():.1%}")

# 1) BERTopic's internal reassignment
try:
    topics = topic_model.reduce_outliers(
        texts,
        topics,
        strategy="c-tf-idf",
        threshold=0.4   
    )
    topic_model.update_topics(texts)
    print(f"After reduce_outliers: {(np.array(topics) == -1).mean():.1%}")
except Exception as e:
    print("reduce_outliers not supported in this version:", e)

# 2) Fallback: force-assign remaining -1 docs to nearest centroid
mask_out = (np.array(topics) == -1)
if mask_out.any():
    valid_topics = sorted(t for t in set(topics) if t != -1)
    topic_to_idx = {t: np.where(np.array(topics) == t)[0] for t in valid_topics}
    centroids = np.vstack([emb[idxs].mean(axis=0) for idxs in topic_to_idx.values()])
    centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)

    sims = cosine_similarity(emb[mask_out], centroids)
    best_idx = sims.argmax(axis=1)
    out_ids = np.where(mask_out)[0]
    for k, doc_id in enumerate(out_ids):
        topics[doc_id] = valid_topics[best_idx[k]]

    print(f"Final outlier rate after centroid reassignment: {(np.array(topics) == -1).mean():.1%}")
else:
    print("No outliers left after reduce_outliers.")


# Save outputs

topics_info = topic_model.get_topic_info()
topics_info.to_csv(os.path.join(OUT_DIR, "topics_overview_reassign.csv"), index=False)

results = df.copy()
results["bertopic_topic"] = topics
results["bertopic_is_outlier"] = (results["bertopic_topic"] == -1)
results["bertopic_prob_max"] = probs.max(1) if probs is not None else np.nan
results.to_csv(os.path.join(OUT_DIR, "doc_assignments_reassign.csv"), index=False)

print(f"Saved topic overview → {OUT_DIR}/topics_overview_reassign.csv")
print(f"Saved document assignments → {OUT_DIR}/doc_assignments_reassign.csv")


# Visualizations
try:
    topic_model.visualize_topics().write_html(os.path.join(OUT_DIR, "viz_topics.html"))
    topic_model.visualize_hierarchy().write_html(os.path.join(OUT_DIR, "viz_hierarchy.html"))
    topic_model.visualize_barchart(top_n_topics=12).write_html(os.path.join(OUT_DIR, "viz_barchart.html"))
    print(f"Visualizations saved in {OUT_DIR}")
except Exception as e:
    print("Visualization error (ok in headless mode):", e)
