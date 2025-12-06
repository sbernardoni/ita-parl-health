# %%
# Import relevant packages
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pprint import pprint
import spacy


# %%


# %%
# Load data 

#set correct working directory 

CSV_PATH = "C:/insert path here/camera_2013_2022_textcleaned_simple.csv"

try:
    df_clean = pd.read_csv(CSV_PATH, encoding="utf-8")
except UnicodeDecodeError:
    df_clean = pd.read_csv(CSV_PATH, encoding="latin-1")

print("Loaded:", CSV_PATH, "rows:", len(df))

display(df_clean.columns.tolist())
df_clean.head(5)

# %%
import spacy

nlp = spacy.load("it_core_news_sm", disable=["ner", "parser"])

# use spaCy's Italian stopword list (already included in the model)
stop_words = {w.lower() for w in nlp.Defaults.stop_words}

# extra artifacts from apostrophe/punctuation stripping
apostrophe_leftovers = {"l", "d", "c", "all", "dell", "nell", "sull", "coll"}
stop_words |= apostrophe_leftovers

def reprocess_texts(texts, batch_size=1000, n_process=8):
    docs = nlp.pipe((t.lower() for t in texts), batch_size=batch_size, n_process=n_process)
    sw = stop_words
    out = []
    for doc in docs:
        lemmas = [
            t.lemma_.lower()
            for t in doc
            if (
                t.is_alpha and
                len(t.lemma_) > 1 and
                t.lemma_.lower() not in sw and
                t.text.lower() not in sw
            )
        ]
        out.append(" ".join(lemmas))
    return out

df_clean["text_repro"] = reprocess_texts(
    df_clean["text_clean"].astype(str).tolist(),
    batch_size=1000,
    n_process=8
)



          


# %%
print(stop_words)

# %%
#Check if reprocessing worked

df_clean[["text_clean", "text_repro"]].sample(5, random_state=1)



# %%
# Create dictionary and corpus for LDA

from gensim import corpora

# Tokenize the cleaned text
texts = [s.split() for s in df_clean["text_repro"].astype(str)]

# Create the dictionary
dictionary = corpora.Dictionary(texts)

# Filter extremes
#    - remove words that appear in fewer than 10 docs
#    - remove words that appear in more than 50% of docs
#    - keep up to 100k unique tokens (a safety cap)
dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)

# Create the bag-of-words corpus
corpus = [dictionary.doc2bow(t) for t in texts]

# Check dictionary size
print(f"Number of unique tokens in dictionary: {len(dictionary):,}")




# %%
#Create function to run LDA and compute coherence using multicoreLDA for speed. We will use this to identify optimal number of topics.
#For final LDA, we will use single core with optimal number of topic so our results are 100% reproducible across maschines.

from gensim import models
from gensim.models.coherencemodel import CoherenceModel



def runLDA(k, corpus, dictionary, texts, workers=8, passes=10, iterations=100, topn=10):
    # Train one LDA model
    lda = models.LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        alpha="symmetric",
        eta="symmetric",
        passes=passes,
        iterations=iterations,
        workers=workers,
        random_state=42
    )

    # Compute coherence (C_v)
    cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence="c_v")
    coherence = cm.get_coherence()

    # Extract top words per topic
    topics = []
    for topic_id, terms in lda.show_topics(num_topics=k, num_words=topn, formatted=False):
        topic_words = ", ".join([w for w, _ in terms])
        topics.append({"Topic": topic_id, "Top_Words": topic_words})

    topics_df = pd.DataFrame(topics)

    return lda, coherence, topics_df



# %%
if False:
    #Run LDA function for k=10 topics to chech if it works and how computationally intensive it is


    LDA10, coh10, topics10 = runLDA(
        k=10,
        corpus=corpus,
        dictionary=dictionary,
        texts=texts,
        workers=8,
        passes=10,
        iterations=100
    )

    # Print coherence score
    print(f"Coherence for 10 topics: {coh10:.4f}")

    # View first few topics
    print(topics10.head())




# %%
if False: 
    #Find Optimal Number of Topics (K)

    k_values = list(range(10, 31, 5))  # 10, 15, 20, 25, 30
    results = []

    for k in k_values:
        print(f"Running LDA for k={k}...")
        lda_model, coherence, topics_df = runLDA(
            k=k,
            corpus=corpus,
            dictionary=dictionary,
            texts=texts,
            workers=8,
            passes=10,
            iterations=100,
            topn=10
        )
        results.append((k, coherence))
        print(f"Coherence for k={k}: {coherence:.4f}")

    # convert results to DataFrame for easier viewing
    coh_results = pd.DataFrame(results, columns=["Num_Topics", "Coherence"])
    print("\nCoherence summary:")
    print(coh_results)

# %%
# For the final LDA, we use single core (LdaModel) with symmetric priors for reproducibility.


def singleLDA(k, corpus, dictionary, texts, workers=8, passes=10, iterations=100, topn=10):
    lda = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        alpha="symmetric",
        eta="symmetric",
        passes=passes,
        iterations=iterations,
        random_state=42,
        update_every=1,
        eval_every=None
    )

    # Extract top words per topic
    topics = []
    for topic_id, terms in lda.show_topics(num_topics=k, num_words=topn, formatted=False):
        topic_words = ", ".join([w for w, _ in terms])
        topics.append({"Topic": topic_id, "Top_Words": topic_words})

    topics_df = pd.DataFrame(topics)
    return lda, topics_df





# %%
# Run LDA function for k=20 topics (final, reproducible)
LDA20, topics20 = singleLDA(
    k=20,
    corpus=corpus,
    dictionary=dictionary,
    texts=texts,
    workers=8,
    passes=10,
    iterations=100
)

# %%

# View first few topics
print(topics20.head())

# Save 20 topics as CSV
topics20.to_csv("Results/LDA Topic/LDA_topics_k20.csv", index=False, encoding="utf-8")

# %%
#Create Document-Topic Matrix for k=20 model
# ---Create document-topic matrix ---
num_topics = 20  
doc_topic = np.zeros((len(corpus), num_topics))

for i, bow in enumerate(corpus):
    for k, p in LDA20.get_document_topics(bow, minimum_probability=0.0):
        doc_topic[i, k] = p

# attach probabilities to df_clean
for k in range(num_topics):
    df_clean[f"topic_{k}"] = doc_topic[:, k]







# %%
# Identify health-related speeches (topics 10, 11, and 13)

health_topics = [10,11,13]  # health, welfare, emergency-related topics
threshold = 0.3         

# Compute combined probability of all health topics
df_clean["p_health"] = doc_topic[:, health_topics].sum(axis=1)

# Filter speeches that are strongly health-related
df_health = df_clean[df_clean["p_health"] >= threshold].copy()



# %%


# Compare distributions of combined probabilities
df_clean["p_health"] = doc_topic[:, [10, 11, 13]].sum(axis=1)

# Look at percentile cutoffs to choose threshold empirically
percentiles = np.percentile(df_clean["p_health"], [50, 75, 85, 90, 95, 97.5, 99])
print(dict(zip([50, 75, 85, 90, 95, 97.5, 99], percentiles)))

#With our threshold of 0.35, we capture speeches around the 95th percentile. 


# %%
# Quick checks
print(f"Total speeches: {len(df_clean)}")
print(f"Health speeches (topic-based): {len(df_health)}")


print("\nDistribution by year:")
print(df_health["year"].value_counts().sort_index())


# percentage of speeches per year that are health-related 
year_totals = df_clean["year"].value_counts().sort_index()
year_health = df_health["year"].value_counts().sort_index()

# align indices to avoid KeyErrors
year_share = (year_health / year_totals * 100).fillna(0).round(2)

print("\nPercentage of speeches per year that are health-related (%):")
print(year_share)


# %%
#Add flag to main dataset

# Make sure df_health already exists as your filtered dataset
df_clean["if_health"] = df_clean.index.isin(df_health.index)


# %%
import pandas as pd

# Combine Center and M5S into one bloc
df_clean["bloc_combined"] = df_clean["side3"].replace({"Center": "Center+M5S", "M5S": "Center+M5S"})
df_health["bloc_combined"] = df_health["side3"].replace({"Center": "Center+M5S", "M5S": "Center+M5S"})

# Compute share of health speeches per quarter per bloc
table = (
    df_clean
    .groupby(["bloc_combined", "quarter"])["if_health"]
    .mean()
    .reset_index()
    .rename(columns={"if_health": "share_health"})
)

# Optional: express as percentage instead of proportion
table["share_health"] = (table["share_health"] * 100).round(2)

# Sort nicely
table = table.sort_values(["bloc_combined", "quarter"])

print(table.head(10))


# %%
#Plot the Results

import matplotlib.pyplot as plt

# Define the figure with a descriptive name
fig_health_share_bloc_quarter, ax = plt.subplots(figsize=(10,6))

# Plot data
for bloc in table["bloc_combined"].unique():
    subset = table[table["bloc_combined"] == bloc]
    ax.plot(subset["quarter"], subset["share_health"], marker="o", label=bloc)

# Labels, title, legend
ax.set_title("Share of Health-Related Speeches by Bloc Over Time")
ax.set_xlabel("Quarter")
ax.set_ylabel("Share of speeches about health (%)")
ax.set_xticklabels(table["quarter"].unique(), rotation=45, ha="right")
ax.legend(title="Bloc")

fig_health_share_bloc_quarter.tight_layout()

# Show in notebook
plt.show()


# %%


# Compute overall share of health speeches per quarter
health_share_overall = (
    df_clean
    .groupby("quarter")["if_health"]
    .mean()
    .reset_index()
    .rename(columns={"if_health": "share_health"})
)

# Convert to percentage
health_share_overall["share_health"] = (health_share_overall["share_health"] * 100).round(2)

# Define figure
fig_health_share_overall, ax = plt.subplots(figsize=(10,6))

# Plot
ax.plot(health_share_overall["quarter"], health_share_overall["share_health"], marker="o", color="steelblue")

# Labels and title
ax.set_title("Overall Share of Health-Related Speeches Over Time")
ax.set_xlabel("Quarter")
ax.set_ylabel("Share of speeches about health (%)")
ax.set_xticklabels(health_share_overall["quarter"], rotation=45, ha="right")

fig_health_share_overall.tight_layout()

plt.show()


# %%
import matplotlib.pyplot as plt

# Count total speeches per quarter per bloc
speeches_per_quarter_bloc = (
    df_clean
    .groupby(["bloc_combined", "quarter"])
    .size()
    .reset_index(name="n_speeches")
    .sort_values(["bloc_combined", "quarter"])
)

print(speeches_per_quarter_bloc.head(10))


# Define figure
fig_speeches_per_quarter_bloc, ax = plt.subplots(figsize=(10,6))

# Plot one line per bloc
for bloc in speeches_per_quarter_bloc["bloc_combined"].unique():
    subset = speeches_per_quarter_bloc[speeches_per_quarter_bloc["bloc_combined"] == bloc]
    ax.plot(subset["quarter"], subset["n_speeches"], marker="o", label=bloc)

# Labels and formatting
ax.set_title("Number of Speeches per Quarter by Bloc")
ax.set_xlabel("Quarter")
ax.set_ylabel("Total number of speeches")
ax.set_xticklabels(speeches_per_quarter_bloc["quarter"].unique(), rotation=45, ha="right")
ax.legend(title="Bloc")

fig_speeches_per_quarter_bloc.tight_layout()
plt.show()


# %%



# %%
# Export df_health as CSV, save as lds_subset_health_topics
df_health.to_csv("C:/insert path here/lda_subset_health_topics.csv")





