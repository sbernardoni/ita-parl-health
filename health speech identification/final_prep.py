# %%
# Import relevant packages
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pprint import pprint
import spacy
import re


# %%
#This Notebook is the final cleaning step of the two different subsets (one based on BERT one based on LDA). I want to make sure that they have the same structure (variable name)


import os
import pandas as pd

# Adjust file path to the script
BASE_PATH = "C:/insert path here/health speech identification"

def load_csv(fname):
    path = os.path.join(BASE_PATH, fname)
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

df_health = load_csv("bert_subset_health_topics.csv")
df_health_lda = load_csv("lda_subset_health_topics.csv")





# %%
#Drop Columns that are not needed for analysis

#For Bert Subset
cols_to_drop_bert = [c for c in df_health.columns 
                if c.startswith("bert") or c.startswith("is")]

df_health = df_health.drop(columns=cols_to_drop_bert)

#For LDA Subset

# Columns that start with "topic"
topic_cols = [c for c in df_health_lda.columns if c.startswith("topic")]

# Explicit columns to remove
explicit_cols = ["p_health"]

# Combine
cols_to_drop_lda = topic_cols + explicit_cols


df_health_lda = df_health_lda.drop(columns=cols_to_drop_lda)



# %%
#Add the text_repro column  to bert subset



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

df_health["text_repro"] = reprocess_texts(
    df_health["text_clean"].astype(str).tolist(),
    batch_size=1000,
    n_process=8
)



          


# %%
#Check if reprocessing worked

df_health[["text_clean", "text_repro"]].sample(5, random_state=1)


# %%
df_health["bloc_combined"] = (
    df_health["side3"]
    .replace({"Center": "Center+M5S", "M5S": "Center+M5S"})
)


# %%
#Count the number of speeches per quarter and bloc_combined for both datasets 

# --- For df_health ---
counts_health = (
    df_health
    .groupby(["quarter", "bloc_combined"])
    .size()
    .unstack(fill_value=0)
)

# --- For df_health_lda ---
counts_health_lda = (
    df_health_lda
    .groupby(["quarter", "bloc_combined"])
    .size()
    .unstack(fill_value=0)
)


# %%
#Remove 2018q1 and q2. The Government was not formed yet and hence the number of speeches is very low

quarters_to_remove = ["2018q1", "2018q2"]

df_health = df_health[~df_health["quarter"].isin(quarters_to_remove)]
df_health_lda = df_health_lda[~df_health_lda["quarter"].isin(quarters_to_remove)]


# %%
#Rerun count to see if removal worked

# --- For df_health ---
counts_health = (
    df_health
    .groupby(["quarter", "bloc_combined"])
    .size()
    .unstack(fill_value=0)
)

# --- For df_health_lda ---
counts_health_lda = (
    df_health_lda
    .groupby(["quarter", "bloc_combined"])
    .size()
    .unstack(fill_value=0)
)


# %%
# Export final cleaned datasets

# Try UTF-8 first, otherwise latin-1

DATA_DIR = "C:/insert path here/health speech identification"
try:
    df_health.to_csv(os.path.join(DATA_DIR, "final_health_subset_bert.csv"),
                     index=False, encoding="utf-8")
except UnicodeEncodeError:
    df_health.to_csv(os.path.join(DATA_DIR, "final_health_subset_bert.csv"),
                     index=False, encoding="latin-1")

try:
    df_health_lda.to_csv(os.path.join(DATA_DIR, "final_health_subset_lda.csv"),
                         index=False, encoding="utf-8")
except UnicodeEncodeError:
    df_health_lda.to_csv(os.path.join(DATA_DIR, "final_health_subset_lda.csv"),
                         index=False, encoding="latin-1")



