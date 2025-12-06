# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from scipy.spatial.distance import jensenshannon


# %%
# Load both datasets, results and robustness 

#set correct path for repo, e.g. where you saved the ita-parl-health folder
BASE_PATH= "C:/insert repo path here/numerical analysis"
# Load both datasets once

data_path_main = BASE_PATH.join('final_health_subset_bert.csv')
df_main = pd.read_csv(data_path_main, dtype=str, low_memory=False)

data_path_lda = BASE_PATH.join('final_health_subset_lda.csv')
df_robust = pd.read_csv(data_path_lda, dtype=str, low_memory=False)


# Choose dataset: "main" for main analyis or "lda" for robustness check
DATASET = "main" \

if DATASET == "main":
    df_health = df_main
elif DATASET == "robust":
    df_health = df_robust
else:
    raise ValueError("Unknown DATASET")

# %%
df = df_health.copy()


# Keep only Left and Right
df = df[df["bloc_combined"].isin(["Left", "Right"])].copy()


# %%
vocab_set = set()

for txt in df["text_repro"]:
    vocab_set.update(txt.split())

vocab = sorted(vocab_set)
vocab_index = {w: i for i, w in enumerate(vocab)}

print("Vocabulary size:", len(vocab))


# %%
def word_distribution(texts):
    counts = Counter()
    for txt in texts:
        counts.update(txt.split())
    total = sum(counts.values())
    return counts, total


# %%
quarters_sorted = sorted(df["quarter"].unique())
results = []

for q in quarters_sorted:
    q_df = df[df["quarter"] == q]

    left_texts = q_df[q_df["bloc_combined"] == "Left"]["text_repro"].tolist()
    right_texts = q_df[q_df["bloc_combined"] == "Right"]["text_repro"].tolist()

    if len(left_texts) == 0 or len(right_texts) == 0:
        continue

    left_counts, left_total = word_distribution(left_texts)
    right_counts, right_total = word_distribution(right_texts)

    if left_total == 0 or right_total == 0:
        continue

    p_left = np.zeros(len(vocab))
    p_right = np.zeros(len(vocab))

    for w, c in left_counts.items():
        idx = vocab_index.get(w)
        if idx is not None:
            p_left[idx] = c / left_total

    for w, c in right_counts.items():
        idx = vocab_index.get(w)
        if idx is not None:
            p_right[idx] = c / right_total

    jsd_val = jensenshannon(p_left, p_right, base=2)

    results.append({
        "quarter": q,
        "jsd_all_words": jsd_val,
        "left_n": len(left_texts),
        "right_n": len(right_texts),
        "left_tokens": left_total,
        "right_tokens": right_total
    })

results_jsd_all = pd.DataFrame(results).sort_values("quarter")
results_jsd_all


# %%
import matplotlib.pyplot as plt

# Create figure + axes
fig, ax = plt.subplots(figsize=(12, 5))

# Make plot border thinner
for spine in ax.spines.values():
    spine.set_linewidth(0.6)


x = range(len(results_jsd_all))
quarters = results_jsd_all["quarter"].tolist()

# --- LINE ---
ax.plot(
    x,
    results_jsd_all["jsd_all_words"],
    marker="o",
    color="#5c2784"   # viridis purple
)

# --- COVID LINE ---
covid_q = "2020q1"
if covid_q in quarters:
    covid_idx = quarters.index(covid_q)
    ax.axvline(covid_idx, color="#1a1a1a", linestyle="--", alpha=0.6)

# --- LABEL POSITIONS ---
label_y = 0.76
text_style = dict(fontsize=12, fontweight="bold")

# Pre-COVID label
pre_left = "2019q1"
pre_right = "2019q2"
if pre_left in quarters and pre_right in quarters:
    pre_x = (quarters.index(pre_left) + quarters.index(pre_right)) / 2
    ax.text(pre_x, label_y, "Pre COVID-19",
            ha="center", va="bottom", **text_style)

# COVID label
post_left = "2021q2"
post_right = "2021q3"
if post_left in quarters and post_right in quarters:
    post_x = (quarters.index(post_left) + quarters.index(post_right)) / 2
    ax.text(post_x, label_y, "COVID-19",
            ha="center", va="bottom", **text_style)

# --- AXES ---
ax.set_ylim(0.4, 0.8)

ax.set_xticks(list(x))
ax.set_xticklabels(quarters, rotation=90, fontsize=12)

ax.set_ylabel("Jensen–Shannon Distance (All Words)", fontsize=14, fontweight="bold")
ax.set_xlabel("Year and Quarter", fontsize=14, fontweight="bold")

ax.grid(alpha=0.3)

fig.tight_layout()
plt.show()

from utils import save_plot


#Export the plot
save_plot(fig, f"JSD Words {DATASET}")


# %%

# Equalize total number of speeches per quarter (keep Left/Right proportions)

def build_equalized_quarter_df(df, random_state=42):
    """
    Downsample speeches so that each quarter has the same total number of speeches,
    while preserving the Left/Right proportion within that quarter.
    Works on a df that already contains only Left/Right blocs.
    """
    df_lr = df.copy()

    # Count speeches per quarter and bloc
    counts_qb = (
        df_lr.groupby(["quarter", "bloc_combined"])
             .size()
             .unstack(fill_value=0)
    )

    # Only keep quarters where both Left and Right are present
    eligible_quarters = counts_qb.index[
        (counts_qb.get("Left", 0) > 0) & (counts_qb.get("Right", 0) > 0)
    ]

    # Total speeches per eligible quarter
    totals = counts_qb.loc[eligible_quarters].sum(axis=1)

    # Target N = smallest total number of speeches among eligible quarters
    target_n = int(totals.min())
    print(f"Target per-quarter N (equalized) = {target_n}")

    rng = np.random.RandomState(random_state)
    sampled_frames = []

    for q in sorted(eligible_quarters):
        qdf = df_lr[df_lr["quarter"] == q]

        left_q = qdf[qdf["bloc_combined"] == "Left"]
        right_q = qdf[qdf["bloc_combined"] == "Right"]

        n_left = len(left_q)
        n_right = len(right_q)
        total_q = n_left + n_right

        if total_q < target_n:
            # Should not happen if target_n is based on min(totals), but just in case:
            continue

        # Proportional allocation to keep bloc mix
        target_left = int(round(target_n * (n_left / total_q)))
        target_right = target_n - target_left  # ensure sum == target_n

        left_s = left_q.sample(n=target_left, random_state=random_state, replace=False)
        right_s = right_q.sample(n=target_right, random_state=random_state, replace=False)

        sampled_frames.append(left_s)
        sampled_frames.append(right_s)

    if not sampled_frames:
        raise ValueError("Downsampling produced an empty dataset. Check data availability.")

    df_equal = pd.concat(sampled_frames, ignore_index=True)

    print("Equalized quarter-by-quarter sample sizes (Left/Right):")
    print(
        df_equal.groupby(["quarter", "bloc_combined"])
                .size()
                .unstack(fill_value=0)
                .head(10)
    )

    return df_equal

# Build equalized sample
df_equal = build_equalized_quarter_df(df, random_state=42)


# %%

# Recompute JSD using equalized per-quarter sample

quarters_sorted_eq = sorted(df_equal["quarter"].unique())
results_eq = []

for q in quarters_sorted_eq:
    q_df = df_equal[df_equal["quarter"] == q]

    left_texts = q_df[q_df["bloc_combined"] == "Left"]["text_repro"].tolist()
    right_texts = q_df[q_df["bloc_combined"] == "Right"]["text_repro"].tolist()

    if len(left_texts) == 0 or len(right_texts) == 0:
        continue

    left_counts, left_total = word_distribution(left_texts)
    right_counts, right_total = word_distribution(right_texts)

    if left_total == 0 or right_total == 0:
        continue

    p_left = np.zeros(len(vocab))
    p_right = np.zeros(len(vocab))

    for w, c in left_counts.items():
        idx = vocab_index.get(w)
        if idx is not None:
            p_left[idx] = c / left_total

    for w, c in right_counts.items():
        idx = vocab_index.get(w)
        if idx is not None:
            p_right[idx] = c / right_total

    jsd_val = jensenshannon(p_left, p_right, base=2)

    results_eq.append({
        "quarter": q,
        "jsd_all_words_equal": jsd_val,
        "left_n_equal": len(left_texts),
        "right_n_equal": len(right_texts),
        "left_tokens_equal": left_total,
        "right_tokens_equal": right_total
    })

results_jsd_equal = pd.DataFrame(results_eq).sort_values("quarter")
results_jsd_equal


# %%
import matplotlib.pyplot as plt

# -------------------------------
# Create figure + axis
# -------------------------------
fig, ax = plt.subplots(figsize=(12, 5))

# Align quarters
quarters_orig = results_jsd_all["quarter"].tolist()
quarters_eq   = results_jsd_equal["quarter"].tolist()
common_q = sorted(set(quarters_orig) & set(quarters_eq))

# Build aligned series
x = list(range(len(common_q)))
jsd_orig = [results_jsd_all.set_index("quarter").loc[q, "jsd_all_words"] for q in common_q]
jsd_eq   = [results_jsd_equal.set_index("quarter").loc[q, "jsd_all_words_equal"] for q in common_q]

# Colors
purple_orig = "#5c2784"   # original line
green_eq    = "#35b779"   # equalized

# -------------------------------
# Lines
# -------------------------------
ax.plot(x, jsd_orig, marker="o", color=purple_orig, linewidth=2.0, alpha=0.9)
ax.plot(x, jsd_eq,   marker="o", color=green_eq,   linewidth=2.0, alpha=0.9)

# -------------------------------
# COVID vertical line
# -------------------------------
covid_q = "2020q1"
if covid_q in common_q:
    covid_idx = common_q.index(covid_q)
    ax.axvline(covid_idx, color="#1a1a1a", linestyle="--", alpha=0.6)

# -------------------------------
# Period labels
# -------------------------------
label_y = 0.95
text_style = dict(fontsize=12, fontweight="bold")

pre_left = "2019q1"
pre_right = "2019q2"
if pre_left in common_q and pre_right in common_q:
    pre_x = (common_q.index(pre_left) + common_q.index(pre_right)) / 2
    ax.text(pre_x, label_y, "Pre COVID-19", ha="center", va="bottom", **text_style)

post_left = "2021q2"
post_right = "2021q3"
if post_left in common_q and post_right in common_q:
    post_x = (common_q.index(post_left) + common_q.index(post_right)) / 2
    ax.text(post_x, label_y, "COVID-19", ha="center", va="bottom", **text_style)

# -------------------------------
# Axes
# -------------------------------
ax.set_ylim(0.4, 1)
ax.set_xticks(x)
ax.set_xticklabels(common_q, rotation=90, fontsize=11)
ax.set_ylabel("Jensen–Shannon Distance (All Words)", fontsize=12, fontweight="bold")
ax.set_xlabel("Year and Quarter", fontsize=12, fontweight="bold")
ax.grid(alpha=0.3)

# -------------------------------
# Inline labels (end of lines)
# -------------------------------
xpos = len(common_q) - 1
shift_x = 0.3  

# Green ABOVE
ax.text(
    xpos - shift_x,
    jsd_eq[-1] + 0.02,
    "Equalized",
    color=green_eq,
    fontsize=12,
    fontweight="bold",
    va="bottom"
)

# Purple BELOW
ax.text(
    xpos - shift_x,
    jsd_orig[-1] - 0.02,
    "Original",
    color=purple_orig,
    fontsize=12,
    fontweight="bold",
    va="top"
)

fig.tight_layout()
plt.show()

# -------------------------------
# Save plot
# -------------------------------
save_plot(fig, f"JSD Words Ori vs EQ {DATASET}")



