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
# Load both datasets, results and robustness 

#set correct path for repo, e.g. where you saved the ita-parl-health folder
BASE_PATH= "C:/insert repo path here/health speech identification"

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
# --- OPTIONAL STEP: remove common health-related terms before analysis ---
#This can be switched on/off here

REMOVE_COMMON_HEALTH_WORDS = False  

import re
from collections import Counter

# Tokenize texts
def tokenize(text):
    if pd.isna(text):
        return []
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if len(t) > 2]

df_health['text_token'] = df_health['text_repro'].apply(tokenize)

if REMOVE_COMMON_HEALTH_WORDS:
    # --- Identify high-frequency (common) topic words ---
    global_counts = Counter([t for toks in df_health['text_token'] for t in toks])
    common_health_words = {w for w, c in global_counts.most_common(200)}  # top 200 frequent words

    # --- Add obvious Italian health-domain terms ---
    extra_health_words = {
        'salute', 'sanità', 'ospedale', 'ospedali', 'medico', 'medici', 'paziente', 'pazienti',
        'infermiere', 'infermieri', 'malattia', 'malattie', 'virus', 'vaccino', 'vaccini',
        'covid', 'pandemia', 'pandemie', 'epidemia', 'epidemie', 'infezione', 'infezioni',
        'contagio', 'contagi', 'cura', 'cure', 'malato', 'malati', 'sanitario', 'sanitari',
        'pubblico', 'pubblica', 'sistema', 'emergenza', 'ospedaliero', 'ospedaliere', 'mascherina', 'mascherine',
        'terapia', 'terapie', 'ricovero', 'ricoveri', 'prevenzione', 'vaccinazione', 'vaccinazioni'
    }

    # Merge and filter
    common_health_words |= extra_health_words

    def filter_common(tokens):
        return [t for t in tokens if t not in common_health_words]

    df_health['text_token'] = df_health['text_token'].apply(filter_common)

    print(f"Common Italian health terms removed (n={len(common_health_words)}).")
else:
    print("Keeping all tokens (no filtering applied).")


# %%
#Create Pre and Post Covid Variable called period

# Helper: parse "2019q4" -> (2019, 4)
def parse_quarter(q_str):
    year_str, qpart = q_str.split('q')
    return int(year_str), int(qpart)

# Maps quarter -> 'pre_covid' / 'post_covid'
def quarter_to_period(q_str):
    year, q = parse_quarter(q_str)
    
    if (year < 2020) or (year == 2020 and q == 1):
        return 'pre_covid'
    else:
        return 'post_covid'

df_health['period'] = df_health['quarter'].apply(quarter_to_period)

df_health[['quarter', 'period']].drop_duplicates().sort_values(['quarter']).head(20)


# %%
from collections import Counter

# Dictionary like:
# { ('Left', 'pre_covid'): Counter(...),
#   ('Left', 'post_covid'): Counter(...),
#   ('Right', 'pre_covid'): Counter(...),
#   ... }
bloc_period_counts = {}

for (bloc, period), subset in df_health.groupby(['bloc_combined', 'period']):
    all_tokens = []
    for tokens in subset['text_token']:
        all_tokens.extend(tokens)
    bloc_period_counts[(bloc, period)] = Counter(all_tokens)


bloc_period_counts.keys()


# %%
import pandas as pd

rows = []

for (bloc, period), counter in bloc_period_counts.items():
    # get top 20 words for this group
    top_words = counter.most_common(20)
    
    for word, count in top_words:
        rows.append({
            'bloc': bloc,
            'period': period,
            'word': word,
            'count': count
        })

# Create the DataFrame
df_top20 = pd.DataFrame(rows)

df_top20.head()


# %%
#This only gives a count, but we want to find out what words are distinctive

#Define function, will be used to run various comparisons

def log_odds_ratio(counts_a, counts_b, alpha=0.01):
    """
    counts_a and counts_b are Counter objects.
    Returns a dataframe with standardized log-odds (z-scores),
    showing which words are distinctive for A vs B.
    """
    vocab = set(counts_a.keys()) | set(counts_b.keys())

    A = np.array([counts_a.get(w, 0) for w in vocab], dtype=float)
    B = np.array([counts_b.get(w, 0) for w in vocab], dtype=float)

    # Add Dirichlet prior
    A_total = A.sum() + alpha * len(vocab)
    B_total = B.sum() + alpha * len(vocab)

    pA = (A + alpha) / A_total
    pB = (B + alpha) / B_total

    # log-odds
    log_odds = np.log(pA / (1 - pA)) - np.log(pB / (1 - pB))

    # standard errors
    se = np.sqrt(1/(A + alpha) + 1/(B + alpha))

    z = log_odds / se

    return pd.DataFrame({
        'word': list(vocab),
        'log_odds': log_odds,
        'z_score': z
    }).sort_values('z_score', ascending=False)


# %%
#left vs right pre-covid

res_left_right_pre = log_odds_ratio(
    bloc_period_counts[('Left', 'pre_covid')],
    bloc_period_counts[('Right', 'pre_covid')]
)



# %%
#left vs right post-covid

res_left_right_post = log_odds_ratio(
    bloc_period_counts[('Left', 'post_covid')],
    bloc_period_counts[('Right', 'post_covid')]
)




# %%
#left pre vs post covid

res_left_shift = log_odds_ratio(
    bloc_period_counts[('Left', 'pre_covid')],
    bloc_period_counts[('Left', 'post_covid')]
)




# %%
#Right pre vs post covid

res_right_shift = log_odds_ratio(
    bloc_period_counts[('Right', 'pre_covid')],
    bloc_period_counts[('Right', 'post_covid')]
)



# %%
#Get the 20 most important words for each comparison

def top_words_df(logodds_df, label_A, label_B, comparison_name, top_n=20):
   
    
    # Top words distinctive for group A (highest z-scores)
    top_A = logodds_df.nlargest(top_n, 'z_score').copy()
    top_A['side'] = label_A

    # Top words distinctive for group B (lowest z-scores)
    top_B = logodds_df.nsmallest(top_n, 'z_score').copy()
    top_B['side'] = label_B

    # Add comparison name to both
    top_A['comparison'] = comparison_name
    top_B['comparison'] = comparison_name

    # Combine and return tidy dataframe
    return pd.concat([top_A, top_B], ignore_index=True)


# %%
#Create Dataframes for the results

df_lr_pre = top_words_df(
    res_left_right_pre,
    label_A='Left',
    label_B='Right',
    comparison_name='Left_vs_Right_pre'
)

df_lr_post = top_words_df(
    res_left_right_post,
    label_A='Left',
    label_B='Right',
    comparison_name='Left_vs_Right_post'
)

df_left_change = top_words_df(
    res_left_shift,
    label_A='Left_pre',
    label_B='Left_post',
    comparison_name='Left_pre_vs_post'
)

df_right_change = top_words_df(
    res_right_shift,
    label_A='Right_pre',
    label_B='Right_post',
    comparison_name='Right_pre_vs_post'
)


# %%
polarization_pre = np.mean(np.abs(res_left_right_pre['z_score']))
polarization_post = np.mean(np.abs(res_left_right_post['z_score']))




# %%
#We want a more refined view of polarization, so we look at quarterly changes

from collections import Counter

quarter_bloc_counts = {}

for (quarter, bloc), subset in df_health.groupby(['quarter', 'bloc_combined']):
    all_tokens = []
    for tokens in subset['text_token']:
        all_tokens.extend(tokens)
    quarter_bloc_counts[(quarter, bloc)] = Counter(all_tokens)



polarization_rows = []

quarters = sorted(df_health['quarter'].unique())

for q in quarters:
    key_left = (q, 'Left')
    key_right = (q, 'Right')
    
    # only compute if both have speeches in that quarter
    if key_left in quarter_bloc_counts and key_right in quarter_bloc_counts:
        
        res = log_odds_ratio(
            quarter_bloc_counts[key_left],
            quarter_bloc_counts[key_right]
        )
        
        # polarization = mean absolute z-score
        pol_score = res['z_score'].abs().mean()
        
        polarization_rows.append({
            'quarter': q,
            'polarization_score': pol_score
        })

# Final time-series dataframe
df_polarization = pd.DataFrame(polarization_rows)
df_polarization


# %%
#We want to see if the polarization is actually due to differences between lef and right or simply due to random variation (permutation test to get significance)

# Prepare structure: for each quarter, a list of (tokens, bloc_label)
quarter_docs = {}

for q, subset in df_health.groupby('quarter'):
    docs = [(tokens, bloc) for tokens, bloc in zip(subset['text_token'], subset['bloc_combined'])]
    quarter_docs[q] = docs

    
 


# %%
#Func for polarization per quarter
def polarization_for_quarter(counts_left, counts_right):
    res = log_odds_ratio(counts_left, counts_right)
    return res['z_score'].abs().mean()


# %%
#Func that randomly assigns bloc labels and computes polarization

import random

def permutation_polarization(quarter, n_iter=500):
   
    docs = quarter_docs[quarter]  # list of (tokens, bloc)
    tokens = [d[0] for d in docs] # list of token lists
    blocs = [d[1] for d in docs]  # list of bloc labels
    
    results = []
    
    for _ in range(n_iter):
        permuted_blocs = random.sample(blocs, len(blocs))
        
        left_tokens = []
        right_tokens = []
        
        for tok, b in zip(tokens, permuted_blocs):
            if b == 'Left':
                left_tokens.extend(tok)
            elif b == 'Right':
                right_tokens.extend(tok)
        
        # Only valid permutation if both sides appear
        if len(left_tokens) > 0 and len(right_tokens) > 0:
            c_left = Counter(left_tokens)
            c_right = Counter(right_tokens)
            pol = polarization_for_quarter(c_left, c_right)
            results.append(pol)
    
    return results


# %%
rows = []

for _, row in df_polarization.iterrows():
    q = row['quarter']
    obs = row['polarization_score']
    
    perm_scores = permutation_polarization(q, n_iter=500)
    perm_scores = np.array(perm_scores)
    
    # p-value: how many permutations produce >= observed polarization?
    pval = (perm_scores >= obs).mean()
    
    rows.append({
        'quarter': q,
        'observed_polarization': obs,
        'p_value': pval
    })

df_polarization_test = pd.DataFrame(rows)
df_polarization_test


# %%
#Last Check: Is Significance in q4 simply due to number of speeches? 

#Function to downsample speeches to equal n per bloc per quarter

def downsample_and_test_quarter(quarter_big, quarter_small, n_iter=500, random_state=42):
   
    # --- 1. Get speeches for the two quarters (Left/Right only) ---
    left_big = df_health[(df_health['quarter'] == quarter_big) & 
                         (df_health['bloc_combined'] == 'Left')]
    right_big = df_health[(df_health['quarter'] == quarter_big) & 
                          (df_health['bloc_combined'] == 'Right')]

    left_small = df_health[(df_health['quarter'] == quarter_small) & 
                           (df_health['bloc_combined'] == 'Left')]
    right_small = df_health[(df_health['quarter'] == quarter_small) & 
                            (df_health['bloc_combined'] == 'Right')]

    # If one bloc is missing in small quarter, we can't match sample sizes sensibly
    if len(left_small) == 0 or len(right_small) == 0:
        return {
            'quarter_big': quarter_big,
            'quarter_small': quarter_small,
            'error': 'One of the blocs (Left/Right) is missing in quarter_small.'
        }

    # --- 2. Determine how many speeches to sample (match small quarter) ---
    n_left = min(len(left_small), len(left_big))
    n_right = min(len(right_small), len(right_big))

    if n_left == 0 or n_right == 0:
        return {
            'quarter_big': quarter_big,
            'quarter_small': quarter_small,
            'error': 'Not enough speeches in quarter_big to downsample.'
        }

    # --- 3. Downsample big quarter to match counts ---
    rng = random.Random(random_state)
    left_big_down = left_big.sample(n=n_left, random_state=random_state)
    right_big_down = right_big.sample(n=n_right, random_state=random_state)

    # --- 4. Build counters for downsampled data ---
    left_big_tokens = [tok for tokens in left_big_down['text_token'] for tok in tokens]
    right_big_tokens = [tok for tokens in right_big_down['text_token'] for tok in tokens]

    c_left_down = Counter(left_big_tokens)
    c_right_down = Counter(right_big_tokens)

    # Downsampled polarization
    res_down = log_odds_ratio(c_left_down, c_right_down)
    pol_down = res_down['z_score'].abs().mean()

    # --- 5. Permutation test on DOWNsampled data ---
    # Build a local docs list for this quarter from the downsampled speeches only
    docs_down = []
    for tokens in left_big_down['text_token']:
        docs_down.append((tokens, 'Left'))
    for tokens in right_big_down['text_token']:
        docs_down.append((tokens, 'Right'))

    tokens_list = [d[0] for d in docs_down]
    blocs_list = [d[1] for d in docs_down]

    perm_pols = []
    for _ in range(n_iter):
        permuted_blocs = rng.sample(blocs_list, len(blocs_list))

        left_perm = []
        right_perm = []

        for tok_list, b in zip(tokens_list, permuted_blocs):
            if b == 'Left':
                left_perm.extend(tok_list)
            elif b == 'Right':
                right_perm.extend(tok_list)

        if len(left_perm) > 0 and len(right_perm) > 0:
            c_left_perm = Counter(left_perm)
            c_right_perm = Counter(right_perm)
            pol_perm = polarization_for_quarter(c_left_perm, c_right_perm)
            perm_pols.append(pol_perm)

    perm_pols = np.array(perm_pols)
    pval_down = (perm_pols >= pol_down).mean() if len(perm_pols) > 0 else None

    # --- 6. Original polarization and p-value from existing dataframes ---
    # Original polarization
    original_pol = None
    if 'df_polarization' in globals():
        match = df_polarization[df_polarization['quarter'] == quarter_big]
        if len(match) > 0:
            original_pol = float(match['polarization_score'].iloc[0])

    # Original p-value
    original_p = None
    if 'df_polarization_test' in globals():
        match_p = df_polarization_test[df_polarization_test['quarter'] == quarter_big]
        if len(match_p) > 0:
            original_p = float(match_p['p_value'].iloc[0])

    return {
        'quarter_big': quarter_big,
        'quarter_small': quarter_small,
        'original_polarization': original_pol,
        'original_p_value': original_p,
        'downsampled_polarization': pol_down,
        'downsampled_p_value': pval_down,
        'left_big_original_n': len(left_big),
        'right_big_original_n': len(right_big),
        'left_big_downsample_n': n_left,
        'right_big_downsample_n': n_right
    }


# %%
downsample_and_test_quarter("2020q2", "2019q3")


# %%
# Visualize 

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import save_plot




# Merge + significance flag
df_plot = df_polarization.merge(df_polarization_test, on="quarter")
df_plot["significant"] = df_plot["p_value"] < 0.05

quarters = df_plot["quarter"].tolist()
x = range(len(quarters))

# create figure and axes (instead of plt.figure)
fig, ax = plt.subplots(figsize=(12, 5))


# Make plot border thinner
for spine in ax.spines.values():
    spine.set_linewidth(0.6)


# ---- COLOR ----
blue = "#31688e"      # viridis blue
white = "#ffffff"     # pure white for fill

# ---- LINE (behind everything) ----
ax.plot(
    x,
    df_plot["observed_polarization"],
    color=blue,
    linewidth=2.0,
    alpha=0.9,
    zorder=1
)

# ---- POINTS ----
for i, row in df_plot.iterrows():
    if row["significant"]:
        # Significant = filled blue
        ax.scatter(
            i,
            row["observed_polarization"],
            s=90,
            color=blue,
            edgecolor="none",
            zorder=3
        )
    else:
        # Non-significant = white fill + blue edge
        ax.scatter(
            i,
            row["observed_polarization"],
            s=90,
            facecolor=white,
            edgecolor=blue,
            linewidth=1.5,
            zorder=2
        )

# ---- COVID marker ----
covid_q = "2020q1"
if covid_q in quarters:
    covid_idx = quarters.index(covid_q)
    ax.axvline(covid_idx, color="#1a1a1a", linestyle="--", alpha=0.6)

# ---- Y-range ----
ax.set_ylim(0.5, 0.65)

# ---- Period labels ----
label_y = 0.63
style = dict(fontsize=12, fontweight="bold")

if "2019q1" in quarters and "2019q2" in quarters:
    ax.text(
        (quarters.index("2019q1") + quarters.index("2019q2")) / 2,
        label_y,
        "Pre COVID-19",
        ha="center",
        **style
    )

if "2021q2" in quarters and "2021q3" in quarters:
    ax.text(
        (quarters.index("2021q2") + quarters.index("2021q3")) / 2,
        label_y,
        "COVID-19",
        ha="center",
        **style
    )

# ---- Axes ----
ax.set_xticks(list(x))
ax.set_xticklabels(quarters, rotation=90, fontsize=12)
ax.set_xlabel("Year and Quarter", fontsize=14, fontweight="bold")
ax.set_ylabel("Mean |z-scored log-odds ratio|", fontsize=14, fontweight="bold")

# ---- Legend ----
legend_elements = [
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor=blue,
        markeredgecolor=blue,
        markersize=8,
        linestyle="",
        label="Significant*"
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor=white,
        markeredgecolor=blue,
        markersize=8,
        linestyle="",
        label="Non-Significant*"
    )
]

ax.legend(handles=legend_elements, fontsize=10, title="Significance status")

# ---- Footnote ----
ax.text(
    0.99,
    -0.24,
    "* Based on p < 0.05 (95% confidence level)",
    ha="right",
    fontsize=9,
    fontweight="bold",
    transform=ax.transAxes   # was plt.gca().transAxes
)

ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()



#Export the plot

save_plot(fig, f"Log Odd Z-Score {DATASET}")




# %%
#Maybe the significance and polarization during covid is simply due to higher number of speeches per bloc? Let's downsample and see

# Equalize *total* speeches per quarter (keep Left/Right proportions).
# Uses target_n = min total speeches among quarters that contain BOTH blocs.

import numpy as np
import pandas as pd

def downsample_equal_quarter_size(df, random_state=42, target_n=None):
    # Keep only Left/Right
    df_lr = df[df['bloc_combined'].isin(['Left', 'Right'])].copy()

    # Count speeches per quarter and ensure both blocs are present
    quarter_counts = (
        df_lr.groupby(['quarter', 'bloc_combined'])
             .size()
             .rename('n')
             .reset_index()
    )
    both_present_quarters = (
        quarter_counts.groupby('quarter')['bloc_combined']
        .nunique()
        .reset_index(name='n_blocs')
    )
    both_present_quarters = both_present_quarters[both_present_quarters['n_blocs'] == 2]['quarter'].tolist()

    # Total speeches per eligible quarter
    totals = (
        df_lr[df_lr['quarter'].isin(both_present_quarters)]
        .groupby('quarter')
        .size()
        .rename('total_n')
    )

    # Choose target_n if not provided
    if target_n is None:
        target_n = int(totals.min())  # the smallest total among eligible quarters

    rng = np.random.RandomState(random_state)
    sampled_frames = []

    for q in sorted(both_present_quarters):
        qdf = df_lr[df_lr['quarter'] == q]

        # Original Left/Right counts for proportional allocation
        n_left = (qdf['bloc_combined'] == 'Left').sum()
        n_right = (qdf['bloc_combined'] == 'Right').sum()
        total_q = n_left + n_right
        if total_q < target_n:
            # Skip if quarter can't meet the target (shouldn't happen if we used global min)
            continue

        # Proportional allocation to keep bloc mix (round and adjust to sum target_n)
        target_left = int(round(target_n * (n_left / total_q)))
        target_right = target_n - target_left  # ensures sum == target_n

        left_s = qdf[qdf['bloc_combined'] == 'Left'].sample(
            n=target_left, random_state=random_state, replace=False
        )
        right_s = qdf[qdf['bloc_combined'] == 'Right'].sample(
            n=target_right, random_state=random_state, replace=False
        )

        sampled_frames.append(left_s)
        sampled_frames.append(right_s)

    if not sampled_frames:
        raise ValueError("Downsampling produced an empty dataset. Check data availability.")

    df_equalQ = pd.concat(sampled_frames, ignore_index=True)

    print(f"Target per-quarter N = {target_n}")
    print(
        df_equalQ.groupby(['quarter', 'bloc_combined'])
                 .size()
                 .unstack(fill_value=0)
                 .head(10)
    )
    return df_equalQ

# Build the equal-quarter dataset (uses global min by default)
df_health_equalQ = downsample_equal_quarter_size(df_health, random_state=42)


# %%
from collections import Counter

# Rebuild counters from the downsampled dataset
quarter_bloc_counts_equalQ = {}
for (quarter, bloc), subset in df_health_equalQ.groupby(['quarter', 'bloc_combined']):
    toks = []
    for tl in subset['text_token']:
        toks.extend(tl)
    quarter_bloc_counts_equalQ[(quarter, bloc)] = Counter(toks)

# Compute polarization time series (mean |z|) on equal-quarter data
rows_equalQ = []
quarters_equalQ = sorted(df_health_equalQ['quarter'].unique())

for q in quarters_equalQ:
    keyL = (q, 'Left')
    keyR = (q, 'Right')
    if keyL in quarter_bloc_counts_equalQ and keyR in quarter_bloc_counts_equalQ:
        res = log_odds_ratio(quarter_bloc_counts_equalQ[keyL], quarter_bloc_counts_equalQ[keyR])
        pol = res['z_score'].abs().mean()
        rows_equalQ.append({'quarter': q, 'polarization_score': pol})

df_polarization_equalQ = pd.DataFrame(rows_equalQ).sort_values('quarter')
df_polarization_equalQ.head()


# %%
import random
import numpy as np

# Build quarter_docs for equal-quarter data
quarter_docs_equalQ = {}
for q, subset in df_health_equalQ.groupby('quarter'):
    docs = [(tokens, bloc) for tokens, bloc in zip(subset['text_token'], subset['bloc_combined'])]
    quarter_docs_equalQ[q] = docs

def permutation_polarization_equalQ(quarter, n_iter=500, random_state=42):
    rng = random.Random(random_state)
    docs = quarter_docs_equalQ[quarter]
    tokens = [d[0] for d in docs]
    blocs = [d[1] for d in docs]
    out = []

    for _ in range(n_iter):
        permuted = rng.sample(blocs, len(blocs))
        left_tokens, right_tokens = [], []
        for tok_list, b in zip(tokens, permuted):
            if b == 'Left':
                left_tokens.extend(tok_list)
            elif b == 'Right':
                right_tokens.extend(tok_list)
        if left_tokens and right_tokens:
            cL = Counter(left_tokens)
            cR = Counter(right_tokens)
            out.append(polarization_for_quarter(cL, cR))
    return np.array(out)

# Compute p-values
rows_p_equalQ = []
for _, r in df_polarization_equalQ.iterrows():
    q = r['quarter']
    obs = r['polarization_score']
    perm = permutation_polarization_equalQ(q, n_iter=500, random_state=42)
    pval = (perm >= obs).mean() if len(perm) else np.nan
    rows_p_equalQ.append({'quarter': q, 'polarization_score': obs, 'p_value': pval})

df_polarization_test_equalQ = pd.DataFrame(rows_p_equalQ).sort_values('quarter')
df_polarization_test_equalQ.head()


# %%
# ==== Downsampled (Equal-Quarter) Visualization ====

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Work directly with df_polarization_test_equalQ
df_plot = df_polarization_test_equalQ.sort_values("quarter").copy()
df_plot["significant"] = df_plot["p_value"] < 0.05

quarters = df_plot["quarter"].tolist()
x = range(len(quarters))

plt.figure(figsize=(12, 5))

# ---- COLOR ----
blue = "#31688e"      # viridis blue
white = "#ffffff"     # pure white for fill

# ---- LINE ----
plt.plot(
    x,
    df_plot["polarization_score"],
    color=blue,
    linewidth=2.0,
    alpha=0.9,
    zorder=1
)

# ---- POINTS ----
for i, row in df_plot.iterrows():
    if row["significant"]:
        plt.scatter(
            i,
            row["polarization_score"],
            s=90,
            color=blue,
            edgecolor="none",
            zorder=3
        )
    else:
        plt.scatter(
            i,
            row["polarization_score"],
            s=90,
            facecolor=white,
            edgecolor=blue,
            linewidth=1.5,
            zorder=2
        )

# ---- COVID marker ----
covid_q = "2020q1"
if covid_q in quarters:
    covid_idx = quarters.index(covid_q)
    plt.axvline(covid_idx, color="#1a1a1a", linestyle="--", alpha=0.6)

# ---- Y-range ----
plt.ylim(0.35, 0.65)  # adjust if needed

# ---- Period labels ----
label_y = 0.63
style = dict(fontsize=12, fontweight="bold")

if "2019q1" in quarters and "2019q2" in quarters:
    plt.text(
        (quarters.index("2019q1") + quarters.index("2019q2")) / 2,
        label_y,
        "Pre COVID-19",
        ha="center",
        **style
    )

if "2021q2" in quarters and "2021q3" in quarters:
    plt.text(
        (quarters.index("2021q2") + quarters.index("2021q3")) / 2,
        label_y,
        "COVID-19",
        ha="center",
        **style
    )

# ---- Axes ----
plt.xticks(x, quarters, rotation=90, fontsize=11)
plt.xlabel("Year and Quarter", fontsize=12, fontweight="bold")
plt.ylabel("Mean |z-scored log-odds ratio|", fontsize=12, fontweight="bold")

# ---- Legend ----
legend_elements = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=blue, markeredgecolor=blue,
           markersize=8, linestyle="", label="Significant*"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=white, markeredgecolor=blue,
           markersize=8, linestyle="", label="Non-Significant*")
]
plt.legend(handles=legend_elements, fontsize=10, title="Significance status")

# ---- Footnote ----
plt.text(
    0.99, -0.24,
    "* Based on p < 0.05 (95% confidence level)",
    ha="right",
    fontsize=9,
    fontweight="bold",
    transform=plt.gca().transAxes
)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------
# Prepare data
# --------------------------
orig = df_polarization.merge(df_polarization_test, on="quarter").copy()
orig = orig.sort_values("quarter").rename(columns={"observed_polarization": "value"})
orig["significant"] = orig["p_value"] < 0.05

eq = df_polarization_test_equalQ.sort_values("quarter").copy()
eq = eq.rename(columns={"polarization_score": "value"})
eq["significant"] = eq["p_value"] < 0.05

quarters = sorted(set(orig["quarter"]).intersection(set(eq["quarter"])))
orig = orig[orig["quarter"].isin(quarters)].reset_index(drop=True)
eq = eq[eq["quarter"].isin(quarters)].reset_index(drop=True)
x = list(range(len(quarters)))

# --------------------------
# Style setup
# --------------------------
blue  = "#31688e"  # original
green = "#35b779"  # equalized
white = "#ffffff"

covid_q = "2020q1"
label_y = 0.63
style = dict(fontsize=12, fontweight="bold")

# figure + axes
fig, ax = plt.subplots(figsize=(12, 5))

# --------------------------
# Lines
# --------------------------
ax.plot(x, orig["value"], color=blue, linewidth=2.0, alpha=0.9, zorder=1)
ax.plot(x, eq["value"],   color=green, linewidth=2.0, alpha=0.9, zorder=1)

# --------------------------
# Points (filled = significant, hollow = non-significant)
# --------------------------
for i, q in enumerate(quarters):
    row_o = orig.iloc[i]
    ax.scatter(
        i, row_o["value"], s=90,
        color=blue if row_o["significant"] else white,
        edgecolor=blue, linewidth=1.5, zorder=3
    )
    row_e = eq.iloc[i]
    ax.scatter(
        i, row_e["value"], s=90,
        color=green if row_e["significant"] else white,
        edgecolor=green, linewidth=1.5, zorder=3
    )

# --------------------------
# COVID marker + labels
# --------------------------
if covid_q in quarters:
    covid_idx = quarters.index(covid_q)
    ax.axvline(covid_idx, color="#1a1a1a", linestyle="--", alpha=0.6)

if "2019q1" in quarters and "2019q2" in quarters:
    ax.text(
        (quarters.index("2019q1") + quarters.index("2019q2")) / 2,
        label_y, "Pre COVID-19", ha="center", **style
    )

if "2021q2" in quarters and "2021q3" in quarters:
    ax.text(
        (quarters.index("2021q2") + quarters.index("2021q3")) / 2,
        label_y, "COVID-19", ha="center", **style
    )

# --------------------------
# Axes
# --------------------------
ax.set_xticks(x)
ax.set_xticklabels(quarters, rotation=90, fontsize=11)
ax.set_xlabel("Year and Quarter", fontsize=12, fontweight="bold")
ax.set_ylabel("Mean |z-scored log-odds ratio|", fontsize=12, fontweight="bold")
ax.set_ylim(0.35, 0.65)
ax.grid(alpha=0.3)

# --------------------------
# Inline labels inside plot
# --------------------------
# Label above the blue line (Original)
ax.text(
    len(quarters) - 1 - 0.5,
    orig["value"].iloc[-1] + 0.03,   # slight upward offset
    "Original",
    color=blue,
    fontsize=12,
    fontweight="bold",
    va="bottom"
)

# Label below the green line (Equalized)
ax.text(
    len(quarters) - 1 - 0.7,
    eq["value"].iloc[-1] - 0.025,   # slight downward offset
    "Equalized",
    color=green,
    fontsize=12,
    fontweight="bold",
    va="top"
)

# --------------------------
# Simplified legend: only significance markers, bottom left
# --------------------------
legend_elements = [
    Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor="black", markeredgecolor="black",
        markersize=8, linestyle="", label="Significant*"
    ),
    Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor=white, markeredgecolor="black",
        markersize=8, linestyle="", label="Non-Significant*"
    )
]

ax.legend(
    handles=legend_elements,
    fontsize=10,
    title="Significance",
    loc="lower left",
    frameon=True
)

# --------------------------
# Footnote
# --------------------------
ax.text(
    0.99, -0.24,
    "* Based on p < 0.05 (95% confidence level)",
    ha="right",
    fontsize=9,
    fontweight="bold",
    transform=ax.transAxes
)

fig.tight_layout()
plt.show()


#Export the comparison plot

save_plot(fig, f"Log Odd Z-Score Ori vs EQ {DATASET}")



