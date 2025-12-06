"""Quick, robust alternatives to embedding/topic models for fast results.

This script computes:
- Top frequent unigrams (by raw counts)
- Top TF-IDF unigrams
- Distinctive terms between pre-COVID and post-COVID using log-odds

Usage:
  - Edit `INPUT_CSV` at the top or pass the path by setting the variable
  - Run: `python simple_text_analysis_last_resort.py`

Outputs printed to console and saved as CSVs next to the input file.
"""
from pathlib import Path
import math
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None
# optional spaCy for lemmatization / POS filtering
# The spaCy import and automatic lemmatization are commented out so
# you can re-enable them later by uncommenting the block below.
# try:
#     import spacy
#     try:
#         # load small Italian model if available; may raise if not installed
#         nlp_it = spacy.load('it_core_news_sm')
#     except Exception:
#         nlp_it = None
# except Exception:
#     spacy = None
#     nlp_it = None
spacy = None
nlp_it = None


# --- CONFIG: point this at your CSV (or change at runtime in the file) ---
INPUT_CSV = Path(r"D:\OneDrive - Università Commerciale Luigi Bocconi\Desktop\Università\Foundations of Social Sciences\Group Project\df_health_lda_final.csv")
# quarter cutoff for pre/post: inclusive post from 2020 Q2 -> '2020q2'
CUTOFF_QUARTER = (2020, 2)  # year, quarter (Q2 2020 is post)subset_health_topics
# number of top terms to report
TOP_N = 30

# small conservative list of Italian stopwords (not exhaustive but helpful)
# scikit-learn CountVectorizer expects a list (or None), not a set
ITALIAN_STOPWORDS = [
    'e','di','la','il','che','da','in','per','con','su','non','una','si','al','le','i',
    'del','della','dei','delle','al','ai','anche','ma','come','tra','fra','sia',
    'sono','stato','stata','ha','hanno','ci','quella','questo','questa','queste','questi',
    'essere','stati', "perchã", "quindi", "giã", "molto", "governo", "piÃ¹", "Ã", "presidente", "decreto", "legge", "solo", 
    "sanitÃ", "Ã¨", "pease", "fare", "essere", "avere", "dire", "dare", "collega", "tema", "Aula", "articolo",
    "emendamento", "chiedere", "credere", "oggi", "regione", "italia" 
]

# Try to augment stopwords with NLTK Italian stopwords if available
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    try:
        nltk_stop = set(nltk_stopwords.words('italian'))
    except Exception:
        # sometimes the stopwords corpus isn't downloaded
        try:
            nltk.download('stopwords')
            nltk_stop = set(nltk_stopwords.words('italian'))
        except Exception:
            nltk_stop = set()
    if nltk_stop:
        # extend our list while preserving order (put nltk words after ours)
        for w in sorted(nltk_stop):
            if w not in ITALIAN_STOPWORDS:
                ITALIAN_STOPWORDS.append(w)
except Exception:
    pass


def get_text_column(df: pd.DataFrame):
    for col in ('text_clean', 'text', 'speech'):
        if col in df.columns:
            return col
    raise SystemExit("No text column found. Expected one of: text_clean, text, speech")


def top_frequencies(series, n=TOP_N, ngram_range=(1,1)):
    # token_pattern matches words of 3+ letters (basic way to filter short articles/connectives)
    token_pattern = r"(?u)\b[A-Za-zÀ-ÿ]{3,}\b"
    vec = CountVectorizer(lowercase=True, stop_words=ITALIAN_STOPWORDS, ngram_range=ngram_range,
                          token_pattern=token_pattern, min_df=5, max_df=0.95)
    X = vec.fit_transform(series.fillna(''))
    sums = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    idx = np.argsort(-sums)[:n]
    return pd.DataFrame({'term': vocab[idx], 'count': sums[idx]})


def generate_wordcloud(series, out_path: Path, max_words=200):
    if WordCloud is None:
        print("wordcloud package not available; skipping wordcloud generation.")
        return
    text = ' '.join(series.fillna('').astype(str).values)
    if not text.strip():
        print("No text for wordcloud; skipping.")
        return
    wc = WordCloud(width=1200, height=600, background_color='white', stopwords=ITALIAN_STOPWORDS, max_words=max_words)
    wc.generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved wordcloud: {out_path}")


# Lemmatization code commented out so it can be re-enabled later.
# def lemmatize_series(series: pd.Series, nlp, batch_size: int = 200, keep_pos=None):
#     """Lemmatize and POS-filter a pandas Series using a spaCy pipeline.
#
#     Returns a pandas Series of lemmatized strings (tokens joined by space).
#     If `nlp` is None, returns an empty series of the same index.
#     """
#     if nlp is None:
#         return pd.Series([''] * len(series), index=series.index)
#     if keep_pos is None:
#         keep_pos = {'NOUN', 'PROPN', 'ADJ', 'VERB'}
#
#     texts = series.fillna('').astype(str)
#     out = []
#     for doc in nlp.pipe(texts, batch_size=batch_size):
#         toks = []
#         for tok in doc:
#             if not tok.is_alpha:
#                 continue
#             if tok.is_stop:
#                 continue
#             if tok.pos_ not in keep_pos:
#                 continue
#             lemma = tok.lemma_.strip()
#             if len(lemma) < 3:
#                 continue
#             toks.append(lemma)
#         out.append(' '.join(toks))
#     return pd.Series(out, index=series.index)


def lemmatize_series(series: pd.Series, nlp, batch_size: int = 200, keep_pos=None):
    """Lemmatize and POS-filter a pandas Series using a spaCy pipeline.

    Defaults to keep only NOUN/PROPN/ADJ to reduce noisy lemmas (VERB excluded).
    If `nlp` is None, returns the raw text series (graceful fallback).
    """
    if nlp is None:
        return series.fillna('').astype(str)

    if keep_pos is None:
        keep_pos = {'NOUN', 'PROPN', 'ADJ'}

    texts = series.fillna('').astype(str)
    out = []
    # use nlp.pipe for speed; disable parser/ner to make it faster if available
    disable = []
    try:
        # spaCy v3 expects component names to disable; guard in case attribute missing
        disable = ['parser', 'ner']
    except Exception:
        disable = []

    for doc in nlp.pipe(texts, batch_size=batch_size, disable=disable):
        toks = []
        for tok in doc:
            if not tok.is_alpha:
                continue
            if tok.is_stop:
                continue
            if tok.pos_ not in keep_pos:
                continue
            lemma = tok.lemma_.strip()
            if len(lemma) < 3:
                continue
            toks.append(lemma.lower())
        out.append(' '.join(toks))
    return pd.Series(out, index=series.index)


def plot_top_terms(df_terms: pd.DataFrame, term_col: str, value_col: str, out_path: Path, title: str, top_n=15):
    df = df_terms.head(top_n).copy()
    plt.figure(figsize=(8, max(4, top_n * 0.25)))
    sns.barplot(x=value_col, y=term_col, data=df, palette='viridis')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot: {out_path}")


def compute_pairwise(name_a: str, series_a, name_b: str, series_b, out_dir: Path, ngram_range=(1,1)):
    """Compute freq/tfidf/log-odds and save CSVs + plots for a pair of series.

    name_a/name_b are short labels used in filenames.
    """
    safe_a = name_a.replace(' ', '_')
    safe_b = name_b.replace(' ', '_')
    prefix = f"{safe_a}_vs_{safe_b}"

    # guard empty
    if len(series_a) == 0 or len(series_b) == 0:
        print(f"Skipping pair {name_a} vs {name_b}: one side has no documents ({len(series_a)} vs {len(series_b)})")
        return

    freq_a = top_frequencies(series_a, n=TOP_N, ngram_range=ngram_range)
    freq_b = top_frequencies(series_b, n=TOP_N, ngram_range=ngram_range)
    tf_a = top_tfidf(series_a, n=TOP_N, ngram_range=ngram_range)
    tf_b = top_tfidf(series_b, n=TOP_N, ngram_range=ngram_range)
    top_a, top_b = distinctive_terms_by_logodds(series_a, series_b, n=TOP_N, ngram_range=ngram_range)

    # save csvs
    freq_a.to_csv(out_dir / f"{prefix}_top_freq_{safe_a}.csv", index=False)
    freq_b.to_csv(out_dir / f"{prefix}_top_freq_{safe_b}.csv", index=False)
    tf_a.to_csv(out_dir / f"{prefix}_top_tfidf_{safe_a}.csv", index=False)
    tf_b.to_csv(out_dir / f"{prefix}_top_tfidf_{safe_b}.csv", index=False)
    top_a.to_csv(out_dir / f"{prefix}_distinctive_{safe_a}_vs_{safe_b}.csv", index=False)
    top_b.to_csv(out_dir / f"{prefix}_distinctive_{safe_b}_vs_{safe_a}.csv", index=False)

    # plots
    try:
        plot_top_terms(freq_a, 'term', 'count', out_dir / f"{prefix}_freq_{safe_a}.png", f"Top freq {name_a}")
        plot_top_terms(freq_b, 'term', 'count', out_dir / f"{prefix}_freq_{safe_b}.png", f"Top freq {name_b}")
        plot_top_terms(top_a.rename(columns={'log_odds': 'log_odds'}), 'term', 'log_odds', out_dir / f"{prefix}_distinctive_{safe_a}.png", f"Distinctive {name_a} vs {name_b}")
        plot_top_terms(top_b.rename(columns={'log_odds': 'log_odds'}), 'term', 'log_odds', out_dir / f"{prefix}_distinctive_{safe_b}.png", f"Distinctive {name_b} vs {name_a}")
    except Exception as e:
        print(f"Plot generation for pair {name_a} vs {name_b} failed:", e)

    # wordclouds
    # Note: wordcloud generation for individual pairs was removed to avoid
    # creating many per-pair images. Wordclouds are generated centrally
    # in `run_all` to produce the eight requested summary visuals.

def fix_specific_mojibake(s: str) -> str:
    if not s:
        return s
    return (s.replace("Ã¨", "è")  # e-grave
             .replace("Ã ", "à")  # a-grave
             .replace("Ã¹", "ù")
             .replace("Ã²", "ò")
             .replace("Ã¬", "ì")
             .replace("Ã©", "é"))


def top_tfidf(series, n=TOP_N, ngram_range=(1,1)):
    token_pattern = r"(?u)\b[A-Za-zÀ-ÿ]{3,}\b"
    vec = TfidfVectorizer(lowercase=True, stop_words=ITALIAN_STOPWORDS, ngram_range=ngram_range,
                          token_pattern=token_pattern, min_df=5, max_df=0.95)
    X = vec.fit_transform(series.fillna(''))
    # mean tfidf across documents as a simple score
    means = np.asarray(X.mean(axis=0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    idx = np.argsort(-means)[:n]
    return pd.DataFrame({'term': vocab[idx], 'tfidf_mean': means[idx]})


def log_odds_ratio(df_a_counts, df_b_counts, prior=0.01):
    """Compute simple log-odds ratio with add-prior smoothing.

    df_a_counts, df_b_counts: pandas Series indexed by term with integer counts.
    Returns DataFrame with term, log_odds.
    """
    # build union vocab
    all_terms = sorted(set(df_a_counts.index).union(df_b_counts.index))
    a = np.array([df_a_counts.get(t, 0) for t in all_terms], dtype=float)
    b = np.array([df_b_counts.get(t, 0) for t in all_terms], dtype=float)
    # apply small prior to avoid zeros
    a += prior
    b += prior
    A = a.sum()
    B = b.sum()
    # probabilities
    pa = a / A
    pb = b / B
    # avoid division by zero
    with np.errstate(divide='ignore'):
        log_odds = np.log(pa / (1 - pa)) - np.log(pb / (1 - pb))
    return pd.DataFrame({'term': all_terms, 'log_odds': log_odds}).sort_values('log_odds', ascending=False)


def distinctive_terms_by_logodds(series_a, series_b, n=TOP_N, ngram_range=(1,1)):
    token_pattern = r"(?u)\b[A-Za-zÀ-ÿ]{3,}\b"
    vec = CountVectorizer(lowercase=True, stop_words=ITALIAN_STOPWORDS, ngram_range=ngram_range,
                          token_pattern=token_pattern, min_df=5, max_df=0.95)
    # fit on combined corpus so vocabulary is shared
    combined = pd.concat([series_a.fillna(''), series_b.fillna('')])
    X_comb = vec.fit_transform(combined)
    # now transform the splits
    n_a = len(series_a)
    X_a = X_comb[:n_a]
    X_b = X_comb[n_a:]
    sums_a = np.asarray(X_a.sum(axis=0)).ravel()
    sums_b = np.asarray(X_b.sum(axis=0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    s_a = pd.Series(sums_a, index=vocab)
    s_b = pd.Series(sums_b, index=vocab)
    df = log_odds_ratio(s_a, s_b, prior=1.0)
    # Invert sign so that positive log_odds indicate terms more characteristic
    # of series_b (second argument), and negative values indicate terms
    # more characteristic of series_a (first argument). This makes
    # pre (series_a) terms negative and post (series_b) terms positive
    # when calling distinctive_terms_by_logodds(pre, post).
    df['log_odds'] = -df['log_odds']

    # Terms characteristic of A (series_a) are those with negative log_odds;
    # sort ascending so the most negative (most A-characteristic) come first.
    top_a = df[df['log_odds'] < 0].sort_values('log_odds', ascending=True).head(n)

    # Terms characteristic of B (series_b) are positive log_odds; sort
    # descending so the most B-characteristic terms come first.
    top_b = df[df['log_odds'] > 0].sort_values('log_odds', ascending=False).head(n)

    return top_a, top_b


def run_all(input_csv: Path):
    print("Loading:", input_csv)
    df = pd.read_csv(input_csv, low_memory=False)
    text_col = get_text_column(df)
    print(f"Using text column: {text_col}")

    # Apply mojibake fixes to the raw text column so downstream analyses use cleaned text
    try:
        df[text_col] = df[text_col].astype(str).apply(fix_specific_mojibake)
    except Exception:
        # if something goes wrong, continue without stopping — the fixer is best-effort
        print(f"Warning: failed to apply mojibake fixer to column {text_col}")

    # prefer quarter column if available; otherwise fall back to year or date
    if 'quarter' in df.columns:
        # parse quarters like '2018q2' (case-insensitive)
        def parse_quarter(q):
            try:
                q = str(q).lower().strip()
                if 'q' in q:
                    y, qn = q.split('q')
                    return int(y), int(qn)
            except Exception:
                return None
            return None

        parsed = df['quarter'].apply(parse_quarter)
        # create mask for post (>= cutoff quarter)
        def is_post(parsed_q):
            if not parsed_q:
                return False
            y, qn = parsed_q
            cy, cq = CUTOFF_QUARTER
            return (y > cy) or (y == cy and qn >= cq)

        mask_post = parsed.apply(is_post)
        # Try to load spaCy Italian model at runtime; if available, lemmatize.
        nlp_local = None
        try:
            import spacy as _spacy
            try:
                nlp_local = _spacy.load('it_core_news_sm')
            except Exception:
                nlp_local = None
        except Exception:
            nlp_local = None

        if nlp_local is not None:
            print('spaCy Italian model found: creating lemmatized text (this may take a moment)...')
            df['text_lemm'] = lemmatize_series(df[text_col], nlp_local)
            # show a small sample to let the user inspect original vs lemmatized text
            try:
                sample_n = min(5, len(df))
                for i, row in df[[text_col, 'text_lemm']].sample(sample_n, random_state=1).iterrows():
                    print('\nORIG:', str(row[text_col])[:200])
                    print('LEMM:', str(row['text_lemm'])[:200])
                    print('---')
            except Exception:
                pass
            analysis_col = 'text_lemm'
            # Force unigrams only (user requested no bigrams)
            ngram_range_for_analysis = (1,1)
        else:
            analysis_col = text_col
            ngram_range_for_analysis = (1,1)
            print("spaCy Italian model not available; running analysis on raw text. To enable lemmatization: pip install spacy && python -m spacy download it_core_news_sm")

        pre = df.loc[~mask_post, analysis_col]
        post = df.loc[mask_post, analysis_col]
        print(f"Using 'quarter' column to split: Documents pre={len(pre)}, post={len(post)} (analysis column: {analysis_col})")
    else:
        # fallback: use year or date
        if 'year' not in df.columns:
            if 'date' in df.columns:
                df['year'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce').dt.year
            else:
                raise SystemExit("No 'quarter', 'year' or 'date' column found to split pre/post. Add a 'quarter' column or a 'date' column.")
        df = df.dropna(subset=[text_col, 'year'])
        df['year'] = df['year'].astype(int)
        # Try to load spaCy Italian model at runtime; if available, lemmatize.
        nlp_local = None
        try:
            import spacy as _spacy
            try:
                nlp_local = _spacy.load('it_core_news_sm')
            except Exception:
                nlp_local = None
        except Exception:
            nlp_local = None

        if nlp_local is not None:
            print('spaCy Italian model found: creating lemmatized text (this may take a moment)...')
            df['text_lemm'] = lemmatize_series(df[text_col], nlp_local)
            # show a small sample to let the user inspect original vs lemmatized text
            try:
                sample_n = min(5, len(df))
                for i, row in df[[text_col, 'text_lemm']].sample(sample_n, random_state=1).iterrows():
                    print('\nORIG:', str(row[text_col])[:200])
                    print('LEMM:', str(row['text_lemm'])[:200])
                    print('---')
            except Exception:
                pass
            analysis_col = 'text_lemm'
            # Force unigrams only (user requested no bigrams)
            ngram_range_for_analysis = (1,1)
        else:
            analysis_col = text_col
            ngram_range_for_analysis = (1,1)
            print("spaCy Italian model not available; running analysis on raw text. To enable lemmatization: pip install spacy && python -m spacy download it_core_news_sm")
        pre = df.loc[df['year'] < CUTOFF_QUARTER[0], analysis_col]
        post = df.loc[df['year'] >= CUTOFF_QUARTER[0], analysis_col]
        print(f"Using 'year' column fallback: Documents pre={len(pre)}, post={len(post)} (analysis column: {analysis_col})")

    # Top frequency terms (unigrams)
    freq_pre = top_frequencies(pre, n=TOP_N, ngram_range=ngram_range_for_analysis)
    freq_post = top_frequencies(post, n=TOP_N, ngram_range=ngram_range_for_analysis)
    print('\nTop frequency (pre):')
    print(freq_pre.head(20).to_string(index=False))
    print('\nTop frequency (post):')
    print(freq_post.head(20).to_string(index=False))

    # generate wordclouds (if wordcloud installed)
    out_dir = input_csv.parent
    try:
        generate_wordcloud(pre, out_dir / 'wordcloud_pre.png')
        generate_wordcloud(post, out_dir / 'wordcloud_post.png')
    except Exception as e:
        print('Wordcloud generation failed:', e)

    # Top TF-IDF
    tf_pre = top_tfidf(pre, n=TOP_N, ngram_range=ngram_range_for_analysis)
    tf_post = top_tfidf(post, n=TOP_N, ngram_range=ngram_range_for_analysis)
    print('\nTop TF-IDF (pre):')
    print(tf_pre.head(20).to_string(index=False))
    print('\nTop TF-IDF (post):')
    print(tf_post.head(20).to_string(index=False))

    # Distinctive terms via log-odds
    top_pre, top_post = distinctive_terms_by_logodds(pre, post, n=TOP_N, ngram_range=ngram_range_for_analysis)
    print('\nMost distinctive terms (pre vs post) — characteristic of PRE:')
    print(top_pre.head(20).to_string(index=False))
    print('\nMost distinctive terms (post vs pre) — characteristic of POST:')
    print(top_post.head(20).to_string(index=False))

    # Save CSVs
    out_dir = input_csv.parent
    freq_pre.to_csv(out_dir / 'top_freq_pre.csv', index=False)
    freq_post.to_csv(out_dir / 'top_freq_post.csv', index=False)
    tf_pre.to_csv(out_dir / 'top_tfidf_pre.csv', index=False)
    tf_post.to_csv(out_dir / 'top_tfidf_post.csv', index=False)
    top_pre.to_csv(out_dir / 'distinctive_pre_vs_post.csv', index=False)
    top_post.to_csv(out_dir / 'distinctive_post_vs_pre.csv', index=False)
    
    # === Additional comparisons: left vs right and pre/post within each side ===
    # only run if `side` column exists
    if 'side3' in df.columns:
        # normalize side entries
        side_norm = df['side3'].fillna('').astype(str).str.strip().str.lower()
        is_left = side_norm == 'left'
        is_right = side_norm == 'right'

        df_left = df.loc[is_left]
        df_right = df.loc[is_right]
        if len(df_left) == 0 or len(df_right) == 0:
            print('No left/right groups found with expected labels; skipping side comparisons.')
        else:
            # build post mask for the full df (ensure defined)
            if 'quarter' in df.columns:
                mask_post_full = mask_post
            else:
                mask_post_full = df['year'] >= CUTOFF_QUARTER[0]

            # pre/post for left and right
            pre_left = df_left.loc[~mask_post_full.loc[df_left.index], analysis_col]
            post_left = df_left.loc[mask_post_full.loc[df_left.index], analysis_col]
            pre_right = df_right.loc[~mask_post_full.loc[df_right.index], analysis_col]
            post_right = df_right.loc[mask_post_full.loc[df_right.index], analysis_col]

            # Generate the 8 requested wordclouds (one file each):
            # - left_pre, right_pre, left_post, right_post,
            # - general_pre, general_post, general_left, general_right
            try:
                # left / right pre/post
                generate_wordcloud(pre_left, out_dir / 'wordcloud_left_pre.png')
                generate_wordcloud(pre_right, out_dir / 'wordcloud_right_pre.png')
                generate_wordcloud(post_left, out_dir / 'wordcloud_left_post.png')
                generate_wordcloud(post_right, out_dir / 'wordcloud_right_post.png')

                # general pre/post (already defined as `pre` and `post` earlier)
                generate_wordcloud(pre, out_dir / 'wordcloud_pre.png')
                generate_wordcloud(post, out_dir / 'wordcloud_post.png')

                # general left / right (all docs for each side)
                generate_wordcloud(df_left[analysis_col], out_dir / 'wordcloud_left.png')
                generate_wordcloud(df_right[analysis_col], out_dir / 'wordcloud_right.png')
            except Exception as e:
                print('Wordcloud generation for the 8 summaries failed:', e)

            # Distinctive terms overall: LEFT vs RIGHT (not pre/post)
            try:
                left_dist, right_dist = distinctive_terms_by_logodds(df_left[analysis_col], df_right[analysis_col], n=TOP_N, ngram_range=ngram_range_for_analysis)
                # save combined CSV sorted by log_odds (right positive, left negative)
                df_lr_all = pd.concat([left_dist, right_dist]).sort_values('log_odds', ascending=False)
                df_lr_all.to_csv(out_dir / 'distinctive_left_vs_right.csv', index=False)

                print('\nMost distinctive terms (left vs right) — characteristic of LEFT (negative log_odds):')
                print(left_dist.head(20).to_string(index=False))
                print('\nMost distinctive terms (left vs right) — characteristic of RIGHT (positive log_odds):')
                print(right_dist.head(20).to_string(index=False))

                # plots for left/right distinctive terms
                try:
                    plot_top_terms(left_dist.rename(columns={'log_odds': 'log_odds'}), 'term', 'log_odds', out_dir / 'distinctive_left_vs_right_left.png', 'Distinctive terms — LEFT vs RIGHT (left)')
                    plot_top_terms(right_dist.rename(columns={'log_odds': 'log_odds'}), 'term', 'log_odds', out_dir / 'distinctive_left_vs_right_right.png', 'Distinctive terms — LEFT vs RIGHT (right)')
                except Exception as e:
                    print('Plot generation for left/right distinctive terms failed:', e)
            except Exception as e:
                print('Left vs Right distinctive term computation failed:', e)

            # comparisons: pre_left vs post_left
            compute_pairwise('pre_left', pre_left, 'post_left', post_left, out_dir, ngram_range=ngram_range_for_analysis)
            # pre_right vs post_right
            compute_pairwise('pre_right', pre_right, 'post_right', post_right, out_dir, ngram_range=ngram_range_for_analysis)
            # pre_left vs pre_right
            compute_pairwise('pre_left', pre_left, 'pre_right', pre_right, out_dir, ngram_range=ngram_range_for_analysis)
            # post_left vs post_right
            compute_pairwise('post_left', post_left, 'post_right', post_right, out_dir, ngram_range=ngram_range_for_analysis)
    else:
        print("No 'side' column in input; skipping side comparisons.")
    # plots for top-frequency
    try:
        plot_top_terms(freq_pre.rename(columns={'term': 'term', 'count': 'count'}), 'term', 'count', out_dir / 'freq_pre_bar.png', 'Top freq terms (pre)')
        plot_top_terms(freq_post.rename(columns={'term': 'term', 'count': 'count'}), 'term', 'count', out_dir / 'freq_post_bar.png', 'Top freq terms (post)')
        # plot top distinctive terms (log-odds)
        plot_top_terms(top_pre.rename(columns={'term': 'term', 'log_odds': 'log_odds'}), 'term', 'log_odds', out_dir / 'distinctive_pre_bar.png', 'Distinctive terms (pre vs post)')
        plot_top_terms(top_post.rename(columns={'term': 'term', 'log_odds': 'log_odds'}), 'term', 'log_odds', out_dir / 'distinctive_post_bar.png', 'Distinctive terms (post vs pre)')
    except Exception as e:
        print('Plot generation failed:', e)
    print(f"\nSaved CSV outputs to {out_dir}")

    # sentiment analysis removed by user request — only frequentist outputs remain


if __name__ == '__main__':
    run_all(INPUT_CSV)
