import os
import gc
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from functools import partial
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import umap
import hdbscan
from bertopic import BERTopic
import unicodedata
from sentence_transformers import SentenceTransformer

#setting up the stopwords
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")