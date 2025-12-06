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
from sklearn.decomposition import LatentDirichletAllocation
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity
from gensim import models
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import umap
import hdbscan
from bertopic import BERTopic
import unicodedata
from sentence_transformers import SentenceTransformer
from collections import Counter
from scipy.spatial.distance import jensenshannon
import spacy
import re
#setting up the stopwords
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
