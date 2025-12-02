import os, gc, time, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from functools import partial
from tqdm import tqdm

MODEL_NAME = "intfloat/multilingual-e5-base"
TEXT_COL   = "text_clean"
BATCH_SIZE = 8
NUM_WORKERS = 4
MAX_LEN = 512

from google.colab import files
uploaded = files.upload()
CSV_PATH = list(uploaded.keys())[0]
try:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(CSV_PATH, encoding="latin-1")

print("Loaded:", CSV_PATH, "rows:", len(df))

def mean_pool_last_hidden(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

class LazyDocDataset(Dataset):
    def __init__(self, texts):
        self.samples = [{"doc_id": i, "text": str(t) if not isinstance(t, str) else t}
                        for i, t in enumerate(texts)]
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def collate_tokenize(batch, tokenizer, max_length, overlap_stride=128):
    texts   = ["passage: " + b["text"] for b in batch]
    doc_ids = [b["doc_id"] for b in batch]

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        stride=overlap_stride,                 # <-- overlap ON
        return_overflowing_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=False,          # not needed
        padding=False,
    )

    # Flatten with mapping back to original sample
    flat_ids, flat_am, flat_doc_ids, flat_tok_lens, flat_pos_in_doc = [], [], [], [], []
    # position of the chunk within its source doc (0-based)
    # HuggingFace gives 'overflow_to_sample_mapping' list aligned with each produced chunk
    pos_counters = {}  # sample_index -> next_chunk_pos
    for sample_idx in enc["overflow_to_sample_mapping"]:
        pos = pos_counters.get(sample_idx, 0)
        pos_counters[sample_idx] = pos + 1
        flat_pos_in_doc.append(pos)

    for ids, am, src_idx, pos in zip(
        enc["input_ids"], enc["attention_mask"], enc["overflow_to_sample_mapping"], flat_pos_in_doc
    ):
        flat_ids.append(torch.tensor(ids, dtype=torch.long))
        flat_am.append(torch.tensor(am,  dtype=torch.long))
        flat_doc_ids.append(doc_ids[src_idx])
        flat_tok_lens.append(int(sum(am)))         # real tokens in this chunk

    # pad to a batch tensor
    if not flat_ids:
        return {"input_ids": torch.empty(0, dtype=torch.long),
                "attention_mask": torch.empty(0, dtype=torch.long),
                "doc_ids": torch.empty(0, dtype=torch.long),
                "tok_len": torch.empty(0, dtype=torch.long),
                "chunk_pos": torch.empty(0, dtype=torch.long)}

    L = max(len(x) for x in flat_ids)
    def pad(x, v):
        if len(x) < L:
            return torch.cat([x, torch.full((L-len(x),), v, dtype=x.dtype)])
        return x

    pad_id = tokenizer.pad_token_id or 0
    input_ids = torch.stack([pad(x, pad_id) for x in flat_ids])
    attention = torch.stack([pad(x, 0) for x in flat_am])
    doc_ids_t = torch.tensor(flat_doc_ids, dtype=torch.long)
    tok_lens  = torch.tensor(flat_tok_lens, dtype=torch.long)
    chunk_pos = torch.tensor(flat_pos_in_doc, dtype=torch.long)  # 0,1,2,...

    return {"input_ids": input_ids, "attention_mask": attention,
            "doc_ids": doc_ids_t, "tok_len": tok_lens, "chunk_pos": chunk_pos}


def hierarchical_e5_embeddings(texts, batch_size=BATCH_SIZE, max_length=MAX_LEN, device=None, use_fp16=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    ds = LazyDocDataset(texts)
    collate = partial(collate_tokenize, tokenizer=tok, max_length=max_length, overlap_stride=128)  # 25% overlap of 512
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,                        # Colab/Linux: safe to use workers
        pin_memory=(device=="cuda"),
        persistent_workers=False,
        prefetch_factor=4,
        collate_fn=collate,
    )

    H = mdl.config.hidden_size
    by_sum, by_w = {}, {}


    amp_ctx = torch.amp.autocast
    amp_args = {"device_type": "cuda"} if device=="cuda" else {"device_type": "cpu"}

    start = time.time()
    with torch.inference_mode():
        with amp_ctx(**amp_args):
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            for i, batch in enumerate(tqdm(dl, total=len(dl), desc="Embedding")):
                if batch["input_ids"].numel() == 0:
                    continue
                out = mdl(
                    input_ids=batch["input_ids"].to(device, non_blocking=True),
                    attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                )
                chunk_emb = mean_pool_last_hidden(
                    out.last_hidden_state,
                    batch["attention_mask"].to(device, non_blocking=True)
                ).float().cpu().numpy()

                # effective per-chunk weight: avoid counting the overlapped tail twice
                stride = 128  # keep in sync with collate
                pos   = batch["chunk_pos"].numpy()
                w_raw = batch["tok_len"].numpy().astype(np.float32)

                # assume 2 specials per chunk (BERT-style [CLS]/[SEP]); robustly clip
                specials = 2.0
                content_len = np.clip(w_raw - specials, 1.0, None)

                # overlap applies only to chunks after the first (pos > 0)
                overlap = np.where(pos > 0, np.minimum(stride, content_len - 1.0), 0.0)

                # effective per-chunk weight
                w_eff = np.clip(content_len - overlap, 1.0, None)

                dids = batch["doc_ids"].numpy()
                for e, weight, d in zip(chunk_emb, w_eff, dids):
                    d = int(d)
                    if d not in by_sum:
                        by_sum[d] = np.zeros((H,), dtype=np.float32)
                        by_w[d]   = 0.0
                    by_sum[d] += e * weight
                    by_w[d]   += weight

                #clean later
                if device=="cuda" and (i+1) % 200 == 0:
                    torch.cuda.synchronize()
                    used = torch.cuda.memory_allocated()/1024**2
                    reserved = torch.cuda.memory_reserved()/1024**2
                    rate = (i+1)*batch_size/max(time.time()-t0,1e-9)
                    print(f"[{i+1}/{len(dl)}] GPU used={used:.0f}MB | reserved={reserved:.0f}MB | {rate:.1f} ch/s")

    num_docs = len(texts)
    doc_embeddings = np.zeros((num_docs, H), dtype=np.float32)
    for d in range(num_docs):
        if d in by_sum and by_w[d] > 0:
            doc_embeddings[d] = by_sum[d] / by_w[d]

    if device=="cuda":
        del mdl; gc.collect(); torch.cuda.empty_cache()
    else:
        del mdl; gc.collect()

    elapsed = time.time()-start
    print(f"Done in {elapsed/60:.2f} min.")
    return doc_embeddings

texts = df[TEXT_COL].astype(str).tolist()
emb = hierarchical_e5_embeddings(texts)
np.save("e5base_doc_embeddings.npy", emb)
emb.shape