"""
Reproduce Sentence-BERT (Reimers & Gurevych, 2019 — arXiv:1908.10084)
with pure PyTorch + HuggingFace transformers only (NO sentence-transformers).

Recipe (paper, section 3 "Classification Objective Function"):
    - encoder: a BERT-family model, mean-pooled over tokens -> one vector per sentence
    - for a pair (a, b) with vectors (u, v):
          feats  = [u ; v ; |u - v|]           (dim = 3 * hidden)
          logits = W_t @ feats                  (W_t: 3*hidden -> 3 NLI classes)
          loss   = cross_entropy(logits, nli_label)
    - train data: SNLI + MultiNLI (AllNLI), labels {entailment, neutral, contradiction}
    - eval: encode STS sentences, cosine similarity, Spearman rho vs gold (NO fine-tuning on STS)

Deviation from the paper: we default to distilbert-base-uncased (what we have cached),
not bert-base-uncased. Pass --model bert-base-uncased for the paper-exact encoder.
"""
import argparse, time, math, random
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, logging as hf_logging
from datasets import load_dataset, concatenate_datasets
from scipy.stats import spearmanr

hf_logging.set_verbosity_error()


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mean_pool(last_hidden, attention_mask):
    """Masked mean over tokens: average only the real (non-pad) tokens."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)      # (B, T, 1)
    summed = (last_hidden * mask).sum(dim=1)                        # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # (B, 1)
    return summed / counts                                         # (B, H)


class PairDataset(Dataset):
    def __init__(self, prem, hyp, label):
        self.prem, self.hyp, self.label = prem, hyp, label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        return self.prem[i], self.hyp[i], self.label[i]


class TokenizingCollator:
    """Tokenize inside the DataLoader so worker processes do it in parallel,
    overlapping CPU tokenization with GPU compute (the loop was GPU-starved)."""
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        a = [x[0] for x in batch]
        b = [x[1] for x in batch]
        y = torch.tensor([x[2] for x in batch], dtype=torch.long)
        enc_a = self.tokenizer(a, padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt")
        enc_b = self.tokenizer(b, padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt")
        return enc_a, enc_b, y


def load_allnli(limit=None):
    """SNLI + MultiNLI train, filtering out label == -1 (no gold)."""
    snli = load_dataset("stanfordnlp/snli", split="train")
    mnli = load_dataset("nyu-mll/multi_nli", split="train").select_columns(
        ["premise", "hypothesis", "label"]
    )
    both = concatenate_datasets([snli, mnli])
    both = both.filter(lambda ex: ex["label"] != -1)
    if limit:
        both = both.shuffle(seed=42).select(range(min(limit, len(both))))
    return both["premise"], both["hypothesis"], both["label"]


def encode(model, tokenizer, sentences, device, max_len, batch_size=64, amp=None):
    """Embed a list of sentences (eval-time, no grad)."""
    amp = amp or nullcontext()
    model.eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            chunk = sentences[i:i + batch_size]
            enc = tokenizer(chunk, padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt").to(device)
            with amp:
                out = model(**enc).last_hidden_state
            vecs.append(mean_pool(out, enc["attention_mask"]).float().cpu())
    return torch.cat(vecs, dim=0)


# The 7 STS benchmarks the paper averages (Table 1), all mirrored by MTEB with a
# uniform schema: sentence1 / sentence2 / score, gold human similarity, test split.
STS_TASKS = {
    "STS12":  "mteb/sts12-sts",
    "STS13":  "mteb/sts13-sts",
    "STS14":  "mteb/sts14-sts",
    "STS15":  "mteb/sts15-sts",
    "STS16":  "mteb/sts16-sts",
    "STS-B":  "mteb/stsbenchmark-sts",
    "SICK-R": "mteb/sickr-sts",
}


def eval_one(model, tokenizer, device, max_len, hf_name):
    """Encode -> cosine -> Spearman rho (x100) on one STS task's test split."""
    ds = load_dataset(hf_name, split="test")
    a = encode(model, tokenizer, ds["sentence1"], device, max_len)
    b = encode(model, tokenizer, ds["sentence2"], device, max_len)
    cos = F.cosine_similarity(a, b).numpy()
    return spearmanr(cos, ds["score"]).correlation * 100.0


def eval_sts(model, tokenizer, device, max_len, task="STS-B"):
    """Cheap single-task monitor used during training (STS-B test by default)."""
    return eval_one(model, tokenizer, device, max_len, STS_TASKS[task])


def eval_suite(model, tokenizer, device, max_len, tag=""):
    """Full 7-task STS suite; prints per-task rho and the paper's averaged number."""
    scores = {}
    for name, hf_name in STS_TASKS.items():
        scores[name] = eval_one(model, tokenizer, device, max_len, hf_name)
    avg = sum(scores.values()) / len(scores)
    row = "  ".join(f"{k} {v:5.2f}" for k, v in scores.items())
    print(f"[STS suite {tag}] {row}  ||  AVG {avg:5.2f}", flush=True)
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--warmup-frac", type=float, default=0.10)
    ap.add_argument("--limit", type=int, default=None, help="subset of AllNLI (for smoke tests)")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=0, help="steps between STS evals (0 = only start/end)")
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast for the training forward (CUDA only)")
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers that tokenize in parallel")
    ap.add_argument("--out", default="sbert_distilbert.pt")
    args = ap.parse_args()

    random.seed(42); torch.manual_seed(42)
    device = pick_device()

    # bf16 autocast: real speedup on A100/H100; no GradScaler needed (unlike fp16).
    # Only CUDA is worth it here — MPS autocast support is flaky, so we no-op there.
    if args.bf16 and device == "cuda":
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        amp_note = "bf16-autocast"
    else:
        if args.bf16:
            print(f"[warn] --bf16 ignored: device={device} (CUDA only); running fp32", flush=True)
        amp = nullcontext()
        amp_note = "fp32"
    print(f"device={device}  model={args.model}  precision={amp_note}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    hidden = model.config.hidden_size
    head = torch.nn.Linear(3 * hidden, 3).to(device)   # paper's W_t

    print(">>> BEFORE training", flush=True)
    eval_suite(model, tokenizer, device, args.max_len, tag="before")

    prem, hyp, label = load_allnli(args.limit)
    print(f"AllNLI pairs: {len(label):,}", flush=True)
    loader = DataLoader(PairDataset(prem, hyp, label), batch_size=args.batch_size,
                        shuffle=True, collate_fn=TokenizingCollator(tokenizer, args.max_len),
                        num_workers=args.workers, pin_memory=(device == "cuda"),
                        persistent_workers=(args.workers > 0))

    total_steps = len(loader) * args.epochs
    warmup = int(args.warmup_frac * total_steps)
    optim = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr)
    sched = get_linear_schedule_with_warmup(optim, warmup, total_steps)
    print(f"total_steps={total_steps:,}  warmup={warmup:,}", flush=True)

    step = 0
    t0 = time.time()
    running = 0.0
    for epoch in range(args.epochs):
        model.train(); head.train()
        for enc_a, enc_b, y in loader:
            y = y.to(device, non_blocking=True)
            enc_a = enc_a.to(device)
            enc_b = enc_b.to(device)
            with amp:
                u = mean_pool(model(**enc_a).last_hidden_state, enc_a["attention_mask"])
                v = mean_pool(model(**enc_b).last_hidden_state, enc_b["attention_mask"])
                feats = torch.cat([u, v, (u - v).abs()], dim=1)
                loss = F.cross_entropy(head(feats), y)

            optim.zero_grad(); loss.backward(); optim.step(); sched.step()
            running += loss.item(); step += 1

            if step % args.log_every == 0:
                rate = step / (time.time() - t0)
                print(f"step {step:>6}/{total_steps}  loss {running/args.log_every:.4f}  "
                      f"{rate:.1f} it/s  eta {(total_steps-step)/max(rate,1e-9)/60:.1f}m", flush=True)
                running = 0.0
            if args.eval_every and step % args.eval_every == 0:
                print(f"  [step {step}] STS-B(test) Spearman: "
                      f"{eval_sts(model, tokenizer, device, args.max_len):.2f}", flush=True)
                model.train(); head.train()

    print(">>> AFTER training", flush=True)
    rho = eval_suite(model, tokenizer, device, args.max_len, tag="after")
    torch.save({"model": model.state_dict(), "head": head.state_dict(), "args": vars(args)}, args.out)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
