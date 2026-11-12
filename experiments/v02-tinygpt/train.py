"""Character-level GPT trained from scratch on Tiny Shakespeare, logging everything the video shows.

python train.py --name main
python train.py --name mem1 --block 1        (same model, it only ever sees the previous letter)
"""
import argparse, json, math, os, re, time
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--data", default="data/tinyshakespeare.txt")
p.add_argument("--block", type=int, default=256)
p.add_argument("--layers", type=int, default=6)
p.add_argument("--heads", type=int, default=6)
p.add_argument("--embd", type=int, default=384)
p.add_argument("--dropout", type=float, default=0.2)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--iters", type=int, default=5000)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=1337)
p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
p.add_argument("--snaps", default="0,25,50,100,200,300,500,750,1000,1500,2000,3000,4000,5000")
p.add_argument("--keep", default="0,100,300,1000,2000,5000")
a = p.parse_args()

torch.manual_seed(a.seed)
text = open(os.path.join(HERE, a.data)).read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
enc = lambda s: [stoi[c] for c in s]
dec = lambda ids: "".join(chars[i] for i in ids)
data = torch.tensor(enc(text), dtype=torch.long)
n = int(0.9 * len(data))
train_d, val_d = data[:n], data[n:]
dev = a.device
OUT = os.path.join(HERE, "runs", a.name)
os.makedirs(os.path.join(OUT, "weights"), exist_ok=True)

def batch(split, g=None):
    d = train_d if split == "train" else val_d
    ix = torch.randint(len(d) - a.block - 1, (a.batch,), generator=g)
    x = torch.stack([d[i:i + a.block] for i in ix]); y = torch.stack([d[i + 1:i + a.block + 1] for i in ix])
    return x.to(dev), y.to(dev)

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(a.embd), nn.LayerNorm(a.embd)
        s.qkv, s.proj = nn.Linear(a.embd, 3 * a.embd), nn.Linear(a.embd, a.embd)
        s.mlp = nn.Sequential(nn.Linear(a.embd, 4 * a.embd), nn.GELU(), nn.Linear(4 * a.embd, a.embd), nn.Dropout(a.dropout))
        s.drop = nn.Dropout(a.dropout)
    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, a.heads, C // a.heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=a.dropout if s.training else 0.0)
        x = x + s.drop(s.proj(y.transpose(1, 2).contiguous().view(B, T, C)))
        return x + s.mlp(s.ln2(x))

class GPT(nn.Module):
    def __init__(s, V):
        super().__init__()
        s.tok, s.pos = nn.Embedding(V, a.embd), nn.Embedding(a.block, a.embd)
        s.blocks = nn.Sequential(*[Block() for _ in range(a.layers)])
        s.ln, s.head = nn.LayerNorm(a.embd), nn.Linear(a.embd, V)
        s.drop = nn.Dropout(a.dropout)
        s.apply(lambda m: (nn.init.normal_(m.weight, 0, 0.02), m.bias is not None and nn.init.zeros_(m.bias)) if isinstance(m, nn.Linear) else nn.init.normal_(m.weight, 0, 0.02) if isinstance(m, nn.Embedding) else None)
    def forward(s, idx):
        T = idx.shape[1]
        x = s.drop(s.tok(idx) + s.pos(torch.arange(T, device=idx.device)))
        return s.head(s.ln(s.blocks(x)))

model = GPT(len(chars)).to(dev)
n_params = sum(t.numel() for t in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.1)
lr_at = lambda it: a.lr * it / 100 if it < 100 else 1e-4 + 0.5 * (a.lr - 1e-4) * (1 + math.cos(math.pi * (it - 100) / max(1, a.iters - 100)))

@torch.no_grad()
def eval_loss(k=20):
    model.eval(); g = torch.Generator().manual_seed(7); r = {}
    for sp in ("train", "val"):
        r[sp] = sum(F.cross_entropy(model(x).view(-1, len(chars)), y.view(-1)).item() for x, y in (batch(sp, g) for _ in range(k))) / k
    model.train(); return r

@torch.no_grad()
def sample_rows(prompts, n_chars, temp, seed):
    model.eval(); g = torch.Generator(device="cpu").manual_seed(seed)
    w = max(len(q) for q in prompts)
    idx = torch.tensor([enc("\n" * (w - len(q)) + q) for q in prompts], device=dev)
    for _ in range(n_chars):
        pr = F.softmax(model(idx[:, -a.block:])[:, -1].float().cpu() / temp, -1)
        idx = torch.cat([idx, torch.multinomial(pr, 1, generator=g).to(dev)], 1)
    model.train(); return [dec(r) for r in idx[:, w:].tolist()]

@torch.no_grad()
def probe(text, k=5):
    model.eval(); ids = torch.tensor([enc(text)], device=dev)
    if len(text) <= a.block: P = F.softmax(model(ids)[0].float(), -1).cpu()
    else: P = torch.stack([F.softmax(model(ids[:, max(0, t + 1 - a.block):t + 1])[0, -1].float(), -1).cpu() for t in range(len(text))])
    out = []
    for t in range(1, len(text)):
        pr = P[t - 1]; tp = torch.topk(pr, k)
        out.append({"ctx_end": text[t - 1], "next": text[t], "p_next": round(pr[stoi[text[t]]].item(), 4),
                    "top": [[chars[i], round(v, 4)] for v, i in zip(tp.values.tolist(), tp.indices.tolist())]})
    model.train(); return out

CORPUS = set(re.findall(r"[a-z]+", text.lower()))
DICT = set(w.strip().lower() for w in open("/usr/share/dict/words")) if os.path.exists("/usr/share/dict/words") else set()
def word_stats(s):
    ws = [w for w in re.findall(r"[A-Za-z]+", s) if len(w) >= 3]
    if not ws: return {"words_3plus": 0, "in_shakespeare": 0, "in_dictionary": 0}
    return {"words_3plus": len(ws), "in_shakespeare": round(sum(w.lower() in CORPUS for w in ws) / len(ws), 4),
            "in_dictionary": round(sum(w.lower() in DICT for w in ws) / len(ws), 4) if DICT else None}

PROMPTS = {"open": "\n", "romeo": "ROMEO:\n", "chat": "YOU:\nWhat is the meaning of life?\n\nHAMLET:\n"}
PROBE = "ROMEO:\nBut soft, what light through yonder window breaks?"
SNAPS = sorted(int(x) for x in a.snaps.split(",") if int(x) <= a.iters)
KEEP = set(int(x) for x in a.keep.split(","))

def snapshot(it, elapsed):
    L = eval_loss()
    keys = list(PROMPTS) + [f"free{i}" for i in range(6)]
    rows = sample_rows([PROMPTS[k] if k in PROMPTS else "\n" for k in keys], 400, 0.8, 1)
    smp = dict(zip(keys, rows))
    long = "\n".join(smp[f"free{i}"] for i in range(6))
    snap = {"iter": it, "seconds": round(elapsed, 1), "chars_seen": it * a.batch * a.block, "loss": L,
            "samples": smp, "words": word_stats(long), "probe": probe(PROBE)}
    json.dump(snap, open(os.path.join(OUT, f"snap_{it:05d}.json"), "w"), indent=1)
    if it in KEEP:
        torch.save({k: v.half() for k, v in model.state_dict().items()}, os.path.join(OUT, "weights", f"w_{it:05d}.pt"))
    print(f"[{a.name}] it {it:5d}  {elapsed/60:5.1f} min  train {L['train']:.3f}  val {L['val']:.3f}  real words {snap['words']}  |  {smp['romeo'][:70]!r}", flush=True)

cfg = dict(vars(a), n_params=n_params, vocab=len(chars), chars="".join(chars), train_chars=len(train_d), val_chars=len(val_d),
           torch=torch.__version__, prompts=PROMPTS, probe=PROBE)
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=1)
print(f"[{a.name}] {n_params:,} parameters, vocab {len(chars)}, block {a.block}, device {dev}", flush=True)

log = open(os.path.join(OUT, "loss.csv"), "w"); log.write("iter,seconds,lr,train_batch_loss\n")
gt = torch.Generator().manual_seed(a.seed)
t0, paused = time.time(), 0.0
for it in range(a.iters + 1):
    if it in SNAPS:
        s0 = time.time(); snapshot(it, s0 - t0 - paused); paused += time.time() - s0
    if it == a.iters: break
    for gp in opt.param_groups: gp["lr"] = lr_at(it)
    x, y = batch("train", gt)
    loss = F.cross_entropy(model(x).view(-1, len(chars)), y.view(-1))
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if it % 10 == 0: log.write(f"{it},{time.time() - t0 - paused:.2f},{lr_at(it):.6f},{loss.item():.4f}\n"); log.flush()
train_s = time.time() - t0 - paused
json.dump(dict(cfg, train_seconds=round(train_s, 1), final=eval_loss(100)), open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print(f"[{a.name}] done: {train_s/60:.1f} min of training (snapshot time excluded)", flush=True)
