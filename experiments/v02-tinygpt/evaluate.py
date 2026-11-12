"""Evaluate a trained run on CPU: full-memory loss, loss by context length, scene structure, copying, and per-letter probabilities.

python evaluate.py --run main --step 2000
python evaluate.py --real            (the same structure metrics on real held-out Shakespeare)
"""
import argparse, json, os, re
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--run")
p.add_argument("--step", type=int)
p.add_argument("--real", action="store_true")
p.add_argument("--data", default="data/tinyshakespeare.txt")
p.add_argument("--n_loss", type=int, default=8000)
p.add_argument("--n_curve", type=int, default=1000)
p.add_argument("--n_samples", type=int, default=48)
p.add_argument("--n_chars", type=int, default=700)
p.add_argument("--temp", type=float, default=0.8)
p.add_argument("--threads", type=int, default=6)
a = p.parse_args()
torch.set_num_threads(a.threads)

text = open(os.path.join(HERE, a.data)).read()
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
enc = lambda s: [stoi[c] for c in s]; dec = lambda ids: "".join(chars[i] for i in ids)
n = int(0.9 * len(text)); train_t, val_t = text[:n], text[n:]
val = torch.tensor(enc(val_t), dtype=torch.long)

HEAD = re.compile(r"^[A-Za-z][A-Za-z' ]{0,30}:$")
def turns(s):
    ls = s.split("\n"); return [l[:-1] for i, l in enumerate(ls) if HEAD.match(l) and (i == 0 or ls[i - 1].strip() == "")]
SPEAKERS = set(turns(text))
TT = turns(text); TALK = {frozenset(q) for q in zip(TT, TT[1:]) if q[0] != q[1]}
CORPUS = set(re.findall(r"[a-z]+", train_t.lower()))

def structure(samples):
    T = V = same = pairs = new = after_first = hand = talk = 0; lines = []; words = []; copied = total = 0
    for s in samples:
        names = turns(s); T += len(names); V += sum(x in SPEAKERS for x in names)
        seen = set()
        for i, x in enumerate(names):
            if i: pairs += 1; same += x == names[i - 1]; after_first += 1; new += x not in seen
            if i and x != names[i - 1]: hand += 1; talk += frozenset((x, names[i - 1])) in TALK
            seen.add(x)
        ls = s.split("\n"); lines += [l for i, l in enumerate(ls) if l.strip() and not (HEAD.match(l) and (i == 0 or ls[i - 1].strip() == ""))]
        words += [w for w in re.findall(r"[A-Za-z]+", s) if len(w) >= 3]
        cov = [False] * len(s)
        for i in range(len(s) - 29):
            if s[i:i + 30] in train_t:
                for j in range(i, i + 30): cov[j] = True
        copied += sum(cov); total += len(s)
    L = [len(l) for l in lines]
    return {"turns": T, "turns_per_1000_chars": round(1000 * T / max(1, total), 2),
            "valid_speaker_share": round(V / max(1, T), 4),
            "same_speaker_twice_share": round(same / max(1, pairs), 4),
            "new_speaker_share_after_first": round(new / max(1, after_first), 4),
            "handovers": hand, "handover_pairs_that_talk_share": round(talk / max(1, hand), 4),
            "line_len_mean": round(sum(L) / max(1, len(L)), 1), "lines_over_80_share": round(sum(x > 80 for x in L) / max(1, len(L)), 4),
            "words_3plus": len(words), "real_word_share": round(sum(w.lower() in CORPUS for w in words) / max(1, len(words)), 4),
            "copied_30char_share": round(copied / max(1, total), 4), "chars": total}

OUT = os.path.join(HERE, "runs", "eval"); os.makedirs(OUT, exist_ok=True)
if a.real:
    g = torch.Generator().manual_seed(3)
    starts = torch.randint(0, len(val_t) - a.n_chars, (a.n_samples,), generator=g).tolist()
    r = {"name": "real_val_text", "structure": structure([val_t[i:i + a.n_chars] for i in starts]), "n_samples": a.n_samples, "n_chars": a.n_chars}
    json.dump(r, open(os.path.join(OUT, "real.json"), "w"), indent=1); print(json.dumps(r["structure"])); raise SystemExit

cfg = json.load(open(os.path.join(HERE, "runs", a.run, "config.json")))
B, E, H = cfg["block"], cfg["embd"], cfg["heads"]

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(E), nn.LayerNorm(E)
        s.qkv, s.proj = nn.Linear(E, 3 * E), nn.Linear(E, E)
        s.mlp = nn.Sequential(nn.Linear(E, 4 * E), nn.GELU(), nn.Linear(4 * E, E), nn.Dropout(0))
        s.drop = nn.Dropout(0)
    def forward(s, x):
        b, t, c = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(c, dim=2)
        q, k, v = (z.view(b, t, H, c // H).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(b, t, c))
        return x + s.mlp(s.ln2(x))

class GPT(nn.Module):
    def __init__(s, V):
        super().__init__()
        s.tok, s.pos = nn.Embedding(V, E), nn.Embedding(B, E)
        s.blocks = nn.Sequential(*[Block() for _ in range(cfg["layers"])])
        s.ln, s.head = nn.LayerNorm(E), nn.Linear(E, V)
        s.drop = nn.Dropout(0)
    def forward(s, idx):
        x = s.tok(idx) + s.pos(torch.arange(idx.shape[1]))
        return s.head(s.ln(s.blocks(x)))

model = GPT(len(chars))
sd = torch.load(os.path.join(HERE, "runs", a.run, "weights", f"w_{a.step:05d}.pt"), map_location="cpu")
model.load_state_dict({k: v.float() for k, v in sd.items()}); model.eval()

@torch.no_grad()
def full_memory_loss():
    g = torch.Generator().manual_seed(11)
    pos = torch.randint(256, len(val), (a.n_loss,), generator=g)
    tot = 0.0
    for i in range(0, a.n_loss, 64):
        P = pos[i:i + 64]
        x = torch.stack([val[q - B:q] for q in P.tolist()]); y = val[P]
        tot += F.cross_entropy(model(x)[:, -1], y, reduction="sum").item()
    return tot / a.n_loss

@torch.no_grad()
def loss_by_context():
    g = torch.Generator().manual_seed(12)
    st = torch.randint(0, len(val) - B - 1, (a.n_curve,), generator=g)
    acc = torch.zeros(B)
    for i in range(0, a.n_curve, 64):
        S = st[i:i + 64].tolist()
        x = torch.stack([val[q:q + B] for q in S]); y = torch.stack([val[q + 1:q + B + 1] for q in S])
        acc += F.cross_entropy(model(x).transpose(1, 2), y, reduction="none").sum(0)
    return [round(v, 4) for v in (acc / a.n_curve).tolist()]

@torch.no_grad()
def generate(prompts, n_chars, seed, keep_probs):
    g = torch.Generator().manual_seed(seed)
    w = max(len(q) for q in prompts)
    idx = torch.tensor([enc("\n" * (w - len(q)) + q) for q in prompts])
    trace = [[] for _ in prompts]
    for _ in range(n_chars):
        logits = model(idx[:, -B:])[:, -1]
        p1 = F.softmax(logits, -1)
        nxt = torch.multinomial(F.softmax(logits / a.temp, -1), 1, generator=g)
        if keep_probs:
            tp = torch.topk(p1, 5)
            for r in range(len(prompts)):
                c = nxt[r, 0].item()
                trace[r].append({"c": chars[c], "p": round(p1[r, c].item(), 4),
                                 "top": [[chars[j], round(v, 4)] for v, j in zip(tp.values[r].tolist(), tp.indices[r].tolist())]})
        idx = torch.cat([idx, nxt], 1)
    return [dec(r) for r in idx[:, w:].tolist()], trace

name = f"{a.run}_{a.step}"
print(f"[{name}] block {B}", flush=True)
fm = full_memory_loss(); print(f"[{name}] full-memory val loss {fm:.4f}", flush=True)
curve = loss_by_context(); print(f"[{name}] loss at context 1 / {B}: {curve[0]:.3f} / {curve[-1]:.3f}", flush=True)
samples, _ = generate(["\n"] * a.n_samples, a.n_chars, 1, False)
st = structure(samples); print(f"[{name}] {json.dumps(st)}", flush=True)
CHAT = "YOU:\nWhat is the meaning of life?\n\nHAMLET:\n"
traced, tr = generate(["\n"] * 4 + ["ROMEO:\n"] * 4 + [CHAT] * 8, 400, 2, True)
res = {"name": name, "run": a.run, "step": a.step, "block": B, "temp": a.temp,
       "full_memory_val_loss": round(fm, 4), "n_loss_positions": a.n_loss,
       "loss_by_context": curve, "n_curve_windows": a.n_curve,
       "structure": st, "n_samples": a.n_samples, "n_chars": a.n_chars, "samples": samples,
       "traced": [{"prompt": q, "text": t, "letters": l} for q, t, l in zip(["\n"] * 4 + ["ROMEO:\n"] * 4 + [CHAT] * 8, traced, tr)]}
json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
print(f"[{name}] saved runs/eval/{name}.json", flush=True)
