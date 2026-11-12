"""Speaker names in the unseen text: loss on name letters vs other letters, and how often each model guesses the next speaker's first letter.

python name_loss.py main:2000 mem32:2000 mem8:2000 mem1:2000
"""
import json, os, re, sys
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(4)
text = open(os.path.join(HERE, "data/tinyshakespeare.txt")).read()
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
n = int(0.9 * len(text)); val_t = text[n:]
val = torch.tensor([stoi[c] for c in val_t], dtype=torch.long)

HEAD = re.compile(r"^[A-Za-z][A-Za-z' ]{0,30}:$")
first, rest, pos = [], [], 0
ls = val_t.split("\n")
for i, l in enumerate(ls):
    if i and HEAD.match(l) and ls[i - 1].strip() == "" and pos >= 256:
        first.append(pos); rest += range(pos + 1, pos + len(l))
    pos += len(l) + 1
name = set(first) | set(rest)
tl = text[:n].split("\n")
known = {l[:-1] for i, l in enumerate(tl) if i and HEAD.match(l) and tl[i - 1].strip() == ""}
who = {p: val_t[p:val_t.index(":", p)] for p in first}
group = {p: ("in view" if who[p] + ":" in val_t[p - 256:p] else "not in view") + (", seen in training" if who[p] in known else ", new character") for p in first}
g = torch.Generator().manual_seed(11)
other = [q for q in torch.randint(256, len(val), (8000,), generator=g).tolist() if q not in name]

def load(run, step):
    cfg = json.load(open(os.path.join(HERE, "runs", run, "config.json")))
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
            q, k, v = (z.view(b, t, H, c // H).transpose(1, 2) for z in s.qkv(s.ln1(x)).split(c, dim=2))
            x = x + s.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).contiguous().view(b, t, c))
            return x + s.mlp(s.ln2(x))
    class GPT(nn.Module):
        def __init__(s):
            super().__init__()
            s.tok, s.pos = nn.Embedding(len(chars), E), nn.Embedding(B, E)
            s.blocks = nn.Sequential(*[Block() for _ in range(cfg["layers"])])
            s.ln, s.head = nn.LayerNorm(E), nn.Linear(E, len(chars))
            s.drop = nn.Dropout(0)
        def forward(s, idx):
            return s.head(s.ln(s.blocks(s.tok(idx) + s.pos(torch.arange(idx.shape[1])))))
    m = GPT()
    m.load_state_dict({k: v.float() for k, v in torch.load(os.path.join(HERE, "runs", run, "weights", f"w_{step:05d}.pt"), map_location="cpu").items()})
    return m.eval(), B

@torch.no_grad()
def loss_at(m, B, P):
    tot = 0.0
    for i in range(0, len(P), 64):
        Q = P[i:i + 64]
        tot += F.cross_entropy(m(torch.stack([val[q - B:q] for q in Q]))[:, -1], val[Q], reduction="sum").item()
    return tot / len(P)

@torch.no_grad()
def guesses(m, B, P):
    r = {}
    for i in range(0, len(P), 64):
        Q = P[i:i + 64]
        pr = F.softmax(m(torch.stack([val[q - B:q] for q in Q]))[:, -1], -1); pt = pr.gather(1, val[Q][:, None])[:, 0]
        rank = (pr > pt[:, None]).sum(1)
        for q, k, a, b in zip(Q, rank.tolist(), pt.tolist(), pr.max(1).values.tolist()): r[q] = (k, a, b)
    return r

def acc(r, P):
    return {"n": len(P), "top1": round(sum(r[q][0] == 0 for q in P) / len(P), 4), "top3": round(sum(r[q][0] < 3 for q in P) / len(P), 4),
            "p_right": round(sum(r[q][1] for q in P) / len(P), 4), "p_top_guess": round(sum(r[q][2] for q in P) / len(P), 4)}

OUTP = os.path.join(HERE, "runs/eval/name_loss.json")
old = json.load(open(OUTP))["runs"] if os.path.exists(OUTP) else {}
out = {"n_first": len(first), "n_rest": len(rest), "n_other": len(other), "groups": {g: sum(v == g for v in group.values()) for g in sorted(set(group.values()))}, "runs": old}
for spec in sys.argv[1:]:
    run, step = spec.split(":"); m, B = load(run, int(step))
    r = {"first_letter_of_name": loss_at(m, B, first), "rest_of_name": loss_at(m, B, rest), "other_letters": loss_at(m, B, other)}
    out["runs"][spec] = {k: round(v, 4) for k, v in r.items()}
    G = guesses(m, B, first)
    out["runs"][spec]["next_speaker"] = {"all": acc(G, first), **{g: acc(G, [q for q in first if group[q] == g]) for g in sorted(set(group.values()))}}
    print(spec, out["runs"][spec], flush=True)
json.dump(out, open(OUTP, "w"), indent=1)
print({k: v for k, v in out.items() if k != "runs"})
