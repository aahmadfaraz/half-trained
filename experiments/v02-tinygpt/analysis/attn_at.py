"""Attention from the last position of PROMPT+TEXT back over the context window, for every layer and head.
python analysis/attn_at.py RUN STEP PROMPT_FILE TEXT_FILE OUT_JSON
"""
import json, os, sys
import torch, torch.nn as nn, torch.nn.functional as F

EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
run, step, pf, tf, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
prompt, text = open(pf).read(), open(tf).read()
data = open(os.path.join(EXP, "data/tinyshakespeare.txt")).read()
chars = sorted(set(data)); stoi = {c: i for i, c in enumerate(chars)}
cfg = json.load(open(os.path.join(EXP, "runs", run, "config.json")))
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
        w = (q @ k.transpose(-2, -1)) / (c // H) ** 0.5
        w = w.masked_fill(torch.triu(torch.ones(t, t, dtype=torch.bool), 1), float("-inf")).softmax(-1)
        s.last = w[0, :, -1, :].detach()
        x = x + s.proj((w @ v).transpose(1, 2).contiguous().view(b, t, c))
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
m.load_state_dict({k: v.float() for k, v in torch.load(os.path.join(EXP, "runs", run, "weights", f"w_{step:05d}.pt"), map_location="cpu").items()})
m.eval()
full = prompt + text
ids = [stoi[c] for c in full][-B:]
with torch.no_grad():
    lg = m(torch.tensor([ids]))[0, -1]
p = F.softmax(lg, -1); top = torch.topk(p, 5)
att = [[[round(x, 5) for x in blk.last[h].tolist()] for h in range(H)] for blk in m.blocks]
json.dump({"run": run, "step": step, "full": full, "start": len(full) - len(ids), "window": len(ids),
           "top": [[chars[j], round(v, 4)] for v, j in zip(top.values.tolist(), top.indices.tolist())], "att": att}, open(out, "w"))
print(out, "len", len(full), "window starts at", len(full) - len(ids), "top", [[chars[j], round(v, 3)] for v, j in zip(top.values.tolist(), top.indices.tolist())])
