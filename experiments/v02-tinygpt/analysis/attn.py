"""Real attention: which earlier letters each head looks at when the model is about to write position `at`.

python analysis/attn.py RUN STEP TEXT_FILE AT OUT_JSON   (AT = index in the text of the letter being predicted)
"""
import json, os, sys, math
import torch, torch.nn.functional as F
sys.argv, args = sys.argv[:1], sys.argv[1:]
run, step, tf, at, out = args[0], int(args[1]), args[2], int(args[3]), args[4]
EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
data = open(os.path.join(EXP, "data/tinyshakespeare.txt")).read(); chars = sorted(set(data)); stoi = {c: i for i, c in enumerate(chars)}
cfg = json.load(open(os.path.join(EXP, "runs", run, "config.json"))); B, E, H = cfg["block"], cfg["embd"], cfg["heads"]
sd = {k: v.float() for k, v in torch.load(os.path.join(EXP, "runs", run, "weights", f"w_{step:05d}.pt"), map_location="cpu").items()}
text = open(tf).read()
ctx = text[max(0, at - B):at]; idx = torch.tensor([[stoi[c] for c in ctx]]); T = idx.shape[1]
x = sd["tok.weight"][idx] + sd["pos.weight"][:T]
att_all = []
for l in range(cfg["layers"]):
    p = f"blocks.{l}."
    h = F.layer_norm(x, (E,), sd[p + "ln1.weight"], sd[p + "ln1.bias"])
    q, k, v = (h @ sd[p + "qkv.weight"].T + sd[p + "qkv.bias"]).split(E, dim=2)
    q, k, v = (z.view(1, T, H, E // H).transpose(1, 2) for z in (q, k, v))
    a = (q @ k.transpose(-2, -1)) / math.sqrt(E // H)
    a = a.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf")).softmax(-1)
    att_all.append(a[0, :, -1, :].tolist())
    y = (a @ v).transpose(1, 2).contiguous().view(1, T, E)
    x = x + y @ sd[p + "proj.weight"].T + sd[p + "proj.bias"]
    h2 = F.layer_norm(x, (E,), sd[p + "ln2.weight"], sd[p + "ln2.bias"])
    x = x + F.gelu(h2 @ sd[p + "mlp.0.weight"].T + sd[p + "mlp.0.bias"]) @ sd[p + "mlp.2.weight"].T + sd[p + "mlp.2.bias"]
x = F.layer_norm(x, (E,), sd["ln.weight"], sd["ln.bias"]); logits = x[0, -1] @ sd["head.weight"].T + sd["head.bias"]
pr = logits.softmax(-1); top = torch.topk(pr, 8)
res = {"run": run, "step": step, "context": ctx, "prefix": text[:at], "start": at - T, "next": text[at] if at < len(text) else None,
       "top": [[chars[j], round(v, 4)] for v, j in zip(top.values.tolist(), top.indices.tolist())], "att": att_all}
json.dump(res, open(out, "w"))
print("context ends:", repr(ctx[-40:]), "| next:", repr(res["next"]), "| top:", res["top"][:5])
for l, A in enumerate(att_all):
    for hh, row in enumerate(A):
        best = sorted(range(T), key=lambda j: -row[j])[:4]
        print(f"L{l}H{hh}", " ".join(f"{ctx[j]!r}@{j - T}:{row[j]:.2f}" for j in best))
