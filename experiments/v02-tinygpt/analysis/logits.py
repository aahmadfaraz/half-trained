"""Raw scores (logits) for the first letter after the chat prompt, and a temperature-0 reply (always the top pick).
python analysis/logits.py RUN STEP OUT_JSON"""
import json, os, sys, importlib.util, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], os.devnull, os.devnull, os.devnull] + sys.argv[3:]
src = open(os.path.join(HERE, "fullodds.py")).read().split("m = GPT()")[0]
g = {"__file__": os.path.join(HERE, "fullodds.py")}; exec(src, g)
GPT, chars, stoi, B, EXP = g["GPT"], g["chars"], g["stoi"], g["B"], g["EXP"]
run, step, out = sys.argv[1], int(sys.argv[2]), sys.argv[6]
m = GPT(); m.load_state_dict({k: v.float() for k, v in torch.load(os.path.join(EXP, "runs", run, "weights", f"w_{step:05d}.pt"), map_location="cpu").items()}); m.eval()
prompt = "YOU:\nWhat is the meaning of life?\n\nHAMLET:\n"
ids = [stoi[c] for c in prompt]
with torch.no_grad():
    lg = m(torch.tensor([ids[-B:]]))[0, -1]
    order = torch.argsort(lg, descending=True).tolist()
    rows = [[chars[j], round(lg[j].item(), 4), round(F.softmax(lg, -1)[j].item(), 6), round(F.softmax(lg / 0.8, -1)[j].item(), 6), round(F.softmax(lg / 0.3, -1)[j].item(), 6), round(F.softmax(lg / 1.5, -1)[j].item(), 6)] for j in order]
    g0 = list(ids)
    for _ in range(600):
        g0.append(int(m(torch.tensor([g0[-B:]]))[0, -1].argmax()))
greedy = "".join(chars[i] for i in g0[len(ids):])
json.dump({"run": run, "step": step, "prompt": prompt, "cols": ["char", "logit", "p_t1", "p_t0.8", "p_t0.3", "p_t1.5"], "first": rows, "greedy": greedy}, open(out, "w"), indent=1)
for r in rows[:8] + rows[-4:]: print(r)
print("---- temperature 0 ----"); print(greedy)
