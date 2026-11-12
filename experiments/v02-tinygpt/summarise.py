"""Summarise runs/eval/*.json into runs/eval/summary.md.

python summarise.py
"""
import glob, json, math, os, random, re

HERE = os.path.dirname(os.path.abspath(__file__))
HEAD = re.compile(r"^[A-Za-z][A-Za-z' ]{0,30}:$")
def turns(s):
    ls = s.split("\n"); return [l[:-1] for i, l in enumerate(ls) if HEAD.match(l) and (i == 0 or ls[i - 1].strip() == "")]

T = turns(open(os.path.join(HERE, "data/tinyshakespeare.txt")).read())
TALK = {frozenset(p) for p in zip(T, T[1:]) if p[0] != p[1]}
random.seed(0)
R = [frozenset(random.sample(T, 2)) for _ in range(20000)]
R = [p for p in R if len(p) == 2]
chance = sum(p in TALK for p in R) / len(R)

COLS = [("valid_speaker_share", "real speaker", "%"), ("same_speaker_twice_share", "same speaker twice", "%"),
        ("new_speaker_share_after_first", "new speaker", "%"), ("handover_pairs_that_talk_share", "handover to someone they talk to", "%"),
        ("line_len_mean", "mean line", ""), ("lines_over_80_share", "lines over 80", "%"),
        ("real_word_share", "real words", "%"), ("copied_30char_share", "copied 30+ chars", "%")]
fmt = lambda v, u: f"{v:.1%}" if u == "%" else f"{v:.3f}" if u == "f3" else f"{v}"
BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256)]

real = json.load(open(os.path.join(HERE, "runs/eval/real.json")))["structure"]
evals = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "runs/eval/*_*.json")))]
evals = [d for d in evals if "handovers" in d.get("structure", {})]

out = ["# V2 evaluation summary", "", "## Every evaluation", "",
       "| run | step | memory | full-memory val loss | e^loss | " + " | ".join(c[1] for c in COLS) + " |",
       "|---|---|---|---|---|" + "---|" * len(COLS),
       "| real held-out text | | | | | " + " | ".join(fmt(real[k], u) for k, _, u in COLS) + " |"]
for d in evals:
    s = d["structure"]
    out.append(f"| {d['run']} | {d['step']} | {d['block']} | {d['full_memory_val_loss']:.3f} | {math.exp(d['full_memory_val_loss']):.1f} | "
               + " | ".join(fmt(s[k], u) for k, _, u in COLS) + " |")
out += ["", f"Handovers are speaker changes in 48 samples of 700 letters (temperature 0.8). A random pair of speakers talks to each other in the book {chance:.1%} of the time.", ""]

groups = {}
for d in evals:
    if d["run"] == "main_s2": continue
    if d["run"] != "mem1_s2" and d["step"] != 2000: continue
    groups.setdefault(re.sub(r"_s\d+b?$", "", d["run"]), []).append(d)
out += ["## Seeds at step 2,000 (mem1 seed 2 at 5,000; mem1 is flat after step 300)", "",
        "| config | seeds | full-memory val loss | handover to someone they talk to | new speaker | same speaker twice | lines over 80 | real words |",
        "|---|---|---|---|---|---|---|---|"]
def span(xs, u):
    m = sum(xs) / len(xs)
    return fmt(m, u) if len(xs) == 1 else f"{fmt(m, u)} ({fmt(min(xs), u)}-{fmt(max(xs), u)})"
for g in ("main", "mem32", "mem8", "mem1"):
    ds = groups.get(g, [])
    if not ds: continue
    S = [d["structure"] for d in ds]
    out.append(f"| {g} | {len(ds)} | {span([d['full_memory_val_loss'] for d in ds], 'f3')} | "
               + " | ".join(span([s[k] for s in S], "%") for k in ("handover_pairs_that_talk_share", "new_speaker_share_after_first", "same_speaker_twice_share", "lines_over_80_share", "real_word_share")) + " |")

out += ["", "## Loss by how many letters the model can see (mean over 1,000 val windows)", "",
        "| run | step | " + " | ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in BINS) + " |",
        "|---|---|" + "---|" * len(BINS)]
for d in evals:
    c = d["loss_by_context"]
    out.append(f"| {d['run']} | {d['step']} | " + " | ".join(f"{sum(c[a - 1:b]) / len(c[a - 1:b]):.3f}" if a <= len(c) else "" for a, b in BINS) + " |")
open(os.path.join(HERE, "runs/eval/summary.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
