"""Analysis as registered in PREREG.md: accuracy per loop count with 95% Wilson intervals, by type; exact McNemar
between loop counts on the same items; visible reasoning; speed.
python3 analyze.py OUT_FILE.md [--bd DIR] [--silent DIR ...] [--gpqa DIR ...]"""
import argparse
import glob
import json
import math
import os
import re
import statistics as st
from collections import defaultdict


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(a, b):
    """exact two-sided p for paired booleans a, b"""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p), n01, n10


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% ({k}/{n}; {100 * lo:.1f}-{100 * hi:.1f})"


def last_int(text):
    nums = re.findall(r"-?\d[\d,]*", (text or "").replace(" ", ""))
    return int(nums[-1].replace(",", "")) if nums else None


def table_acc(lines, recs_by_loop, types, label):
    loops = sorted(recs_by_loop)
    lines.append(f"\n### {label}: accuracy by loop count (95% Wilson interval)\n")
    lines.append("| Loops | " + " | ".join(["all"] + types) + " |")
    lines.append("|---" * (len(types) + 2) + "|")
    for n in loops:
        rs = recs_by_loop[n]
        row = [fmt(sum(r["ok"] for r in rs), len(rs))]
        for t in types:
            sub = [r for r in rs if r["type"] == t]
            row.append(fmt(sum(r["ok"] for r in sub), len(sub)) if sub else "-")
        lines.append(f"| {n} | " + " | ".join(row) + " |")
    lines.append("\nPaired exact McNemar (same items): p, items right only at the first count, only at the second\n")
    pairs = [(a, b) for i, a in enumerate(loops) for b in loops[i + 1:]]
    for a, b in pairs:
        ia = {r["id"]: r["ok"] for r in recs_by_loop[a]}
        ib = {r["id"]: r["ok"] for r in recs_by_loop[b]}
        ids = sorted(set(ia) & set(ib))
        p, n01, n10 = mcnemar([ia[i] for i in ids], [ib[i] for i in ids])
        verdict = "real" if p < 0.05 else "no clear difference"
        lines.append(f"- {a} vs {b}: p = {p:.4f} ({n01} vs {n10}) → {verdict}")


def bd(d, lines, out):
    by = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(d, "results_*_loops*.jsonl")):
        part, n = re.search(r"results_(\w+)_loops(\d+)", f).groups()
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            ok = r["correct"]
            by[part][int(n)].append({"id": r["id"], "type": r["type"], "ok": ok if isinstance(ok, bool) else ok,
                                     "rchars": r.get("reasoning_chars") or 0, "finish": r.get("finish"), "error": r.get("error"),
                                     "pred_n": (r.get("timings") or {}).get("predicted_n") or 0,
                                     "tps": (r.get("timings") or {}).get("predicted_per_second") or 0,
                                     "content": r.get("content")})
    if "arith" in by:
        table_acc(lines, by["arith"], ["m22", "m32", "expr"], "Arithmetic, effort none (visible thinking allowed)")
        lines.append("\n| Loops | mean / median reasoning chars | mean generated tokens | truncated | errors | mean tok/s |\n|---|---|---|---|---|---|")
        for n in sorted(by["arith"]):
            rs = by["arith"][n]
            rc = [r["rchars"] for r in rs]
            lines.append(f"| {n} | {st.mean(rc):.0f} / {st.median(rc):.0f} | {st.mean(r['pred_n'] for r in rs):.1f} | "
                         f"{sum(r['finish'] == 'length' for r in rs)} | {sum(bool(r['error']) for r in rs)} | {st.mean(r['tps'] for r in rs):.1f} |")
    if "memory" in by:
        lines.append("\n### Memory (8 lockers per item, effort none): codes recalled\n\n| Loops | mean recall | items with 8/8 | truncated | mean reasoning chars |\n|---|---|---|---|---|")
        for n in sorted(by["memory"]):
            rs = by["memory"][n]
            rec = [float(r["ok"]) for r in rs]
            full = sum(1 for x in rec if x == 1.0)
            lines.append(f"| {n} | {100 * st.mean(rec):.1f}% (sd {100 * st.pstdev(rec):.1f}) | {fmt(full, len(rs))} | "
                         f"{sum(r['finish'] == 'length' for r in rs)} | {st.mean(r['rchars'] for r in rs):.0f} |")
    out["bd"] = {p: {n: [dict(r, content=None) for r in v] for n, v in by[p].items()} for p in by}


def silent(dirs, lines, out):
    recs = defaultdict(list)
    for d in dirs:
        pj = os.path.join(d, "prompts.json")
        if not os.path.exists(pj):
            continue
        ans = {it["id"]: it["ans"] for it in json.load(open(pj, encoding="utf-8"))["items"]}
        for f in glob.glob(os.path.join(d, "silent_loops*.json")):
            n = int(re.search(r"loops(\d+)", f).group(1))
            for r in json.load(open(f, encoding="utf-8")):
                text = r["generated"].split("<|end_message|>")[0]
                g = last_int(text)
                recs[n].append({"id": r["id"], "type": r["id"].split("_")[0], "ok": g == ans[r["id"]], "got": g, "text": text,
                                "kind": kind(text, "<|end_message|>" in r["generated"]),
                                "p_top": r["steps"][0][0][2] if r["steps"] else None})
    if recs:
        table_acc(lines, recs, ["m22", "m32", "expr"], "Arithmetic, FORCED SILENT (no thinking message possible)")
        silent_kinds(lines, recs)
        out["silent"] = recs


def kind(text, done):
    if re.fullmatch(r"\s*-?[\d,]+\.?\s*", text) and done:
        return "bare"
    if re.match(r"\s*(We|The|Let|I|To|Okay|First|Need|So)\b", text):
        return "words"
    if re.search(r"[×x*=+]", text):
        return "working"
    return "other"


def silent_kinds(lines, recs):
    lines.append("\n### Exploratory (not pre-registered): what the forced-silent replies look like\n")
    lines.append("bare = only a number, then the message ends; words = starts writing reasoning in words inside the answer; "
                 "working = writes the calculation (e.g. '72 × 64 = …'). With a 10-token limit, 'words' replies are cut off before any answer.\n")
    lines.append("| Loops | bare (right) | words (right) | working (right) | other |\n|---|---|---|---|---|")
    for n in sorted(recs):
        c = defaultdict(lambda: [0, 0])
        for r in recs[n]:
            c[r["kind"]][0] += 1
            c[r["kind"]][1] += r["ok"]
        lines.append(f"| {n} | " + " | ".join(f"{c[k][0]} ({c[k][1]})" for k in ("bare", "words", "working")) + f" | {c['other'][0]} |")
    loops = sorted(recs)
    lines.append("\nPaired: items answered bare at both loop counts, accuracy on exactly those items (exact McNemar)\n")
    for i, a in enumerate(loops):
        for b in loops[i + 1:]:
            ia = {r["id"]: r for r in recs[a] if r["kind"] == "bare"}
            ib = {r["id"]: r for r in recs[b] if r["kind"] == "bare"}
            ids = sorted(set(ia) & set(ib))
            if not ids:
                continue
            p, n01, n10 = mcnemar([ia[x]["ok"] for x in ids], [ib[x]["ok"] for x in ids])
            ka, kb = sum(ia[x]["ok"] for x in ids), sum(ib[x]["ok"] for x in ids)
            lines.append(f"- {a} vs {b}: {len(ids)} items, right {ka} vs {kb}, p = {p:.4f}")


CARD_GPQA = {1: 69.6, 2: 79.2, 3: 84.2, 4: 81.9, 5: 83.1}


def gpqa(dirs, lines, out):
    recs = defaultdict(list)
    for d in dirs:
        for f in glob.glob(os.path.join(d, "results_gpqa_loops*.jsonl")):
            n = int(re.search(r"loops(\d+)", f).group(1))
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                t = r.get("timings") or {}
                recs[n].append({"id": r["id"], "type": "gpqa", "ok": bool(r["correct"]), "got": r.get("got"),
                                "rchars": r.get("reasoning_chars") or 0, "finish": r.get("finish"), "error": r.get("error"),
                                "pred_n": t.get("predicted_n") or 0, "tps": t.get("predicted_per_second") or 0,
                                "letter_only": bool(re.fullmatch(r"\W*[ABCD]\W*", (r.get("content") or "").strip()))})
    if not recs:
        return
    full = max(len(v) for v in recs.values())
    part = {n: v for n, v in recs.items() if len(v) < full}
    recs = defaultdict(list, {n: v for n, v in recs.items() if len(v) == full})
    if part:
        lines.append("\n### GPQA Diamond: incomplete loop counts (run stopped at Kaggle's 12-hour limit; first questions in run order, not a random sample; not compared)\n")
        for n, v in sorted(part.items()):
            lines.append(f"- {n} loops: {len(v)} of {full} questions answered, {sum(r['ok'] for r in v)} right")
    table_acc(lines, recs, [], f"GPQA Diamond, effort low ({full} questions per loop count)")
    lines.append("\n| Loops | ours | card | truncated | errors | no letter found | reply is just a letter | "
                 "reasoning chars mean / median | generated tokens mean / median | mean tok/s |\n|---|---|---|---|---|---|---|---|---|---|")
    for n in sorted(recs):
        rs = recs[n]
        rc, pn = [r["rchars"] for r in rs], [r["pred_n"] for r in rs]
        card = f"{CARD_GPQA[n]}%" if n in CARD_GPQA else "-"
        lines.append(f"| {n} | {100 * sum(r['ok'] for r in rs) / len(rs):.1f}% | {card} | {sum(r['finish'] == 'length' for r in rs)} | "
                     f"{sum(bool(r['error']) for r in rs)} | {sum(r['got'] is None for r in rs)} | {sum(r['letter_only'] for r in rs)}/{len(rs)} | "
                     f"{st.mean(rc):.0f} / {st.median(rc):.0f} | {st.mean(pn):.0f} / {st.median(pn):.0f} | {st.mean(r['tps'] for r in rs):.1f} |")
    lines.append("\nThe card's settings are not stated and its file is the native checkpoint (ours: IQ4_NL GGUF), so only the shape is compared (PREREG Q3).")
    out["gpqa"] = recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--bd")
    ap.add_argument("--silent", nargs="*", default=[])
    ap.add_argument("--gpqa", nargs="*", default=[])
    a = ap.parse_args()
    lines, out = ["# Inkling loops: results (generated by analyze.py)"], {}
    if a.bd:
        bd(a.bd, lines, out)
    if a.silent:
        silent(a.silent, lines, out)
    if a.gpqa:
        gpqa(a.gpqa, lines, out)
    open(a.out, "w").write("\n".join(lines) + "\n")
    json.dump(out, open(a.out[:-3] + ".json", "w"), default=str)
    print("\n".join(lines))


main()
