"""The numbers PREREG.md promises, for every model and match.
python llm_report.py -> runs/llm_report.json and runs/llm_report.md"""
import json, glob, os, ast, statistics as st

DEADLINE = "2026-11-17 23:59:59"
REG, EXT = range(90000, 90005), range(90005, 90020)
MATCHES = [("wall_runner", "Wall runner: the model runs, our wall chaser chases"),
           ("open_chaser", "Open chaser: the model chases our open runner"),
           ("open_runner", "Open runner: the model runs, our open chaser chases")]


def model_of(cfg, match):
    return cfg["runner"] if match.endswith("runner") else cfg["chaser"]


def usage(d):
    u = d.get("usage") or {}
    if isinstance(u, str):
        try: u = ast.literal_eval(u)
        except (ValueError, SyntaxError): u = {}
    return u.get("in", 0) or 0, u.get("out", 0) or 0


def summarize(games, wall):
    if not games: return None
    dec = [d for g in games for d in g.get("decisions", [])]
    secs = [float(d["seconds"]) for d in dec if d.get("seconds") not in (None, "")]
    toks = [usage(d) for d in dec]
    out = {"n": len(games), "caught": sum(1 for g in games if g["caught"]),
           "mean_s": round(st.mean(g["seconds"] for g in games), 2),
           "decisions": len(dec), "invalid": sum(1 for d in dec if str(d.get("valid")) != "True"),
           "s_per_decision": round(st.mean(secs), 2) if secs else None,
           "tokens_in_per_decision": round(st.mean(t[0] for t in toks)) if toks else None,
           "tokens_out_per_decision": round(st.mean(t[1] for t in toks)) if toks else None}
    caught_s = [g["seconds"] for g in games if g["caught"]]
    out["mean_s_when_caught"] = round(st.mean(caught_s), 2) if caught_s else None
    if wall:
        out["loops"] = round(st.mean(g.get("loops", 0) for g in games), 2)
        out["blocked"] = round(st.mean(g.get("blocked", 0) for g in games), 3)
        out["near_wall"] = round(st.mean(g.get("near_wall", 0) for g in games), 3)
    return out


rep = {}
for match, _ in MATCHES:
    for f in sorted(glob.glob(os.path.join("runs", "llm", match, "*.json"))):
        d = json.load(open(f)); name = os.path.basename(f)[:-5]
        ok = [g for g in d["games"] if g.get("finished", "") <= DEADLINE]
        late = [g["seed"] for g in d["games"] if g.get("finished", "") > DEADLINE]
        r = {"model": model_of(d, match), "registered": summarize([g for g in ok if g["seed"] in REG], match == "wall_runner"),
             "extended": summarize([g for g in ok if g["seed"] in EXT], match == "wall_runner"),
             "late_not_played": late, "in_progress": os.path.exists(f + ".partial")}
        rep.setdefault(match, {})[name] = r

complete = {n for n in rep.get("wall_runner", {}) if all((rep[m].get(n) or {}).get("registered") and rep[m][n]["registered"]["n"] == 5 for m, _ in MATCHES)}
for m in rep:
    for n in rep[m]: rep[m][n]["all_three_registered_sets_done"] = n in complete
json.dump(rep, open(os.path.join("runs", "llm_report.json"), "w"), indent=1)


def line(name, r, key):
    s = r[key]
    if not s: return None
    wall = f" | {s['loops']} | {s['blocked']:.0%}" if "loops" in s else ""
    spd = "–" if s["s_per_decision"] is None else f"{s['s_per_decision']} s"
    tok = "–" if not s["decisions"] else f"{s['tokens_in_per_decision']} / {s['tokens_out_per_decision']}"
    return f"| {name} | {s['caught']} of {s['n']} | {s['mean_s']} s | {s['invalid']} of {s['decisions']} | {spd} | {tok}{wall} |"


md = ["# Tag: models against our agents (from runs/llm, deadline " + DEADLINE[:16] + " UAE)", ""]
for match, title in MATCHES:
    wall = match == "wall_runner"
    head = "| Player | Caught | Mean game | Invalid replies | Time per decision | Tokens in / out per decision |" + (" Loops | Wall blocks the view |" if wall else "")
    sep = "|---|---|---|---|---|---|" + ("---|---|" if wall else "")
    for key, label in (("registered", "Registered set (seeds 90000-90004)"), ("extended", "Extended set (seeds 90005-90019)")):
        rows = [line(n, r, key) for n, r in sorted(rep.get(match, {}).items())]
        rows = [x for x in rows if x]
        if rows: md += [f"## {title}: {label}", "", head, sep] + rows + [""]
    notes = [f"- {n}: a game was still in progress at the deadline (not played)." for n, r in sorted(rep.get(match, {}).items()) if r["in_progress"]]
    notes += [f"- {n}: finished after the deadline, not played: seeds {r['late_not_played']}." for n, r in sorted(rep.get(match, {}).items()) if r["late_not_played"]]
    if notes: md += notes + [""]
md += ["Players named control-* are the controls, played through the same harness (20 games each here; the 500-game controls are in runs/controls).",
       "Models with all three registered sets done: " + (", ".join(sorted(complete)) or "none") + "."]
open(os.path.join("runs", "llm_report.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
