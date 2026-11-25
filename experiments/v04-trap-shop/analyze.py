"""Summarise trip receipts: python analyze.py runs/pilot2"""
import glob, json, sys
from collections import defaultdict
from shop.engine import TRAPS

R = [json.load(open(f)) for f in sorted(glob.glob(sys.argv[1].rstrip("/") + "/*.receipt.json"))]
by = defaultdict(list)
for r in R: by[r["model"]].append(r)
for m, rs in by.items():
    print(f"\n== {m}: {len(rs)} trips")
    for r in rs:
        print(f"  s{r['seed']:<3}{r['cond']:<8}{r['status']:<14}steps {r['steps']:>2}  invalid {r.get('invalid_steps', 0)}  reprompts {r.get('reprompts', 0)}  "
              f"spent {r['spent']:>7.2f} best {r['cheapest_correct']:>7.2f} wasted {str(r['wasted']):>6}  missing {r['missing']} wrong {[w['id'] for w in r['wrong_or_extra']]}  "
              f"data {r['data_shared_beyond_needed']}  tok {r.get('tokens_in')}/{r.get('tokens_out')}  {r.get('wall_seconds')} s  rl {len(r['rate_limit_events'])}")
    print("  trap               taken/active   control: decoy bought")
    for t in TRAPS:
        act = [r for r in rs if r["traps"][t]["active"]]
        ctl = [r for r in rs if r["cond"] == "control" and r["traps"][t]["decoy_bought"] is not None]
        took = sum(1 for r in act if r["traps"][t]["taken"])
        cb = sum(1 for r in ctl if r["traps"][t]["decoy_bought"] and r["seed"] in {a["seed"] for a in act})
        cn = sum(1 for r in ctl if r["seed"] in {a["seed"] for a in act})
        print(f"  {t:<19}{took}/{len(act):<12}{cb}/{cn}" if act else f"  {t:<19}-")
    for c in ("trap", "control"):
        w = [r["wasted"] for r in rs if r["cond"] == c and r["wasted"] is not None]
        n = sum(1 for r in rs if r["cond"] == c)
        print(f"  {c}: complete baskets {len(w)}/{n}, wasted total {sum(w):.2f}" + (f", mean {sum(w) / len(w):.2f}" if w else ""))
