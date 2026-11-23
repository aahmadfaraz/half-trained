"""Normal-model baseline (PREREG change of 23 Nov): Qwen3.5 4B via local Ollama on the same 300 arithmetic prompts.
A: no thinking, answer straight away, <=10 tokens. B: thinking allowed, <=4096 tokens. Greedy. Same scoring as the forced-silent test.
python3 baseline/qwen_baseline.py PROMPTS_JSON OUT_DIR [A|B]"""
import json, os, re, sys, time, urllib.request
URL, MODEL = "http://127.0.0.1:11434/api/chat", "qwen3.5:4b"
prompts, out, conds = sys.argv[1], sys.argv[2], (sys.argv[3:] or ["A", "B"])
items = json.load(open(prompts, encoding="utf-8"))["items"]
os.makedirs(out, exist_ok=True)
def last_int(t):
    n = re.findall(r"-?\d[\d,]*", (t or "").replace(" ", ""))
    return int(n[-1].replace(",", "")) if n else None
for cond in conds:
    think, num = (False, 10) if cond == "A" else (True, 4096)
    path = os.path.join(out, f"qwen_{cond}.jsonl"); done = set()
    if os.path.exists(path): done = {json.loads(l)["id"] for l in open(path)}
    with open(path, "a") as f:
        for it in items:
            if it["id"] in done: continue
            body = {"model": MODEL, "messages": [{"role": "user", "content": it["prompt"]}], "think": think, "stream": False,
                    "options": {"temperature": 0, "seed": 20261122, "num_predict": num}}
            t = time.time()
            d = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1200))
            m = d["message"]; g = last_int(m.get("content"))
            f.write(json.dumps({"id": it["id"], "type": it["id"].split("_")[0], "cond": cond, "got": g, "ok": g == it["ans"], "content": m.get("content"),
                                "thinking_chars": len(m.get("thinking") or ""), "eval_count": d.get("eval_count"), "done_reason": d.get("done_reason"),
                                "eval_s": round((d.get("eval_duration") or 0) / 1e9, 3), "wall_s": round(time.time() - t, 2)}, ensure_ascii=False) + "\n"); f.flush()
    rs = [json.loads(l) for l in open(path)]
    print(cond, f"{sum(r['ok'] for r in rs)}/{len(rs)} right", flush=True)
