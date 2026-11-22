import concurrent.futures as cf
import csv
import glob
import json
import os
import random
import re
import subprocess
import time
import urllib.request

PARTS = os.environ.get("HT_PARTS", "arith,memory").split(",")
LOOPS = [int(x) for x in os.environ.get("HT_LOOPS", "1,2,3,4,5,6").split(",")]
SEED = 20261122
OUT = "/kaggle/working"
TMP = "/kaggle/tmp"
REPO = "ggml-org/TML-d24-varloop-research-GGUF"
FILE = "TML-d24-varloop-research-IQ4_NL.gguf"
BIN = "/kaggle/tmp/llama.cpp/build/bin"
log_lines = []


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    log_lines.append(line)
    with open(f"{OUT}/log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def arith_set():
    rng = random.Random(SEED)
    items = []
    for i in range(100):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        items.append({"id": f"m22_{i}", "type": "m22", "q": f"{a} × {b}", "ans": a * b})
    for i in range(100):
        a, b = rng.randint(100, 999), rng.randint(10, 99)
        items.append({"id": f"m32_{i}", "type": "m32", "q": f"{a} × {b}", "ans": a * b})
    for i in range(100):
        a, b, c, d = (rng.randint(10, 99) for _ in range(4))
        items.append({"id": f"expr_{i}", "type": "expr", "q": f"{a} × {b} + {c} × {d}", "ans": a * b + c * d})
    for it in items:
        it["prompt"] = f"What is {it['q']}? Reply with just the number."
    return items


WORDS = ("the river morning garden window quiet yellow bridge market train letter mountain story careful "
         "bright winter paper evening village ocean simple forest table kitchen music orange silver "
         "student teacher doctor street cloud summer harbor clock station library island valley").split()


def memory_set():
    rng = random.Random(SEED + 1)
    items = []
    for i in range(100):
        letters = rng.sample("ABCDEFGHJKLMNPQRSTUVWXYZ", 8)
        codes = {L: f"{rng.randint(0, 9999):04d}" for L in letters}
        sents = []
        for _ in range(150):
            n = rng.randint(8, 14)
            w = [rng.choice(WORDS) for _ in range(n)]
            sents.append(" ".join(w).capitalize() + ".")
        pos = sorted(rng.sample(range(len(sents)), 8))
        for L, p in zip(letters, pos):
            sents.insert(p, f"The code for locker {L} is {codes[L]}.")
        ask = sorted(letters)
        prompt = (" ".join(sents) + "\n\nList the codes for lockers " + ", ".join(ask) +
                  ". Write one per line as 'locker letter: code'. Nothing else.")
        items.append({"id": f"mem_{i}", "type": "memory", "codes": codes, "prompt": prompt})
    return items


def gpqa_set():
    path = (glob.glob("/kaggle/input/**/gpqa_diamond.csv", recursive=True) or [""])[0]
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    items = []
    for i, r in enumerate(rows):
        rng = random.Random(SEED + 1000 + i)
        opts = [r["Correct Answer"].strip(), r["Incorrect Answer 1"].strip(), r["Incorrect Answer 2"].strip(), r["Incorrect Answer 3"].strip()]
        order = list(range(4))
        rng.shuffle(order)
        letters = "ABCD"
        body = "\n".join(f"{letters[k]}) {opts[j]}" for k, j in enumerate(order))
        ans = letters[order.index(0)]
        prompt = r["Question"].strip() + "\n\n" + body + "\n\nAnswer with the letter only."
        items.append({"id": r.get("Record ID") or f"gpqa_{i}", "type": "gpqa", "ans": ans, "prompt": prompt})
    return items


def score(it, content):
    c = content or ""
    if it["type"] in ("m22", "m32", "expr"):
        nums = re.findall(r"-?\d[\d,]*", c.replace(" ", ""))
        got = int(nums[-1].replace(",", "")) if nums else None
        return {"got": got, "correct": got == it["ans"]}
    if it["type"] == "memory":
        ok = 0
        for L, code in it["codes"].items():
            m = re.search(rf"(?m)^\W*{L}\b[^\n\d]{{0,20}}(\d{{4}})", c)
            ok += bool(m and m.group(1) == code)
        return {"got": ok, "correct": ok / 8}
    m = re.search(r"\b([ABCD])\b", c.strip())
    got = m.group(1) if m else None
    return {"got": got, "correct": got == it["ans"]}


def chat(prompt, effort, max_tokens):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"reasoning_effort": effort}}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return {"content": m.get("content"), "reasoning_chars": len(m.get("reasoning_content") or ""),
            "reasoning_text": m.get("reasoning_content") or "", "finish": d["choices"][0].get("finish_reason"),
            "usage": d.get("usage"), "timings": d.get("timings"), "wall_s": round(time.time() - t, 2)}


PART_CFG = {"arith": ("none", 256, arith_set), "memory": ("none", 256, memory_set), "gpqa": ("low", 3072, gpqa_set)}


def start_server(path, n, par):
    for _ in range(60):
        try:
            urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
            time.sleep(2)
        except Exception:
            break
    args = [f"{BIN}/llama-server", "-m", path, "-ngl", "999", "-c", str(4096 * par), "--parallel", str(par), "-fit", "off",
            "-fa", "on", "-b", "512", "-ub", "256", "-ot", r"blk\.([0-9]|1[01])\.=CUDA0", "-ot", r"blk\.(1[2-9]|2[0-3])\.=CUDA1",
            "--port", "8080", "--jinja", "--override-kv", f"inkling2_loop.loop.count=int:{n}"]
    srv = subprocess.Popen(args, stdout=open(f"{OUT}/server_loops{n}_np{par}.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 600:
        if srv.poll() is not None:
            return None
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5) as r:
                if r.status == 200:
                    return srv
        except Exception:
            pass
        time.sleep(3)
    stop(srv)
    return None


def stop(srv):
    try:
        os.killpg(srv.pid, 15)
        srv.wait(60)
    except Exception:
        try:
            os.killpg(srv.pid, 9)
        except Exception:
            pass


def main():
    t0 = time.time()
    tgz = (glob.glob("/kaggle/input/**/llama-bin-cuda75.tgz", recursive=True) or [""])[0]
    code, o = sh(f"mkdir -p /kaggle/tmp/llama.cpp/build && cd /kaggle/tmp/llama.cpp/build && tar xzf {tgz} && rm -rf bin && mv llama-bin bin")
    os.environ["LD_LIBRARY_PATH"] = BIN + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    log(f"parts {PARTS} loops {LOOPS} | binaries {tgz} exit {code}")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, FILE, local_dir=f"{TMP}/models")
    log(f"model ready in {round(time.time() - t0)} s")
    sets = {p: PART_CFG[p][2]() for p in PARTS}
    os.makedirs(f"{OUT}/testsets", exist_ok=True)
    for p, items in sets.items():
        if p != "gpqa":
            json.dump(items, open(f"{OUT}/testsets/{p}.json", "w"))
        else:
            json.dump([{"id": it["id"], "ans": it["ans"]} for it in items], open(f"{OUT}/testsets/gpqa_keys.json", "w"))
    summary = []
    for n in LOOPS:
        srv, par = None, 0
        for par in (4, 2, 1):
            srv = start_server(path, n, par)
            if srv:
                break
        if not srv:
            log(f"loops={n}: server failed")
            summary.append({"loops": n, "error": "server failed"})
            continue
        mem = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader")[1].strip().replace("\n", " | ")
        log(f"loops={n} up with {par} slots, gpu {mem}")
        for p in PARTS:
            effort, mt, _ = PART_CFG[p]
            items = sets[p]
            t1 = time.time()
            out = open(f"{OUT}/results_{p}_loops{n}.jsonl", "w")

            def run(it):
                try:
                    a = chat(it["prompt"], effort, mt)
                except Exception as e:
                    a = {"error": repr(e), "content": None}
                s = score(it, a.get("content"))
                rec = {"loops": n, "part": p, "id": it["id"], "type": it["type"], "effort": effort, **s,
                       **{k: a.get(k) for k in ("content", "reasoning_chars", "finish", "usage", "timings", "wall_s", "error")}}
                if p != "gpqa":
                    rec["reasoning_text"] = a.get("reasoning_text")
                return rec

            recs = []
            with cf.ThreadPoolExecutor(par) as ex:
                for rec in ex.map(run, items):
                    recs.append(rec)
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.close()
            acc = sum(float(r["correct"]) for r in recs) / len(recs)
            errs = sum(1 for r in recs if r.get("error"))
            trunc = sum(1 for r in recs if r.get("finish") == "length")
            toks = [((r.get("timings") or {}).get("predicted_n") or 0) for r in recs]
            line = {"loops": n, "part": p, "n": len(recs), "acc": round(acc, 4), "errors": errs, "truncated": trunc,
                    "mean_gen_tokens": round(sum(toks) / len(toks), 1), "minutes": round((time.time() - t1) / 60, 1)}
            summary.append(line)
            log(json.dumps(line))
            json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
        stop(srv)
    log(f"done in {round((time.time() - t0) / 60)} min")


main()
