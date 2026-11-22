import glob
import json
import os
import random
import subprocess
import time
import urllib.request

import numpy as np

SEED = 20261122
OUT, TMP = "/kaggle/working", "/kaggle/tmp"
COMMIT = "afe77e18fadc2e673c6a6299495a443e17378ce4"
REPO, FILE = "ggml-org/TML-d24-varloop-research-GGUF", "TML-d24-varloop-research-IQ4_NL.gguf"
SRC = f"{TMP}/llama.cpp"
LOOPS = [1, 2, 3, 4, 5, 6]
N_GEN = 12
PROBE_CPP = r"""@@PROBE_CPP@@"""
log_lines = []


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    log_lines.append(line)
    open(f"{OUT}/log.txt", "w").write("\n".join(log_lines) + "\n")


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def arith_items():
    rng = random.Random(SEED)
    items = []
    for i in range(100):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        items.append({"id": f"m22_{i}", "q": f"{a} × {b}", "ans": a * b})
    for i in range(100):
        a, b = rng.randint(100, 999), rng.randint(10, 99)
        items.append({"id": f"m32_{i}", "q": f"{a} × {b}", "ans": a * b})
    for i in range(100):
        a, b, c, d = (rng.randint(10, 99) for _ in range(4))
        items.append({"id": f"expr_{i}", "q": f"{a} × {b} + {c} × {d}", "ans": a * b + c * d})
    keep = [it for it in items if int(it["id"].split("_")[1]) < 10]
    for it in keep:
        it["prompt"] = f"What is {it['q']}? Reply with just the number."
    return keep


def build():
    t = time.time()
    code, o = sh(f"git init -q {SRC} && cd {SRC} && git remote add origin https://github.com/ggml-org/llama.cpp && "
                 f"git fetch -q --depth 1 origin {COMMIT} && git checkout -q FETCH_HEAD && git rev-parse HEAD")
    log(f"source: exit {code} {o.strip()[-60:]}")
    os.makedirs(f"{SRC}/tools/ht-probe", exist_ok=True)
    open(f"{SRC}/tools/ht-probe/probe.cpp", "w").write(PROBE_CPP)
    open(f"{SRC}/tools/ht-probe/CMakeLists.txt", "w").write(
        "set(TARGET ht-probe)\nadd_executable(${TARGET} probe.cpp)\n"
        "target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})\n"
        "target_compile_features(${TARGET} PRIVATE cxx_std_17)\n"
        "target_compile_options(${TARGET} PRIVATE -Wno-deprecated-declarations)\n")
    with open(f"{SRC}/tools/CMakeLists.txt", "a") as f:
        f.write("\nadd_subdirectory(ht-probe)\n")
    code, o = sh(f"cd {SRC} && cmake -B build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=OFF "
                 f"&& cmake --build build -j{os.cpu_count()} --target llama-server ht-probe")
    open(f"{OUT}/build.log", "w").write(o[-20000:])
    log(f"build: exit {code} in {round((time.time() - t) / 60, 1)} min")
    return code == 0


def templates(model, items):
    srv = subprocess.Popen([f"{SRC}/build/bin/llama-server", "-m", model, "-ngl", "0", "-c", "4096", "--jinja", "--port", "8080",
                            "--override-kv", "inkling2_loop.loop.count=int:3"],
                           stdout=open(f"{OUT}/server.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(400):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(3)
    out = {}
    for it in items:
        body = {"messages": [{"role": "user", "content": it["prompt"]}], "chat_template_kwargs": {"reasoning_effort": "none"}}
        req = urllib.request.Request("http://127.0.0.1:8080/apply-template", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        out[it["id"]] = json.load(urllib.request.urlopen(req, timeout=120))["prompt"]
    os.killpg(srv.pid, 15)
    srv.wait(60)
    return out


def main():
    t0 = time.time()
    log(f"cpus {os.cpu_count()} | " + sh("free -g | head -2 | tail -1")[1].strip())
    if not build():
        return
    from huggingface_hub import hf_hub_download
    model = hf_hub_download(REPO, FILE, local_dir=f"{TMP}/models")
    log(f"model ready in {round(time.time() - t0)} s")
    items = arith_items()
    tpl = templates(model, items)
    pdir = f"{TMP}/prompts"
    os.makedirs(pdir, exist_ok=True)
    for it in items:
        open(f"{pdir}/{it['id']}.txt", "w").write(tpl[it["id"]])
    open(f"{pdir}/cat.txt", "w").write("The cat sat on the")
    json.dump({"items": items, "templated": tpl, "cat": "The cat sat on the"}, open(f"{OUT}/prompts.json", "w"), ensure_ascii=False, indent=1)
    log("prompt example: " + json.dumps(tpl[items[0]["id"]]))
    for n in LOOPS:
        t1 = time.time()
        raw = f"{TMP}/raw{n}"
        code, o = sh(f"{SRC}/build/bin/ht-probe {model} {n} {pdir} {raw} {N_GEN} {os.cpu_count()}")
        open(f"{OUT}/probe_loops{n}.log", "w").write(o[-30000:])
        res, arrays = [], {}
        for js in sorted(glob.glob(f"{raw}/*.json")):
            d = json.load(open(js))
            b = open(js[:-5] + ".bin", "rb").read()
            for tsr in d.pop("tensors"):
                dt = np.float32 if tsr["type"] == "f32" else np.int32
                a = np.frombuffer(b, dtype=dt, count=tsr["n"], offset=tsr["offset"])
                arrays[f"{d['id']}|{tsr['name']}"] = a.astype(np.float16) if dt == np.float32 else a
            res.append(d)
        np.savez_compressed(f"{OUT}/probe_loops{n}.npz", **arrays)
        json.dump(res, open(f"{OUT}/probe_loops{n}.json", "w"), ensure_ascii=False)
        log(f"loops={n}: exit {code}, {len(res)} prompts, {len(arrays)} tensors, {round((time.time() - t1) / 60, 1)} min")
    log(f"done in {round((time.time() - t0) / 60)} min")


main()
