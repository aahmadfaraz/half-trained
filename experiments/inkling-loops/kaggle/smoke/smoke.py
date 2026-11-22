import json, os, subprocess, time, urllib.request

OUT = "/kaggle/working"
TMP = "/kaggle/tmp"
REPO = "ggml-org/TML-d24-varloop-research-GGUF"
FILE = "TML-d24-varloop-research-IQ4_NL.gguf"
PROMPTS = [
    ("mult", "What is 47 times 38? Reply with just the number.", "1786"),
    ("sisters", "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have? Reply with just the number.", "1"),
    ("capital", "What is the capital of Australia? Reply with one word.", "Canberra"),
]
LOOPS = [1, 2, 3, 4, 5, 6, 7, 8]
PORT_CHECK = True
log_lines = []


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    log_lines.append(line)
    with open(f"{OUT}/smoke_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")


def sh(cmd, **kw):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    return r.returncode, (r.stdout + r.stderr)


def gpu_mem():
    _, o = sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader")
    return o.strip().replace("\n", " | ")


os.makedirs(TMP, exist_ok=True)
res = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}
res["gpu"] = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")[1].strip()
res["nvcc"] = sh("nvcc --version | tail -2")[1].strip()
res["disk"] = sh("df -h / /kaggle/working | tail -2")[1].strip()
log(f"gpu: {res['gpu']} | disk: {res['disk']}")

t0 = time.time()
code, o = sh("git clone --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/tmp/llama.cpp")
res["llama_commit"] = sh("git -C /kaggle/tmp/llama.cpp log -1 --format='%H %cI'")[1].strip()
log(f"clone {code}: {res['llama_commit']}")
_, libs = sh("ls /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/local/nvidia/lib64/libcuda.so.1 2>/dev/null")
libcuda = (libs.split() or [""])[0]
res["libcuda"] = libcuda
log(f"libcuda: {libcuda}")
code, o = sh(f"cd /kaggle/tmp/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=OFF -DCUDA_cuda_driver_LIBRARY={libcuda} && cmake --build build -j$(nproc) --target llama-server llama-cli")
res["build_s"] = round(time.time() - t0)
sh("mkdir -p /kaggle/working/llama-bin && cp -a /kaggle/tmp/llama.cpp/build/bin/. /kaggle/working/llama-bin/ && cd /kaggle/working && tar czf llama-bin-cuda75.tgz llama-bin && rm -rf llama-bin")
log(f"build exit {code} in {res['build_s']} s")
if code != 0:
    open(f"{OUT}/build_error.txt", "w").write(o[-20000:])
    raise SystemExit("build failed")

t0 = time.time()
from huggingface_hub import hf_hub_download
path = hf_hub_download(REPO, FILE, local_dir=f"{TMP}/models")
res["download_s"] = round(time.time() - t0)
res["model_bytes"] = os.path.getsize(path)
log(f"model {res['model_bytes']} bytes in {res['download_s']} s")


def chat(prompt, effort, max_tokens):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"reasoning_effort": effort}}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return {"content": m.get("content"), "reasoning": m.get("reasoning_content"), "finish": d["choices"][0].get("finish_reason"),
            "usage": d.get("usage"), "timings": d.get("timings"), "wall_s": round(time.time() - t, 2)}


for n in LOOPS:
    run = {"loops": n}
    for _ in range(60):
        try:
            urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
            time.sleep(2)
        except Exception:
            break
    srv = subprocess.Popen(["/kaggle/tmp/llama.cpp/build/bin/llama-server", "-m", path, "-ngl", "999", "-c", "8192", "--parallel", "1",
                            "--port", "8080", "--jinja", "--override-kv", f"inkling2_loop.loop.count=int:{n}"],
                           stdout=open(f"{OUT}/server_loops{n}.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    ok = False
    t0 = time.time()
    while time.time() - t0 < 600:
        if srv.poll() is not None:
            break
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5) as r:
                if r.status == 200:
                    ok = True
                    break
        except Exception:
            pass
        time.sleep(3)
    run["load_s"] = round(time.time() - t0)
    run["server_ok"] = ok
    run["gpu_mem"] = gpu_mem()
    log(f"loops={n} server_ok={ok} load {run['load_s']} s gpu {run['gpu_mem']}")
    if ok:
        run["answers"] = []
        for name, p, want in PROMPTS:
            for effort in ("none", "low"):
                try:
                    a = chat(p, effort, 1024)
                    a.update({"prompt": name, "effort": effort, "expected": want})
                    run["answers"].append(a)
                    tps = (a.get("timings") or {}).get("predicted_per_second")
                    log(f"  {name}/{effort}: {str(a['content'])[:60]!r} tok/s={tps}")
                except Exception as e:
                    run["answers"].append({"prompt": name, "effort": effort, "error": repr(e)})
                    log(f"  {name}/{effort}: error {e!r}")
    try:
        os.killpg(srv.pid, 15)
        srv.wait(60)
    except Exception:
        try:
            os.killpg(srv.pid, 9)
        except Exception:
            pass
    run["exit_code"] = srv.poll()
    res["runs"].append(run)
    json.dump(res, open(f"{OUT}/smoke_results.json", "w"), indent=1)
    time.sleep(5)

res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(res, open(f"{OUT}/smoke_results.json", "w"), indent=1)
log("done")
