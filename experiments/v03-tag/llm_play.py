"""Tag between any two players: our trained agents, scripted bots, or language models.

python llm_play.py --chaser agent:runs/wall_s1 --runner gemini:gemini-4-hydrogen --layout wall --games 5 --out runs/llm/wall_runner/gemini-4-hydrogen.json

Player specs: agent:<run folder> (our PPO policy, a move every step) · agent@5:<run folder> (same, each move held 5 steps)
              random · greedy (scripted: straight at / straight away)
              ollama:<name> · groq:<id> · openrouter:<id> · gemini:<id>
The game pauses while a model thinks, and while a free quota refills. A model sees what our agent sees, written as
text, and picks one of the same 9 moves, held for --k steps. Rules for the recorded matches: PREREG.md.
Keys come from the environment or ENV_FILE. A game in progress is saved to <out>.partial and picked up again on restart."""
import argparse, json, os, pickle, re, ssl, sys, time, urllib.error, urllib.request
import numpy as np, torch
from stable_baselines3 import PPO
from arena import Arena, LAYOUTS, SIZE, RAD, VMAX, ACC, DRAG, T_MAX, DT, ray_dists
from tagstats import wall_stats

PROMPT_V = 1
CTX = ssl.create_default_context(cafile="/etc/ssl/cert.pem") if os.path.exists("/etc/ssl/cert.pem") else None
DIRN = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
URLS = {"groq": "https://api.groq.com/openai/v1/chat/completions", "openrouter": "https://openrouter.ai/api/v1/chat/completions"}
KEYS = {"groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
PATIENCE = 48 * 3600


class Abort(Exception):
    """The provider could not answer at all. The game stays saved and continues on the next start; it is not a result."""


def load_env():
    path = os.environ.get("ENV_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"))
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                if v.strip(): os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def describe(A, me, layout, k):
    them = 1 - me; p, v, q, w = A.pos[0, me], A.vel[0, me], A.pos[0, them], A.vel[0, them]
    rays = np.minimum(ray_dists(A.pos[:, me], A.rects)[0], SIZE)
    role = ("You are the RUNNER: avoid being touched until the time runs out." if me == 1
            else "You are the CHASER: touch the runner before the time runs out.")
    wall = ("" if layout == "open" else "There is one solid wall inside the room: a thin vertical block from (4.8, 2.5) to (5.2, 7.5). "
            "Nobody can pass through it. You can go around it through the gap above it (y > 7.5) or the gap below it (y < 2.5).\n")
    other = "runner" if me == 0 else "chaser"
    return ("You are playing a game of tag against one opponent. " + role + "\n\n"
            "The room is a 10 by 10 square. x grows to the east (right), y grows to the north (up). The outer walls are solid.\n" + wall +
            f"Both players are circles of radius {RAD:.1f} with the same top speed ({VMAX:.1f} per second). "
            f"A touch happens when the two centres are closer than {2 * RAD:.1f}. A game lasts at most {T_MAX * DT:.0f} seconds.\n"
            "A move is a push in one of 8 compass directions, or no push. "
            f"Every {DT:.1f} s, a push adds {ACC * DT:.1f} to your velocity in that direction, then friction removes {DRAG * DT:.0%} of your velocity, "
            f"and your speed is capped at {VMAX:.1f}. So you keep some of your old speed, and turning takes a moment.\n"
            f"You will be asked again every {k * DT:.1f} seconds with fresh numbers. You have no memory of earlier turns.\n\n"
            f"Right now (time {A.t[0] * DT:.1f} s of {T_MAX * DT:.0f} s):\n"
            f"You: position ({p[0]:.2f}, {p[1]:.2f}), velocity ({v[0]:.2f}, {v[1]:.2f}).\n"
            f"The {other}: position ({q[0]:.2f}, {q[1]:.2f}), velocity ({w[0]:.2f}, {w[1]:.2f}). "
            f"From you it is ({q[0] - p[0]:+.2f}, {q[1] - p[1]:+.2f}) away, distance {np.linalg.norm(q - p):.2f}.\n"
            "Free space from you to the nearest wall, by direction: " + ", ".join(f"{d} {r:.1f}" for d, r in zip(DIRN, rays)) + ".\n\n"
            f"Choose your move for the next {k * DT:.1f} seconds: 0 = no push, " + ", ".join(f"{i + 1} = {n}" for i, n in enumerate(DIRN)) + ".\n"
            "Reply with exactly two lines:\nWHY: <one short sentence>\nMOVE: <number>")


def parse(text):
    """Returns (move, valid, why). Without a 'MOVE: n' line the reply is invalid and the player stands still."""
    text = text or ""
    m = re.findall(r"MOVE\s*[:=]\s*\**\s*([0-8])\b", text, re.I)
    w = re.findall(r"WHY\s*:\s*\**\s*(.+)", text, re.I)
    why = w[-1].strip().strip("*").strip()[:300] if w else ""
    return (int(m[-1]), True, why) if m else (0, False, why)


def quota_wait(kind, msg, i):
    """Seconds to wait before asking again after a rate-limit answer."""
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?", msg)
    if m and any(m.groups()): return int(m[1] or 0) * 3600 + int(m[2] or 0) * 60 + float(m[3] or 0) + 2
    if re.search(r"per.?day|daily|\bRPD\b|\bTPD\b", msg, re.I):
        now = time.gmtime(); into = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
        reset = 8 * 3600 if kind == "gemini" else 0
        return (reset - into) % 86400 + 180
    m = re.search(r"retryDelay\W+([\d.]+)s", msg)
    return float(m[1]) + 2 if m else min(90, 5 * 2 ** min(i, 4))


def http(kind, url, headers, body):
    """Returns (json, headers, retries, seconds spent waiting, seconds the answer took)."""
    waited, i, fails = 0.0, 0, 0
    while True:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "User-Agent": "half-trained-tests/1.0", **headers})
        t0 = time.time(); msg = ""
        try:
            with urllib.request.urlopen(req, timeout=600, context=CTX) as r: out = json.load(r); h = {k.lower(): v for k, v in r.headers.items()}
            err = out.get("error") if isinstance(out, dict) else None
            if not err: return out, h, i, round(waited, 1), round(time.time() - t0, 2)
            msg = json.dumps(err)[:4000]
            try: code = int(err.get("code"))
            except (AttributeError, TypeError, ValueError): code = 429 if re.search(r"rate.?limit|quota", msg, re.I) else 500
        except urllib.error.HTTPError as e:
            code = e.code; msg = e.read()[:4000].decode(errors="replace")
            if e.headers.get("retry-after") and "try again in" not in msg: msg += f" try again in {e.headers['retry-after']}s"
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            code = 599; msg = f"{type(e).__name__}: {e}"
        if code == 429: wait = quota_wait(kind, msg, i)
        elif code >= 500: fails += 1; wait = min(300, 10 * 2 ** min(fails, 5))
        else: raise RuntimeError(f"HTTP {code}: {msg[:300]}")
        if fails > 12 or waited + wait > PATIENCE: raise Abort(f"HTTP {code}: {msg[:300]}")
        if wait > 60: print(f"   waiting {wait / 60:.0f} min ({kind}: {msg[:140]!r})", flush=True)
        time.sleep(wait); waited += wait; i += 1


class LLM:
    k_own, is_llm = None, True

    def __init__(self, spec, max_out, pace, think="default"):
        self.spec = spec; self.kind, _, self.name = spec.partition(":")
        self.max_out, self.pace, self.last, self.limits, self.think = max_out, pace, 0.0, {}, think

    def ask(self, prompt):
        k, n = self.kind, self.name; msgs = [{"role": "user", "content": prompt}]
        gap = self.pace - (time.time() - self.last)
        if gap > 0: time.sleep(gap)
        if k == "ollama":
            body = {"model": n, "messages": msgs, "stream": False, "options": {"num_predict": self.max_out}}
            if self.think != "default": body["think"] = self.think == "on"
            r, h, retries, waited, secs = http(k, "http://localhost:11434/api/chat", {}, body)
            msg = r["message"]; text, think = msg.get("content", ""), msg.get("thinking", "")
            usage = {"in": r.get("prompt_eval_count"), "out": r.get("eval_count")}; served = r.get("model", n); stop = r.get("done_reason")
        elif k in URLS:
            body = {"model": n, "messages": msgs, ("max_tokens" if k == "openrouter" else "max_completion_tokens"): self.max_out}
            r, h, retries, waited, secs = http(k, URLS[k], {"Authorization": f"Bearer {os.environ[KEYS[k]]}"}, body)
            ch = (r.get("choices") or [{}])[0]; msg = ch.get("message") or {}; text = msg.get("content") or ""; stop = ch.get("finish_reason")
            think = msg.get("reasoning") or msg.get("reasoning_content") or ""
            u = r.get("usage") or {}; usage = {"in": u.get("prompt_tokens"), "out": u.get("completion_tokens")}; served = r.get("model", n)
            self.limits = {kk: v for kk, v in h.items() if kk.startswith("x-ratelimit")} or self.limits
        elif k == "gemini":
            key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_AI_STUDIO_API_KEY"]
            r, h, retries, waited, secs = http(k, f"https://generativelanguage.googleapis.com/v1beta/models/{n}:generateContent", {"x-goog-api-key": key},
                                               {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                                                "generationConfig": {"maxOutputTokens": self.max_out, "thinkingConfig": {"includeThoughts": True}}})
            c = (r.get("candidates") or [{}])[0]; parts = (c.get("content") or {}).get("parts", []); stop = c.get("finishReason")
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")); think = "".join(p.get("text", "") for p in parts if p.get("thought"))
            u = r.get("usageMetadata", {}); usage = {"in": u.get("promptTokenCount"), "out": u.get("candidatesTokenCount"), "think": u.get("thoughtsTokenCount")}
            served = r.get("modelVersion", n)
        else:
            raise ValueError(self.spec)
        self.last = time.time()
        inner = re.findall(r"<think>(.*?)</think>", text, re.S)
        if inner: think = (think + "\n" + "\n".join(inner)).strip(); text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        text = re.sub(r"^.*</think>", "", text, flags=re.S).strip()
        return {"text": text, "thinking": think, "usage": usage, "served": served, "stop": stop, "seconds": secs, "waited": waited, "retries": retries}

    def act(self, A, i, layout, k):
        prompt = describe(A, i, layout, k); r = self.ask(prompt); move, ok, why = parse(r["text"])
        return move, None, {"move": move, "valid": ok, "why": why, "prompt": prompt, **r}


class Agent:
    is_llm = False

    def __init__(self, run, role, k_own):
        self.pol = PPO.load(os.path.join(run, f"final_{'chaser' if role == 0 else 'runner'}.zip"), device="cpu").policy
        self.pol.set_training_mode(False); self.k_own = k_own

    def act(self, A, i, layout, k):
        with torch.no_grad():
            dist = self.pol.get_distribution(torch.as_tensor(A.obs(i)))
            a = int(dist.get_actions(deterministic=False)[0]); p = dist.distribution.probs[0].numpy()
        return a, [round(float(x), 3) for x in p], None


class Random:
    k_own, is_llm = 1, False
    def act(self, A, i, layout, k): return int(np.random.randint(0, 9)), None, None


class Greedy:
    k_own, is_llm = None, False
    def act(self, A, i, layout, k):
        d = A.pos[0, 1] - A.pos[0, 0]
        return int(round(np.arctan2(d[1], d[0]) / (np.pi / 4))) % 8 + 1, None, None


def make(spec, role, a):
    if spec == "random": return Random()
    if spec == "greedy": return Greedy()
    m = re.fullmatch(r"agent(?:@(\d+))?:(.+)", spec)
    if m: return Agent(m[2], role, int(m[1] or 1))
    return LLM(spec, a.max_out, a.pace, a.think)


def play_one(seed, layout, P, k, part):
    st = pickle.load(open(part, "rb")) if os.path.exists(part) else None
    if st and st["seed"] == seed:
        A = st["A"]; torch.set_rng_state(st["torch"]); np.random.set_state(st["np"])
        print(f"seed {seed}: continuing from {A.t[0] * DT:.1f} s", flush=True)
    else:
        A = Arena(1, layout, seed); A.spawn(np.array([True])); torch.manual_seed(seed); np.random.seed(seed)
        st = {"seed": seed, "A": A, "start": [round(float(c), 3) for c in (*A.pos[0, 0], *A.pos[0, 1])],
              "hold": [0, 0], "xy": [], "acts": [], "probs": [[], []], "decisions": []}
    while True:
        due = [A.t[0] % (P[i].k_own or k) == 0 for i in (0, 1)]
        if any(due[i] and P[i].is_llm for i in (0, 1)):
            st["torch"], st["np"] = torch.get_rng_state(), np.random.get_state()
            pickle.dump(st, open(part + ".tmp", "wb")); os.replace(part + ".tmp", part)
        new = []
        for i in (0, 1):
            if due[i]:
                st["hold"][i], pr, rec = P[i].act(A, i, layout, k)
                if rec: new.append({"t": round(float(A.t[0] * DT), 1), "player": i, "at": time.strftime("%Y-%m-%d %H:%M:%S"), **rec})
                if pr: st["probs"][i].append(pr)
        st["decisions"] += new
        _, catch, timeout = A.step(np.array([st["hold"]]))
        st["xy"].append([round(float(c), 3) for c in (*A.pos[0, 0], *A.pos[0, 1])]); st["acts"].append(list(st["hold"]))
        if catch[0] or timeout[0]: break
    if os.path.exists(part): os.remove(part)
    out = {"seed": seed, "caught": bool(catch[0]), "seconds": round(len(st["xy"]) * DT, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "start": st["start"], "xy": st["xy"], "acts": st["acts"], "decisions": st["decisions"]}
    for i, name in ((0, "chaser_probs"), (1, "runner_probs")):
        if st["probs"][i]: out[name] = st["probs"][i]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chaser", required=True); ap.add_argument("--runner", required=True)
    ap.add_argument("--layout", choices=list(LAYOUTS), default="open")
    ap.add_argument("--games", type=int, default=5); ap.add_argument("--seed0", type=int, default=90_000)
    ap.add_argument("--k", type=int, default=5, help="steps a model or scripted bot holds each move")
    ap.add_argument("--max_out", type=int, default=4096); ap.add_argument("--pace", type=float, default=0, help="seconds between calls to one model")
    ap.add_argument("--think", choices=["default", "on", "off"], default="default", help="local (ollama) models only")
    ap.add_argument("--light", action="store_true", help="keep only outcomes (for large control sets)")
    ap.add_argument("--until", default="", help="start no new game after this local time, e.g. '2026-11-17 23:59'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); load_env(); torch.set_num_threads(1)
    P = [make(a.chaser, 0, a), make(a.runner, 1, a)]
    cfg = {"chaser": a.chaser, "runner": a.runner, "layout": a.layout, "k": a.k, "seed0": a.seed0, "max_out": a.max_out, "think": a.think, "prompt_v": PROMPT_V}
    res = {**cfg, "games": [], "aborted": []}
    if os.path.exists(a.out):
        res = json.load(open(a.out))
        if any(res.get(kk) != v for kk, v in cfg.items()): sys.exit(f"{a.out} was made with different settings")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = {g["seed"] for g in res["games"]}
    for s in range(a.seed0, a.seed0 + a.games):
        if s in done: continue
        if a.until and time.time() > time.mktime(time.strptime(a.until, "%Y-%m-%d %H:%M")): print("deadline reached", flush=True); break
        try: g = play_one(s, a.layout, P, a.k, a.out + ".partial")
        except Abort as e:
            res["aborted"].append({"seed": s, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "reason": str(e)[:400]})
            json.dump(res, open(a.out, "w")); print(f"seed {s}: stopped, not a result ({str(e)[:200]})", flush=True); sys.exit(3)
        if a.layout == "wall": g.update(wall_stats(g["xy"]))
        if a.light: g = {kk: v for kk, v in g.items() if kk not in ("start", "xy", "acts", "decisions", "chaser_probs", "runner_probs")}
        res["games"].append(g); res["limits"] = [getattr(p, "limits", {}) for p in P]
        d = g.get("decisions", [])
        if d:
            bad = sum(not x["valid"] for x in d); secs = sum(x["seconds"] for x in d)
            toks = sum(sum(v or 0 for v in x["usage"].values()) for x in d)
            print(f"seed {s}: {'caught' if g['caught'] else 'survived'} at {g['seconds']} s · {len(d)} decisions, {bad} invalid, {secs:.0f} s thinking, {toks} tokens", flush=True)
        if d or s == a.seed0 + a.games - 1 or (s - a.seed0) % 100 == 99: json.dump(res, open(a.out, "w"))
    c = [g["caught"] for g in res["games"]]; t = [g["seconds"] for g in res["games"]]
    print(f"{a.chaser} chases {a.runner} ({a.layout}): caught {sum(c)}/{len(c)}, mean game {np.mean(t):.1f} s")
