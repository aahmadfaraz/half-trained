"""Free model APIs: groq:<id>, openrouter:<id> (":free" models only), gemini:<id>. Keys by name from the environment or ENV_FILE."""
import json, os, re, ssl, time, urllib.error, urllib.request

CTX = ssl.create_default_context(cafile="/etc/ssl/cert.pem") if os.path.exists("/etc/ssl/cert.pem") else None
URLS = {"groq": "https://api.groq.com/openai/v1/chat/completions", "openrouter": "https://openrouter.ai/api/v1/chat/completions"}
KEYS = {"groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY", "gemini": "GOOGLE_AI_STUDIO_API_KEY"}
PATIENCE = 6 * 3600


class Abort(Exception):
    pass


def load_env():
    path = os.environ.get("ENV_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"))
    if os.path.exists(path):
        for line in open(path):
            k, eq, v = line.strip().partition("=")
            if eq and not k.startswith("#") and v.strip(): os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def quota_wait(kind, msg, i):
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?", msg)
    if m and any(m.groups()): return int(m[1] or 0) * 3600 + int(m[2] or 0) * 60 + float(m[3] or 0) + 2
    if re.search(r"per.?day|daily|\bRPD\b|\bTPD\b|PerDay", msg, re.I): raise Abort("daily quota: " + msg[:300])
    m = re.search(r"retryDelay\W+([\d.]+)s", msg)
    return float(m[1]) + 2 if m else min(90, 5 * 2 ** min(i, 4))


def http(kind, url, headers, body, events, retry=True):
    waited, i, fails = 0.0, 0, 0
    while True:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "User-Agent": "research-client/1.0", **headers})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=300, context=CTX) as r:
                h = {k.lower(): v for k, v in r.headers.items()}; out = json.loads(r.read())
            err = out.get("error") if isinstance(out, dict) else None
            if not err: return out, h, i, round(waited, 1), round(time.time() - t0, 2)
            msg = json.dumps(err)[:3000]
            try: code = int(err.get("code"))
            except (AttributeError, TypeError, ValueError): code = 429 if re.search(r"rate.?limit|quota", msg, re.I) else 500
        except urllib.error.HTTPError as e:
            code = e.code; msg = e.read()[:3000].decode(errors="replace")
            if e.headers.get("retry-after") and "try again in" not in msg: msg += f" try again in {e.headers['retry-after']}s"
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            code = 599; msg = f"{type(e).__name__}: {e}"
        events.append({"t": round(time.time(), 1), "code": code, "msg": re.sub(r"org_[A-Za-z0-9]+", "org_<id>", msg)[:600]})
        if not retry: raise Abort(f"HTTP {code} (no retries): {msg[:300]}")
        if code == 429: wait = quota_wait(kind, msg, i)
        elif code >= 500: fails += 1; wait = min(300, 10 * 2 ** min(fails, 5))
        else: raise RuntimeError(f"HTTP {code}: {msg[:300]}")
        if fails > 8 or waited + wait > PATIENCE: raise Abort(f"HTTP {code}: {msg[:300]}")
        print(f"   {kind} {code}: waiting {wait:.0f} s", flush=True)
        time.sleep(wait); waited += wait; i += 1


class LLM:
    def __init__(self, spec, max_out=4096, retry=True):
        self.spec = spec; self.kind, _, self.name = spec.partition(":")
        if self.kind == "openrouter" and not self.name.endswith(":free"): raise ValueError("OpenRouter: only :free models")
        self.max_out, self.limits, self.events, self.retry = max_out, {}, [], retry

    def ask(self, system, prompt, extra=()):
        """extra: further (role, text) turns after the prompt, role "assistant" or "user"."""
        k, n = self.kind, self.name
        if k in URLS:
            body = {"model": n, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}] + [{"role": r, "content": t} for r, t in extra],
                    ("max_tokens" if k == "openrouter" else "max_completion_tokens"): self.max_out}
            r, h, retries, waited, secs = http(k, URLS[k], {"Authorization": f"Bearer {os.environ[KEYS[k]]}"}, body, self.events, self.retry)
            ch = (r.get("choices") or [{}])[0]; msg = ch.get("message") or {}; text = msg.get("content") or ""; stop = ch.get("finish_reason")
            think = msg.get("reasoning") or msg.get("reasoning_content") or ""
            u = r.get("usage") or {}; usage = {"in": u.get("prompt_tokens"), "out": u.get("completion_tokens")}; served = r.get("model", n)
            self.limits = {kk: v for kk, v in h.items() if kk.startswith("x-ratelimit")} or self.limits
        elif k == "gemini":
            body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": [{"text": prompt}]}] + [{"role": "model" if r == "assistant" else "user", "parts": [{"text": t}]} for r, t in extra],
                    "generationConfig": {"maxOutputTokens": self.max_out, "thinkingConfig": {"includeThoughts": True}}}
            r, h, retries, waited, secs = http(k, f"https://generativelanguage.googleapis.com/v1beta/models/{n}:generateContent",
                                               {"x-goog-api-key": os.environ[KEYS[k]]}, body, self.events, self.retry)
            c = (r.get("candidates") or [{}])[0]; parts = (c.get("content") or {}).get("parts", []); stop = c.get("finishReason")
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")); think = "".join(p.get("text", "") for p in parts if p.get("thought"))
            u = r.get("usageMetadata", {}); usage = {"in": u.get("promptTokenCount"), "out": u.get("candidatesTokenCount"), "think": u.get("thoughtsTokenCount")}
            served = r.get("modelVersion", n)
        else:
            raise ValueError(self.spec)
        inner = re.findall(r"<think>(.*?)</think>", text, re.S)
        if inner: think = (think + "\n" + "\n".join(inner)).strip(); text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        text = re.sub(r"^.*</think>", "", text, flags=re.S).strip()
        return {"text": text, "thinking": think, "usage": usage, "served": served, "stop": stop, "seconds": secs, "waited": waited, "retries": retries}
