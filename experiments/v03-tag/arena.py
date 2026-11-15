"""Two-player tag in a square arena, vectorised over many games (numpy).
Agent 0 = chaser, agent 1 = runner. Same top speed. A catch is when they touch."""
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

SIZE, RAD, DT, ACC, DRAG, VMAX, T_MAX = 10.0, 0.3, 0.1, 6.0, 2.0, 2.0, 300
ANG = np.arange(8) * np.pi / 4
DIRS = np.vstack([[0.0, 0.0], np.round(np.c_[np.cos(ANG), np.sin(ANG)], 12)])
RAYS = np.round(np.c_[np.cos(ANG), np.sin(ANG)], 12)
LAYOUTS = {"open": np.zeros((0, 4)), "wall": np.array([[4.8, 2.5, 5.2, 7.5]])}
OBS = 18


def ray_dists(p, rects):
    n = len(p); out = np.empty((n, 8))
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, (rx, ry) in enumerate(RAYS):
            tx = (SIZE - p[:, 0]) / rx if rx > 0 else (-p[:, 0] / rx if rx < 0 else np.full(n, np.inf))
            ty = (SIZE - p[:, 1]) / ry if ry > 0 else (-p[:, 1] / ry if ry < 0 else np.full(n, np.inf))
            d = np.minimum(tx, ty)
            for x0, y0, x1, y1 in rects:
                if rx != 0: a, b = (x0 - p[:, 0]) / rx, (x1 - p[:, 0]) / rx; lo_x, hi_x = np.minimum(a, b), np.maximum(a, b)
                else: inside = (p[:, 0] >= x0) & (p[:, 0] <= x1); lo_x, hi_x = np.where(inside, -np.inf, np.inf), np.where(inside, np.inf, -np.inf)
                if ry != 0: a, b = (y0 - p[:, 1]) / ry, (y1 - p[:, 1]) / ry; lo_y, hi_y = np.minimum(a, b), np.maximum(a, b)
                else: inside = (p[:, 1] >= y0) & (p[:, 1] <= y1); lo_y, hi_y = np.where(inside, -np.inf, np.inf), np.where(inside, np.inf, -np.inf)
                near, far = np.maximum(lo_x, lo_y), np.minimum(hi_x, hi_y)
                hit = (near <= far) & (far > 0)
                d = np.where(hit, np.minimum(d, np.maximum(near, 0)), d)
            out[:, k] = d
    return out


def collide(p, v, rects):
    for ax in (0, 1):
        lo, hi = p[:, ax] < RAD, p[:, ax] > SIZE - RAD
        p[:, ax] = np.clip(p[:, ax], RAD, SIZE - RAD)
        v[lo, ax] = np.maximum(v[lo, ax], 0); v[hi, ax] = np.minimum(v[hi, ax], 0)
    for x0, y0, x1, y1 in rects:
        q = np.c_[np.clip(p[:, 0], x0, x1), np.clip(p[:, 1], y0, y1)]
        dv = p - q; dist = np.linalg.norm(dv, axis=1)
        hit = dist < RAD
        if not hit.any(): continue
        inside = hit & (dist < 1e-9)
        if inside.any():
            pen = np.c_[p[inside, 0] - x0, x1 - p[inside, 0], p[inside, 1] - y0, y1 - p[inside, 1]]
            k = pen.argmin(1); nrm = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])[k]
            dv[inside] = nrm * 1e-6; dist[inside] = 1e-6
        nrm = dv[hit] / dist[hit, None]
        p[hit] = q[hit] + nrm * RAD
        vn = (v[hit] * nrm).sum(1)
        v[hit] -= np.minimum(vn, 0)[:, None] * nrm
    return p, v


class Arena:
    def __init__(self, n, layout="open", seed=0):
        self.n, self.rects, self.rng = n, LAYOUTS[layout], np.random.default_rng(seed)
        self.pos, self.vel = np.zeros((n, 2, 2)), np.zeros((n, 2, 2))
        self.t, self.dprev = np.zeros(n, int), np.zeros(n)

    def free(self, m):
        pts = self.rng.uniform(RAD + 0.5, SIZE - RAD - 0.5, (m, 2))
        for x0, y0, x1, y1 in self.rects:
            bad = (pts[:, 0] > x0 - RAD - 0.3) & (pts[:, 0] < x1 + RAD + 0.3) & (pts[:, 1] > y0 - RAD - 0.3) & (pts[:, 1] < y1 + RAD + 0.3)
            while bad.any():
                pts[bad] = self.rng.uniform(RAD + 0.5, SIZE - RAD - 0.5, (bad.sum(), 2))
                bad = (pts[:, 0] > x0 - RAD - 0.3) & (pts[:, 0] < x1 + RAD + 0.3) & (pts[:, 1] > y0 - RAD - 0.3) & (pts[:, 1] < y1 + RAD + 0.3)
        return pts

    def spawn(self, idx):
        idx = np.flatnonzero(idx) if idx.dtype == bool else idx
        if not len(idx): return
        a, b = self.free(len(idx)), self.free(len(idx))
        far = np.linalg.norm(a - b, axis=1) < 4.0
        while far.any():
            b[far] = self.free(far.sum()); far = np.linalg.norm(a - b, axis=1) < 4.0
        self.pos[idx, 0], self.pos[idx, 1] = a, b
        self.vel[idx] = 0; self.t[idx] = 0
        self.dprev[idx] = np.linalg.norm(a - b, axis=1)

    def obs(self, i):
        j = 1 - i; p, q = self.pos[:, i], self.pos[:, j]
        d = np.linalg.norm(q - p, axis=1, keepdims=True)
        return np.hstack([p / SIZE * 2 - 1, self.vel[:, i] / VMAX, (q - p) / SIZE, self.vel[:, j] / VMAX, d / SIZE,
                          np.minimum(ray_dists(p, self.rects), SIZE) / SIZE, (self.t / T_MAX)[:, None]]).astype(np.float32)

    def step(self, acts):
        for i in (0, 1):
            a = DIRS[acts[:, i]] * ACC
            v = self.vel[:, i] + a * DT
            v *= 1 - DRAG * DT
            sp = np.linalg.norm(v, axis=1, keepdims=True)
            v = np.where(sp > VMAX, v / np.maximum(sp, 1e-9) * VMAX, v)
            p = self.pos[:, i] + v * DT
            p, v = collide(p, v, self.rects)
            self.pos[:, i], self.vel[:, i] = p, v
        self.t += 1
        d = np.linalg.norm(self.pos[:, 0] - self.pos[:, 1], axis=1)
        catch = d < 2 * RAD
        r_chaser = catch * 1.0 + 0.02 * (self.dprev - d)
        self.dprev = d
        return r_chaser, catch, self.t >= T_MAX


class TagVec(VecEnv):
    """SB3 vector env for one role; the other role is played by opponent policies, one per env group."""

    def __init__(self, n, role, layout="open", seed=0):
        super().__init__(n, spaces.Box(-np.inf, np.inf, (OBS,), np.float32), spaces.Discrete(9))
        self.role, self.A = role, Arena(n, layout, seed)
        self.groups = [(None, np.arange(n))]
        self.ret, self.len = np.zeros(n), np.zeros(n, int)

    def set_groups(self, groups): self.groups = groups

    def reset(self):
        self.A.spawn(np.ones(self.num_envs, bool)); self.ret[:] = 0; self.len[:] = 0
        return self.A.obs(self.role)

    def step_async(self, actions): self._a = np.asarray(actions).reshape(-1)

    def step_wait(self):
        n = self.num_envs; oo = self.A.obs(1 - self.role); ao = np.zeros(n, int)
        for pol, idx in self.groups:
            ao[idx] = pol(oo[idx]) if pol is not None else self.A.rng.integers(0, 9, len(idx))
        acts = np.zeros((n, 2), int); acts[:, self.role] = self._a; acts[:, 1 - self.role] = ao
        rc, catch, timeout = self.A.step(acts)
        r = rc if self.role == 0 else -rc
        done = catch | timeout
        self.ret += r; self.len += 1
        obs = self.A.obs(self.role)
        infos = [{} for _ in range(n)]
        for i in np.flatnonzero(done):
            infos[i] = {"terminal_observation": obs[i].copy(), "TimeLimit.truncated": bool(timeout[i] and not catch[i]),
                        "episode": {"r": float(self.ret[i]), "l": int(self.len[i]), "t": 0.0}, "caught": bool(catch[i])}
        if done.any():
            self.A.spawn(done); self.ret[done] = 0; self.len[done] = 0
            obs[done] = self.A.obs(self.role)[done]
        return obs, r.astype(np.float32), done, infos

    def close(self): pass
    def get_attr(self, name, indices=None): return [getattr(self, name)] * self.num_envs
    def set_attr(self, name, value, indices=None): setattr(self, name, value)
    def env_method(self, name, *a, indices=None, **k): return [None] * self.num_envs
    def env_is_wrapped(self, cls, indices=None): return [False] * self.num_envs
    def seed(self, seed=None): return [None] * self.num_envs


def play(n, chaser, runner, layout="open", seed=0, record=0):
    """One game per slot. chaser/runner: obs -> actions (None = random). Returns outcomes and the first `record` games."""
    A = Arena(n, layout, seed); A.spawn(np.ones(n, bool))
    live, caught, t_end = np.ones(n, bool), np.zeros(n, bool), np.full(n, T_MAX)
    tracks = [] if record else None
    while live.any():
        acts = np.zeros((n, 2), int)
        for i, pol in ((0, chaser), (1, runner)):
            o = A.obs(i); acts[:, i] = pol(o) if pol is not None else A.rng.integers(0, 9, n)
        if record: tracks.append(np.c_[A.pos[:record].reshape(record, 4), acts[:record]].round(3))
        _, c, to = A.step(acts)
        newly = live & (c | to); caught |= newly & c; t_end = np.where(newly, A.t, t_end); live &= ~newly
        A.t[~live] = T_MAX + 1
    rec = None
    if record:
        tr = np.stack(tracks, 1)
        rec = [{"steps": int(t_end[k]), "caught": bool(caught[k]), "xy": tr[k, : t_end[k], :4].tolist(), "act": tr[k, : t_end[k], 4:].astype(int).tolist()} for k in range(record)]
    return caught, t_end, rec
