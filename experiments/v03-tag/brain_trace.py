"""Replay one logged game with the same seed and record what happens inside the chaser at every step:
its 18 inputs, both hidden layers (128 each), its 9 move odds, plus the policy weights.
python brain_trace.py runs/llm/open_runner/control-agent.json 0 runs/brain_trace.json"""
import json, sys
import numpy as np, torch
import llm_play as L

src, gi, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ref = json.load(open(src))["games"][gi]
seed = ref["seed"]


class Traced(L.Agent):
    def __init__(self, run, role, k_own):
        super().__init__(run, role, k_own)
        net = self.pol.mlp_extractor.policy_net
        self.rows, self.h = [], {}
        acts = [m for m in net if isinstance(m, torch.nn.Tanh)]
        acts[0].register_forward_hook(lambda m, i, o: self.h.__setitem__(1, o[0].numpy().copy()))
        acts[1].register_forward_hook(lambda m, i, o: self.h.__setitem__(2, o[0].numpy().copy()))

    def act(self, A, i, layout, k):
        x = A.obs(i)[0].copy()
        a, p, rec = super().act(A, i, layout, k)
        self.rows.append({"in": [round(float(v), 3) for v in x], "h1": [round(float(v), 3) for v in self.h[1]],
                          "h2": [round(float(v), 3) for v in self.h[2]], "p": p, "a": a})
        return a, p, rec


chaser, runner = Traced("runs/open_s1", 0, 1), L.Agent("runs/open_s1", 1, 1)
g = L.play_one(seed, "open", [chaser, runner], 5, "/dev/null.part")
same = g["xy"] == ref["xy"]
pn = chaser.pol.mlp_extractor.policy_net
lin = [m for m in pn if isinstance(m, torch.nn.Linear)] + [chaser.pol.action_net]
W = [np.round(m.weight.detach().numpy(), 4).tolist() for m in lin]
json.dump({"seed": seed, "same_as_logged_game": same, "caught": g["caught"], "xy": g["xy"], "steps": chaser.rows,
           "W": W, "shapes": [list(np.array(w).shape) for w in W]}, open(out, "w"), separators=(",", ":"))
print("seed", seed, "steps", len(chaser.rows), "same as logged game:", same, "shapes", [np.array(w).shape for w in W])
