import gymnasium as gym

ENV_ID = "LunarLander-v3"
ACTION_NAMES = {0: "nothing", 1: "left engine", 2: "main engine", 3: "right engine"}
BROKEN_ACTION = {"none": None, "left": 1, "right": 3}


class BrokenEngine(gym.ActionWrapper):
    """The broken engine's action does nothing."""

    def __init__(self, env, broken="none"):
        super().__init__(env)
        self.dead = BROKEN_ACTION[broken]

    def action(self, act):
        return 0 if (self.dead is not None and int(act) == self.dead) else act


def make_env(broken="none", render_mode=None):
    return BrokenEngine(gym.make(ENV_ID, render_mode=render_mode), broken)
