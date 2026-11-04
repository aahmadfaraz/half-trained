import time, torch, gymnasium as gym, platform
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
print("python", platform.python_version(), "torch", torch.__version__, "mps", torch.backends.mps.is_available(), "gymnasium", gym.__version__)
ids = [e for e in gym.envs.registry if "LunarLander" in e]; print("envs", ids)
env_id = "LunarLander-v3" if "LunarLander-v3" in ids else ids[0]
env = make_vec_env(env_id, n_envs=8, seed=0)
model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999, gae_lambda=0.98, ent_coef=0.01, device="cpu", seed=0, verbose=0)
t = time.time(); model.learn(total_timesteps=100_000); dt = time.time() - t
print(f"BENCH {env_id}: 100k steps in {dt:.1f}s -> {100_000/dt:.0f} steps/s")
m, s = evaluate_policy(model, gym.make(env_id), n_eval_episodes=10)
print(f"after 100k steps: mean reward {m:.1f} +/- {s:.1f} (solved = 200)")
