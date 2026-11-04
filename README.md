# Half-Trained

Code behind the Half-Trained YouTube channel: watch AI learn, fails included.

Every AI in our videos is trained for real with the code here. Each experiment folder has the
environment, training and evaluation scripts, settings, seeds and results.

## Experiments
| # | Video | Folder |
|---|---|---|
| 01 | AI Learns to Land a Rocket. Then I Break One Engine. | `experiments/v01-rocket` |

## Setup
Python 3.12:
```
pip install torch "gymnasium[box2d]" stable-baselines3
```

MIT License.
