# Tag: models against our agents (from runs/llm, deadline 2026-11-17 23:59 UAE)

## Wall runner: the model runs, our wall chaser chases: Registered set (seeds 90000-90004)

| Player | Caught | Mean game | Invalid replies | Time per decision | Tokens in / out per decision | Loops | Wall blocks the view |
|---|---|---|---|---|---|---|---|
| control-agent | 0 of 5 | 30.0 s | 0 of 0 | – | – | 3.48 | 67% |
| control-agent@5 | 5 of 5 | 17.1 s | 0 of 0 | – | – | 1.8 | 38% |
| control-greedy | 5 of 5 | 5.64 s | 0 of 0 | – | – | 0.04 | 21% |
| control-hold-s1 | 0 of 5 | 30.0 s | 0 of 0 | – | – | 1.54 | 88% |
| control-random | 5 of 5 | 3.84 s | 0 of 0 | – | – | 0.02 | 30% |
| gemini-3.5-flash-lite | 5 of 5 | 6.26 s | 0 of 65 | 0.98 s | 526 / 30 | 0.22 | 19% |
| gemini-4-hydrogen | 1 of 1 | 18.7 s | 1 of 38 | 36.6 s | 556 / 35 | 1.93 | 20% |
| gemini-decision-1 | 5 of 5 | 4.44 s | 0 of 47 | 0.66 s | 634 / 0 | 0.18 | 35% |
| gpt-oss-120b | 5 of 5 | 5.48 s | 0 of 56 | 2.4 s | 567 / 831 | 0.08 | 20% |
| nemotron-3-ultra-550b | 5 of 5 | 5.06 s | 39 of 52 | 111.18 s | 529 / 3754 | 0.07 | 22% |
| qwen3.5-4b | 5 of 5 | 4.92 s | 0 of 52 | 4.34 s | 538 / 39 | 0.04 | 14% |
| qwen3.8-27b | 5 of 5 | 5.18 s | 0 of 54 | 0.21 s | 538 / 35 | 0.03 | 20% |

## Wall runner: the model runs, our wall chaser chases: Extended set (seeds 90005-90019)

| Player | Caught | Mean game | Invalid replies | Time per decision | Tokens in / out per decision | Loops | Wall blocks the view |
|---|---|---|---|---|---|---|---|
| control-agent | 0 of 15 | 30.0 s | 0 of 0 | – | – | 3.84 | 61% |
| control-agent@5 | 15 of 15 | 16.52 s | 0 of 0 | – | – | 1.84 | 26% |
| control-greedy | 15 of 15 | 5.29 s | 0 of 0 | – | – | 0.07 | 16% |
| control-hold-s1 | 0 of 15 | 30.0 s | 0 of 0 | – | – | 1.68 | 77% |
| control-random | 15 of 15 | 4.41 s | 0 of 0 | – | – | 0.03 | 19% |
| gemini-3.5-flash-lite | 15 of 15 | 5.45 s | 0 of 170 | 1.1 s | 526 / 31 | 0.17 | 17% |
| gemini-decision-1 | 15 of 15 | 4.41 s | 0 of 139 | 0.63 s | 634 / 0 | 0.24 | 27% |
| qwen3.5-4b | 15 of 15 | 4.81 s | 0 of 150 | 5.33 s | 538 / 39 | 0.11 | 17% |
| qwen3.8-27b | 15 of 15 | 5.87 s | 0 of 183 | 0.22 s | 538 / 35 | 0.12 | 14% |

- gemini-4-hydrogen: a game was still in progress at the deadline (not played).

## Open chaser: the model chases our open runner: Registered set (seeds 90000-90004)

| Player | Caught | Mean game | Invalid replies | Time per decision | Tokens in / out per decision |
|---|---|---|---|---|---|
| control-agent | 5 of 5 | 4.78 s | 0 of 0 | – | – |
| control-agent@5 | 5 of 5 | 10.14 s | 0 of 0 | – | – |
| control-greedy | 0 of 5 | 30.0 s | 0 of 0 | – | – |
| control-random | 0 of 5 | 30.0 s | 0 of 0 | – | – |
| gemini-3.5-flash-lite | 0 of 5 | 30.0 s | 0 of 300 | 0.93 s | 453 / 27 |
| gemini-decision-1 | 0 of 5 | 30.0 s | 0 of 300 | 0.68 s | 561 / 0 |
| gpt-oss-120b | 0 of 5 | 30.0 s | 11 of 300 | 3.71 s | 495 / 1466 |
| nemotron-3-ultra-550b | 0 of 1 | 30.0 s | 38 of 60 | 85.04 s | 458 / 3432 |
| qwen3.5-4b | 0 of 5 | 30.0 s | 0 of 300 | 5.34 s | 466 / 37 |
| qwen3.8-27b | 0 of 5 | 30.0 s | 0 of 300 | 0.21 s | 466 / 33 |

## Open chaser: the model chases our open runner: Extended set (seeds 90005-90019)

| Player | Caught | Mean game | Invalid replies | Time per decision | Tokens in / out per decision |
|---|---|---|---|---|---|
| control-agent | 15 of 15 | 5.49 s | 0 of 0 | – | – |
| control-agent@5 | 15 of 15 | 8.76 s | 0 of 0 | – | – |
| control-greedy | 0 of 15 | 30.0 s | 0 of 0 | – | – |
| control-random | 0 of 15 | 30.0 s | 0 of 0 | – | – |
| gemini-3.5-flash-lite | 0 of 15 | 30.0 s | 0 of 900 | 1.97 s | 453 / 26 |
| gemini-decision-1 | 3 of 15 | 27.03 s | 0 of 813 | 0.65 s | 561 / 0 |
| qwen3.5-4b | 0 of 15 | 30.0 s | 0 of 900 | 4.59 s | 466 / 37 |
| qwen3.8-27b | 0 of 8 | 30.0 s | 0 of 480 | 0.27 s | 466 / 34 |

- nemotron-3-ultra-550b: a game was still in progress at the deadline (not played).
- qwen3.8-27b: finished after the deadline, not played: seeds [90013].

## Open runner: the model runs, our open chaser chases: Registered set (seeds 90000-90004)

| Player | Caught | Mean game | Invalid replies | Time per decision | Tokens in / out per decision |
|---|---|---|---|---|---|
| control-agent | 5 of 5 | 4.78 s | 0 of 0 | – | – |
| control-agent@5 | 5 of 5 | 5.0 s | 0 of 0 | – | – |
| control-greedy | 5 of 5 | 4.18 s | 0 of 0 | – | – |
| control-random | 5 of 5 | 3.42 s | 0 of 0 | – | – |
| gemini-3.5-flash-lite | 5 of 5 | 3.68 s | 0 of 38 | 0.92 s | 454 / 30 |
| gemini-decision-1 | 5 of 5 | 3.04 s | 0 of 32 | 0.62 s | 561 / 0 |
| gpt-oss-120b | 4 of 4 | 4.42 s | 1 of 37 | 2.82 s | 495 / 1079 |
| qwen3.5-4b | 5 of 5 | 3.72 s | 0 of 38 | 4.92 s | 466 / 37 |
| qwen3.8-27b | 5 of 5 | 3.62 s | 0 of 37 | 0.21 s | 466 / 35 |

## Open runner: the model runs, our open chaser chases: Extended set (seeds 90005-90019)

| Player | Caught | Mean game | Invalid replies | Time per decision | Tokens in / out per decision |
|---|---|---|---|---|---|
| control-agent | 15 of 15 | 5.49 s | 0 of 0 | – | – |
| control-agent@5 | 15 of 15 | 5.54 s | 0 of 0 | – | – |
| control-greedy | 15 of 15 | 4.46 s | 0 of 0 | – | – |
| control-random | 15 of 15 | 3.46 s | 0 of 0 | – | – |
| gemini-3.5-flash-lite | 2 of 2 | 4.4 s | 0 of 19 | 3.37 s | 453 / 32 |
| gemini-decision-1 | 15 of 15 | 3.2 s | 0 of 103 | 0.73 s | 561 / 0 |
| qwen3.5-4b | 15 of 15 | 3.42 s | 0 of 110 | 4.81 s | 465 / 36 |

- gemini-3.5-flash-lite: finished after the deadline, not played: seeds [90007].
- gpt-oss-120b: finished after the deadline, not played: seeds [90004].

Players named control-* are the controls, played through the same harness (20 games each here; the 500-game controls are in runs/controls).
Models with all three registered sets done: control-agent, control-agent@5, control-greedy, control-random, gemini-3.5-flash-lite, gemini-decision-1, qwen3.5-4b, qwen3.8-27b.
