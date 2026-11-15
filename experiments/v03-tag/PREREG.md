# Tag: tiny agents vs large language models. Rules fixed before the games

Written on Sun 15 Nov 2026 (UAE) and committed to this repo before the first recorded game.
Anything changed later is listed at the bottom with the date and the reason. Finished games are never discarded.

## The players

**Tiny.** The seed-1 agents from the self-play runs in this folder: `runs/open_s1` for the open room and `runs/wall_s1`
for the room with one wall. Final weights, moves sampled from the policy as in training, one move every 0.1 s.
Each agent has 39,178 parameters; 20,105 of them choose the moves (the rest estimate the score during training).

**Giants.** Free access only. One user message per decision, no system prompt, the provider's default sampling settings,
at most 4,096 output tokens, no memory between decisions.

| Model | Access | Model ID | Notes |
|---|---|---|---|
| Gemini 4 Hydrogen | Google AI Studio free API | `gemini-4-hydrogen` | thought summaries requested |
| gpt-oss-120b | Groq free API | `openai/gpt-oss-120b` | |
| Qwen3.8-27B | Groq free API | `qwen/qwen3.8-27b` | |
| Nemotron 3 Ultra 550B | OpenRouter `:free` | `nvidia/nemotron-3-ultra-550b-a55b:free` | 50 requests a day, so it is expected to finish match 1 only |
| Qwen3.5 4B | this Mac, Ollama, Q4_K_M | `qwen3.5:4b` | thinking off: with thinking on, one move takes minutes here |

## How a model plays

- The game is paused while a model answers, and while a free quota refills. Waiting never counts against a model.
- At each decision the model gets the text built by `describe()` in `llm_play.py` (prompt version 1): the rules, its own
  position and velocity, the opponent's position and velocity, the offset and distance between them, the free space to
  the nearest wall in 8 directions, and the time. This is what our agent's 18 input numbers contain. In the wall room
  the model is also told where the wall is.
- It picks one of the same 9 moves (no push, or a push in one of 8 directions). The move is held for 0.5 s (`--k 5`).
- A reply counts only if it contains `MOVE: n` with n from 0 to 8. Anything else (no MOVE line, an empty reply, a reply
  cut off at the token cap) means the model stands still for that 0.5 s. These are counted and reported as invalid.
- No retries for a better answer and no replays: every finished game counts.
  If the program is stopped in the middle of a game, it continues that game from the last saved decision point.

## The matches

Seeds 90000 to 90004: 5 games per match and model, the same starting positions for everyone. Order: match 1, 2, 3.

1. **Wall runner.** The model is the runner in the room with the wall. The chaser is our `wall_s1` chaser.
2. **Open chaser.** The model is the chaser in the open room. The runner is our `open_s1` runner.
3. **Open runner.** The model is the runner in the open room. The chaser is our `open_s1` chaser.

Not played: a model as chaser in the wall room. Our own chaser catches the trained runner there only 1-4% of the time,
so there is nothing to compare against.

**Extended set.** A model that finishes its 5 games in all three matches before the deadline continues with seeds
90005 to 90019 in the same order. Reported separately from the 5-game set.

**Deadline.** Tue 17 Nov 2026, 23:59 UAE. Games not finished by then are listed as not played.

## Controls (no API, same harness)

Seeds 90000 to 90499 (500 games) in each match, in place of the model:
- our own agent as trained (a move every 0.1 s);
- our own agent with each move held for 0.5 s, the same handicap the models have (`agent@5:`);
- `greedy`, a scripted bot: push straight at the runner, or straight away from the chaser, re-aimed every 0.5 s;
- `random`: a random move every 0.1 s.

## What we will report

For every model and match: caught x of n, mean game length, invalid replies, seconds and tokens per decision, and for the
wall room the loops around the wall and the share of time the wall blocks the line of sight (as in `analyze.py`).
Every game, prompt, reply and exposed reasoning text is in `runs/llm/`.

Questions fixed in advance:
1. Wall runner: is any model caught as rarely as our trained runner?
2. Open chaser: does any model catch our runner within 30 s? Our chaser does, every time, in about 5 s.
3. Open runner: how long does each model last? Our runner lasts about 5 s.

## Exhibition (only if quota remains; reported separately)

Model against model in the open room, 0.5 s moves for both: each ordered pair among Gemini 4 Hydrogen, gpt-oss-120b
and Qwen3.8-27B plays one game (6 games, seeds 91000 to 91005).

## Before this file

Test games and single test requests were used to build the harness and are not results: 6 local games with `qwen3.5:4b`
(4 finished, 2 stopped early) on earlier prompt drafts, and up to 3 single requests to each API.

## Changes after registration

**15 Nov 2026, 20:30 UAE, before any recorded model game.** No change to the prompt, the moves, the matches or the seeds.

1. **One more control: a tiny runner trained with the same handicap.** The controls above were run first (500 games each,
   `runs/controls/`). In the wall room our runner as trained was caught 12 times in 500. The same runner with each move
   held for 0.5 s was caught 492 times in 500. So holding a move for 0.5 s, which every model has to do, is a large
   handicap by itself. To show what a tiny agent can do under that handicap, the wall runner keeps training with 0.5 s
   moves against the same fixed chaser (`train_hold.py`, 10 million decisions, seeds 1 to 3 → `runs/hold_wall_s1..3`),
   and then plays the same 500 control games as `agent@5:runs/hold_wall_s1`. Question 1 is answered against both
   runners: as trained (a move every 0.1 s) and trained with 0.5 s moves.
2. **More is recorded per game, nothing is played differently:** the time each decision was made and each game finished,
   and for the wall room the loops, line-of-sight and near-wall numbers (`tagstats.py`). `--until` stops new games
   from starting after the deadline. A game that finishes after the deadline is listed as not played.
3. **Cerebras is not used.** Its trial key answered "payment required" on the single test request. It was never in the
   table above.
4. **Request counts so far, corrected:** OpenRouter got 3 single test requests (2 were stopped after waiting minutes for
   an answer); Gemini and each Groq model got 1.

**15 Nov 2026, 20:50 UAE, while the games are running.** Again nothing about the prompt, the moves, the matches or the seeds.

5. **Gemini 4 Hydrogen's free API allows 20 requests a day** (the API's own error message, 20:35). Its first game
   stopped at 10.0 s of game time after 20 decisions, uncaught so far, and continues when the quota refills (midnight
   Pacific time). At that rate it can finish about one 30-second game before the deadline. The rule stands: what is not
   finished by the deadline is listed as not played. No second project or key is used to get around the limit.
6. **A model that sends no complete answer within 20 minutes is asked the same question again.** The first request
   to Nemotron hung for 10 minutes without an answer. No answer that arrives is ever dropped.
7. **Two more requests that are not results:** one Groq request to `qwen/qwen3.8-27b` to read a rate-limit message,
   and one local test game (`qwen3.5:4b`, seed 70000, open room) to check change 6.
8. **The local model's server was restarted at 20:51** with its prompt cache off: it had grown to 12 GB of memory.
   The game in progress (`qwen3.5:4b`, open chaser, seed 90003) continued from its last saved decision, at 11.5 s.

**15 Nov 2026, 21:50 UAE: two more players, added before any of their games.** The matches, seeds, deadline, 0.5 s
moves and reporting are the same as for everyone else.

9. Gemini 4 Hydrogen's limit of 20 requests a day was found only once the games had started. The AI Studio rate-limit
   page then showed which Google models have room for full sets on the free tier, and two are added:

   | Model | Access | Model ID | Notes |
   |---|---|---|---|
   | Gemini 3.5 Flash-Lite | Google AI Studio free API (500 requests a day) | `gemini-3.5-flash-lite` | plays exactly like the other chat models: same prompt, same reply format |
   | Gemini Decision 1 | Google's Decisions API, free tier (10,000 requests a day) | `gemini-decision-1` | not a chat model: it returns odds for a list of options and cannot write |

   **How Gemini Decision 1 is asked.** It gets the same text as the others up to and including the "Free space…" line.
   Then one multiple-choice question: "Which move should you make for the next 0.5 seconds?", with the nine moves as
   the options (`CRITERIA` in `llm_play.py`). Its move is the option the API returns as its choice. The odds it gives
   for all nine options are recorded with every decision. It gives no WHY sentence. An answer that is not one of the
   nine options counts as invalid and it stands still.
10. **Requests that are not results:** one Decisions API request and one throwaway game (seed 70001, wall room) for
    Gemini Decision 1; one request to Gemini 3.5 Flash-Lite.
11. **The local model's games are paused from 22:19 on 15 Nov** to free memory for other work on this Mac. Its registered
    5-game sets were already complete in all three matches; it had played 20, 18 and 5 games (wall runner, open chaser,
    open runner). The extended set continues from the saved decision when the model server is started again.
