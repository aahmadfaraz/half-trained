# v02 · I Trained a Tiny ChatGPT on Shakespeare to See What's Inside

A small GPT, written from scratch, reads Tiny Shakespeare one character at a time and learns to continue it.
We watch it learn, ask it "What is the meaning of life?", then cut its memory (the context window) from 256 letters to 32, 8 and 1.
Every number in the video comes from the files in this folder.

**Watch the video:** link coming when it's published (Sun 15 Nov 2026)

## Results

**Setup**

| What | Number |
|---|---|
| Model | GPT, one character per token: 6 layers, 6 heads, 384 wide, 256-letter context |
| Parameters | 10,795,841 |
| Vocabulary | 65 characters |
| Data | Tiny Shakespeare, 1,115,394 characters: the first 90% (1,003,854) for training, the last 10% (111,540) kept unseen |
| Training | 5,000 steps of 16,384 letters each (64 × 256), on an M4 GPU (PyTorch MPS) |
| Time (main run) | step 2,000 at 45.0 min, step 5,000 at 109 min (snapshot time excluded) |
| Model used in the video | step 2,000, the lowest loss on unseen text of the 14 snapshots (1.474). By step 5,000 it rose to 1.720. |

**Learning.** At every snapshot the model reads `ROMEO:\nBut soft, what light through yonder window breaks?` and gives its odds
for each next letter. "Chance of the h" is its chance for the last letter of "through" after seeing "throug".
"Real words" is the share of words (3+ letters) in six new 400-letter samples (temperature 0.8) that appear in Tiny Shakespeare.

| Step | Time | Chance of the "h" | Real words |
|---|---|---|---|
| 0 | 0 min | 1.4% | 0.8% |
| 100 | 2.2 min | 31.1% | 15.3% |
| 200 | 4.5 min | 60.5% | 23.0% |
| 300 | 6.7 min | 86.4% | 56.7% |
| 500 | 11.4 min | 99.6% | 81.5% |
| 1,000 | 23.1 min | 99.97% | 93.1% |
| 2,000 | 45.0 min | 99.98% | 93.8% |

**Asking it a question.** Prompt: `YOU:\nWhat is the meaning of life?\n\nHAMLET:\n`. "HAMLET" and "meaning of life" never appear in Tiny Shakespeare.
Its odds for the first letter of the reply (main, step 2,000):

| | I | W | A | T | M | other 60 |
|---|---|---|---|---|---|---|
| Raw score (logit) | 6.69 | 6.57 | 6.56 | 6.52 | 6.34 | |
| Chance | 13.4% | 11.9% | 11.7% | 11.3% | 9.4% | 42.2% |
| Chance at temperature 0.8 | 15.7% | 13.5% | 13.3% | 12.7% | 10.1% | 34.6% |

All nine replies this model gave to the question (temperature 0.8, step 2,000, seed 1), first line of each. These are all of them, none left out.
None mentions life; every one hands the scene to another character.

| # | Reply | Next speaker |
|---|---|---|
| 1 | Well, well. | BRUTUS |
| 2 | Why, my lord; | KING HENRY VI |
| 3 | It will stand these woes will then again. | ROMEO |
| 4 | Right, my gracious lord! for 'tis post down. | KING HENRY VI |
| 5 | You are as pointed by me; so much if you | MISTRESS OVERDONE |
| 6 | Nay, I'll do remember me to my greatness. | GREMIO |
| 7 | O heavens! what lady heavens be proportion'd | ESCALUS |
| 8 | If he may not call them along; | ISABELLA |
| 9 | I love him to see his good will. | HENRY BOLINGBROKE |

Reply 1 is the step-2,000 snapshot sample (`runs/main/snap_02000.json`, `samples.chat`).
Replies 2-9 are the chat traces of the same model (`runs/eval/main_2000.json`, `traced[8:16]`).
The other seeds answer differently: see `traced[8:16]` in `runs/eval/main_s2b_2000.json` and `main_s3_2000.json`.

At temperature 0 (always the top pick, 600 letters, main seed 1) it writes "I will not so, I will not stay the ground.",
then KING RICHARD III: "Then be the stroke of the world with the world.", then QUEEN ELIZABETH and KING RICHARD III take turns saying
"The king is the king of the king of the king." until the 600 letters run out. With no randomness the next letter depends only
on the last 256 letters, so once the window holds nothing but the loop, it can't leave.

**Memory.** Same model, data and steps; only the context length changes, and the batch grows to keep 16,384 letters per step.
Three seeds per size, all at step 2,000 (the 1-letter seed 2 at step 5,000; it is flat after step 300). Mean (range):

| Memory | Loss on unseen text | e^loss | Real words | Next speaker is one the last speaker talks to in the book |
|---|---|---|---|---|
| 256 letters | 1.462 (1.453-1.467) | 4.3 | 96.8% | 67.3% (63.7-72.8%) |
| 32 | 1.463 (1.456-1.477) | 4.3 | 97.6% | 10.3% (9.6-10.7%) |
| 8 | 1.497 (1.487-1.506) | 4.5 | 97.0% | 10.7% (10.4-11.5%) |
| 1 | 2.450 (2.446-2.454) | 11.6 | 14.4% | writes no real names |
| Real unseen text | | | 92.7% | 99.0% |

- Loss: average over 8,000 positions of the unseen text, each with a full window. e^loss is roughly how many letters it is choosing between.
  Cutting memory from 256 to 32 letters leaves its spelling almost unchanged; at 1 letter it falls apart.
- Real words and speakers: 48 samples × 700 letters per model (temperature 0.8). Real words are 3+ letter words found in the training part.
- "Talks to" means the two speak one after the other somewhere in the book. A random pair of speakers does that 9.0% of the time,
  so with 32 or 8 letters the speaker changes are barely better than random.
- With 8 letters it loses track of line ends: 17.6% of its lines run past 80 letters (256 letters: 0.0%; real text: 0.0%).

The 32-letter model's first sample (`runs/eval/mem32_2000.json`) goes GREGORY → POLIXENES → KING RICHARD III → SICINIUS.
In Tiny Shakespeare those speakers sit at 40%, 64-76%, 24-28% and 1-13% of the file: four plays in four turns.
Given the same text up to the end of GREGORY's speech (113 letters), the 256-letter model's top guess for the next speaker's first letter
is S at 73.6% (SAMPSON is GREGORY's partner in Romeo and Juliet). The 32-letter model, which can no longer see the name, spreads it out:
C 11.1%, P 10.0%, L 8.8%, B 8.5%, S 7.9%. In the 256-letter model, head 3 of layer 5 puts 97.7% of its attention on the "Y" of GREGORY,
107 letters back (99.8% on the name's letters). The whole name "SAMPSON:" gets 34.9% from the 256-letter model and 0.1% from the 32-letter one.

## Reproduce

Python 3.12, then `pip install -r ../../requirements.txt` (this experiment only needs torch).

```
# Tiny Shakespeare, checked against its SHA-256
bash get_data.sh

# seed 1 (the code's default seed, 1337): 256 letters of memory, then 32, 8 and 1
python train.py --name main
python train.py --name mem32 --block 32 --batch 512
python train.py --name mem8 --block 8 --batch 2048
python train.py --name mem1 --block 1 --batch 16384

# seeds 2 and 3, same pattern for every memory size, e.g.
python train.py --name main_s2b --block 256 --batch 64 --seed 2 --keep 2000,5000
python train.py --name main_s3 --block 256 --batch 64 --seed 3 --keep 2000,5000

# evaluate on CPU, then summarise
python evaluate.py --run main --step 2000
python evaluate.py --run main --step 5000
python evaluate.py --real
python name_loss.py main:2000 mem32:2000 mem8:2000 mem1:2000 main:5000
python summarise.py
```

`memory_runs.sh`, `seed_runs.sh`, `seed_runs2.sh` and `eval_queue.sh` hold the full queue of 13 training runs and their
evaluations (macOS: they use `caffeinate`; set `PY=` to your Python). Seed 2 of the 256-letter model was run twice
(`main_s2` kept only its step-5,000 weights); the two copies match within 0.001 loss at every snapshot.

Weights (`runs/*/weights/*.pt`) and `data/tinyshakespeare.txt` are not committed; the commands above recreate them.
GPU training is not bit-for-bit repeatable, so retrained numbers will be close, not identical. Evaluation and the analysis
scripts run on CPU and give identical files when rerun on the same weights.

**On-screen odds and attention** (run from this folder, after training):

```
python analysis/logits.py main 2000 analysis/out/logits.json
python analysis/fullodds.py main 2000 analysis/inputs/chat_prompt.txt analysis/inputs/We.txt analysis/out/odds_We.json
python analysis/probs2.py main 2000 analysis/inputs/chat_prompt.txt analysis/inputs/chat_reply_200.txt analysis/out/probs2_chat.json
python analysis/probs.py mem32 2000 analysis/inputs/newline.txt analysis/inputs/gregory_sample.txt analysis/out/probs_mem32_gregory.json
python analysis/attn.py main 2000 analysis/inputs/gregory.txt 113 analysis/out/attn_gregory_main.json
python analysis/attn.py mem32 2000 analysis/inputs/gregory.txt 113 analysis/out/attn_gregory_mem32.json
python analysis/attn.py main 2000 analysis/inputs/chat_full.txt 56 analysis/out/attn_chat_B.json
python analysis/attn_at.py main 2000 analysis/inputs/chat_prompt.txt analysis/inputs/chat_reply.txt analysis/out/attn_chat_end.json
python analysis/probs2.py main 2000 analysis/inputs/gregory_113.txt analysis/inputs/sampson.txt analysis/out/sampson_main.json
python analysis/probs2.py mem32 2000 analysis/inputs/gregory_113.txt analysis/inputs/sampson.txt analysis/out/sampson_mem32.json
```

## Files
| Path | What |
|---|---|
| `get_data.sh` | downloads Tiny Shakespeare and checks its SHA-256 |
| `train.py` | the model and training loop; 14 snapshots per run (loss, samples, real words, letter-by-letter odds on the Romeo line) and weights |
| `evaluate.py` | CPU evaluation: loss on unseen text, loss by letters seen, speakers and line lengths in 48 samples, copying, traced chat replies |
| `name_loss.py` | speaker names in the unseen text: loss on name letters, and how often each model guesses the next speaker's first letter |
| `summarise.py` | turns `runs/eval/*.json` into `runs/eval/summary.md` |
| `*.sh` | the queues that ran all 13 trainings and evaluations |
| `runs/<run>/` | config, final summary, loss every 10 steps (`loss.csv`), snapshots and training log for each run |
| `runs/eval/` | evaluation results, `summary.md`, `name_loss.json`; `v1/` is an earlier pass of the seed-1 runs, before handovers were counted |
| `analysis/*.py` | raw scores and temperature-0 reply (`logits.py`), full odds (`fullodds.py`), per-letter odds (`probs.py`, `probs2.py`), attention (`attn.py`, `attn_at.py`) |
| `analysis/inputs/` | the texts they read: the chat prompt, the step-2,000 snapshot reply (`chat_reply.txt`, all 400 letters; `chat_reply_200.txt`, the first 200; `chat_full.txt`, prompt + reply), and the 32-letter model's first sample (`gregory_sample.txt`; `gregory.txt` drops its leading newline) |
| `analysis/out/` | their outputs |
