# The 12-Hour ML Engineer Crash Course (hiring-manager edition)

You have ~12 hours, no coding background, and an ML engineer interview. This course is
built around one fact about how hiring managers actually interview: **they do not grade
definitions, they grade depth probes.** Every topic gets asked at three levels:

1. *What is X?* (textbook)
2. *Why / when does X break?* (applied)
3. *Here is a messy situation involving X — what do you do and how would you know?* (experience)

Level 3 is where real experience shows. Every module here is written so you can answer
at level 3, and every module ends with the **"experience signals"** — the things people
only learn from being burned in production — and the **follow-up chains** interviewers use.

**Read this before the modules:** modules 01–11 are the *breadth* layer — accurate,
necessary vocabulary, and the same canon every prep guide contains. The *depth* layer is
[12-depth-layer.md](12-depth-layer.md) plus `code/10`–`code/12`: mechanisms you can
derive predictions from, each backed by an experiment you run yourself, and a
[lab notebook](notebook-template.md) that turns those runs into true, citable experience.
The "experience signals" sections in 01–09 coach phrasings about habits you do not have;
treat them as *things you now understand*, and use notebook statements instead
(12 §1 and §7 explain how). If your time is short, the depth layer outranks modules 07 and 11.

## The honest strategy (read this first)

In 12 hours you will not become a senior ML engineer, and an experienced interviewer will
detect a fabricated background within one follow-up question. That is not a reason to
despair; it is a reason to pick the strategy that can actually win:

- **Do not claim experience you don't have.** Bluffing is the single fastest way to lose a
  technical interview. Fabricated "war stories" collapse under "what was the actual number?"
- **Do show reasoning, vocabulary precision, and awareness of failure modes.** A candidate
  who says *"I haven't run this in production, but my understanding is the two-tower model
  can't use user-item cross features, which is why you need a separate ranking stage"* sounds
  far stronger than one who vaguely claims to have "built recommender systems."
- **Turn gaps into questions.** *"I don't know how you handle delayed conversion labels here —
  do you use an attribution window?"* signals maturity.
- **Your actual story is a strong card.** "I met X by accident, got invited, and spent 12
  hours building a structured understanding of the field — here's the repo" demonstrates
  learning velocity and initiative, which are two of the top things managers hire for in
  junior/mid candidates. Use it; don't hide it.
- **Aim for the right outcome.** The realistic win is: an impressed interviewer, a referral
  into a junior/associate ML or data role, a take-home project, or a "come back in 3 months."
  Pursue that explicitly.

## What a hiring manager is actually scoring

| Signal | What it looks like | What kills it |
|---|---|---|
| Problem solving | Clarifies the question, states assumptions, reasons from first principles | Jumping to a buzzword answer |
| Technical depth | Survives 3 levels of "why?" | Definitions without mechanisms |
| Practical judgment | Baselines first, suspects data before models, knows cost/latency tradeoffs | "I'd use a transformer" for a tabular problem |
| Honesty / calibration | "I don't know, here's how I'd find out" | Confident wrong answers |
| Communication | Structured answers, checks in, uses precise terms | Rambling, vague terms ("the AI learns") |
| Learning velocity | Visible curiosity, fast uptake of hints | Defensiveness when corrected |

## The 12-hour schedule

Study in this order. Times include running the code. If you run out of time, the triage
order is at the bottom.

| Hours | Module | Why it's here |
|---|---|---|
| 0:00–1:15 | [01 Python essentials](01-python-essentials.md) + `code/01_python_basics.py` | You must be able to read and write basic Python; coding screens are common |
| 1:15–2:00 | [02 Math minimum](02-math-minimum.md) | Only the math that interviews actually touch |
| 2:00–3:30 | [03 ML fundamentals](03-ml-fundamentals.md) + `code/07_metrics.py`, `code/08_sklearn_tabular.py` | Most interview questions live here (metrics, leakage, overfitting, GBDTs) |
| 3:30–5:00 | [04 Deep learning](04-deep-learning.md) + `code/02_numpy_mlp_from_scratch.py` | Backprop, optimizers, normalization, CNN/RNN/transformers |
| 5:00–5:15 | Break. Really. | Memory consolidation |
| 5:15–6:30 | [05 PyTorch](05-pytorch.md) + `code/03_torch_training_loop.py`, `code/04_attention_from_scratch.py` | The training loop and the bug catalog |
| 6:30–7:30 | [06 GPUs & CUDA](06-gpu-cuda.md) + `code/09_gpu_arithmetic.py` | How a GPU works, memory vs compute bound, OOM, mixed precision |
| 7:30–8:30 | [07 Architectures & LLMs](07-architectures-and-llms.md) | Transformers in depth, fine-tuning, inference, RAG |
| 8:30–9:45 | [08 Recommendation systems](08-recommendation-systems.md) + `code/05_matrix_factorization_recsys.py`, `code/06_two_tower_recsys.py` | Full industrial funnel; this is where "experience" questions concentrate |
| 9:45–10:30 | [09 ML systems & MLOps](09-ml-systems-and-mlops.md) | Train/serve skew, monitoring, drift, incidents |
| 10:30–11:00 | [10 Interview playbook](10-interview-playbook.md) | Question bank with level-3 answers, design template, scripts |
| 11:00–12:00 | [12 Depth layer](12-depth-layer.md) + `code/10`–`code/12` + [notebook](notebook-template.md) | Mechanisms, predictions, and your own measured numbers; this is what survives a depth probe |
| spare | [11 Flashcards](11-flashcards.md) | Rapid recall pass |

**Triage order if time runs out:** 03 → 12 (§4, §2.2–2.3, §5.2/5.6, §3.4 with their
scripts) → 10 → 04 → 05 → 08 → 06 → 09 → 01 → 07 → 02 → 11.
If you have a coding screen, move 01 to second place.

**Already read the modules?** Spend the remaining time on 12 and the notebook only.

**Sleep:** if the interview is right after the 12 hours, trade the last 2–3 hours of study
for sleep. Sleep-deprived reasoning loses more interviews than missing module 07 does.

## Running the code

Every script runs on CPU in seconds and prints what it is demonstrating.

```bash
cd crash-course
uv venv .venv && uv pip install --python .venv/bin/python numpy scikit-learn
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python code/01_python_basics.py
```

(`pip install numpy scikit-learn torch` also works; the CPU torch wheel just downloads faster.)

## How to use each module

1. Read it once straight through (don't take notes yet).
2. Run the script, read the script's comments — they are part of the lesson.
3. Go back to the **Follow-up chains** and say the answers out loud. Out loud. Interviews
   are spoken. Read the **Experience signals** as "things I now understand", never as
   claims about habits; the notebook entries from 12 are what you cite as experience.
4. Move on. Do not rabbit-hole; the flashcards will catch what slipped.

## The one-paragraph mental model of the whole field

Machine learning is **fitting a function to data by minimizing a loss**. Everything else is
detail: *which function family* (linear model, tree ensemble, neural network), *which loss*
(a differentiable stand-in for the metric you actually care about), *how to minimize it*
(gradient descent and its variants, computed by backpropagation, executed on GPUs because
it's mostly matrix multiplication), *how to know it generalizes* (held-out data, proper
splits, metrics matched to the business problem), and *how to keep it working in production*
(same features at training and serving time, monitoring for drift, retraining, experiments).
Recommendation systems are the most industrialized instance of all of this: a funnel of
cheap retrieval models followed by expensive ranking models, trained on biased logs of their
own past decisions. Large language models are the same loop at enormous scale — next-token
prediction with transformers — where the engineering problems are memory, bandwidth, and
cost. If you keep this paragraph in your head, you can derive a lot of answers on the spot.
