# 07 · Architectures and LLMs (60 minutes)

Module 04 covered the transformer mechanics. This module is the landscape: which models
exist, how LLMs are built, fine-tuned, served, and evaluated, and the numbers that let you
reason about cost.

## 1. The three transformer families

| Family | Attention | Pretraining objective | Used for | Examples |
|---|---|---|---|---|
| Encoder-only | bidirectional | masked language modeling | classification, NER, embeddings for retrieval/reranking | BERT, RoBERTa, DeBERTa, sentence-transformers |
| Decoder-only | causal | next-token prediction | generation, chat, reasoning, in-context learning | GPT, LLaMA, Mistral, Qwen, Claude, Gemini |
| Encoder–decoder | bidirectional enc, causal dec | span corruption / seq2seq | translation, summarization, ASR | T5, BART, Whisper |

Everything "LLM" today is decoder-only. Embedding models for RAG/recsys are usually
encoder-style (or decoder models with pooling) trained contrastively.

## 2. Tokenization

- Text → integer IDs via a learned **subword** vocabulary: **BPE** (merge the most frequent
  pairs; GPT), **WordPiece** (BERT), **SentencePiece/Unigram** (language-agnostic; LLaMA/T5).
  Vocab sizes 32k–256k. A token ≈ 0.75 English words; code and non-English are less
  efficient (more tokens per character → higher cost, shorter effective context).
- The tokenizer is **part of the model**: mismatched tokenizer versions silently break
  everything. Special tokens (`<bos>`, `<eos>`, `<pad>`, chat-template tokens) matter.
- Explains LLM quirks: poor at counting letters, arithmetic on digits tokenized unevenly,
  trailing-space sensitivity.

## 3. Pretraining and scaling

- Objective: maximize `Σ log P(token_t | tokens_<t)` over trillions of tokens of web, code,
  books; heavy **deduplication** and **quality filtering** matter as much as model size.
- **Compute**: training FLOPs ≈ **6 × N × D** (N params, D tokens; 2 forward + 4 backward
  per param-token). Example: 7B × 2T tokens = 8.4e22 FLOPs; on 1,000 H100s at 40% MFU
  (~4e17 FLOP/s) ≈ 2.4 days. Inference ≈ 2 × N FLOPs per token.
- **Scaling laws**: loss falls as a power law in N, D, and compute (Kaplan 2020).
  **Chinchilla** (2022): for a fixed compute budget, scale N and D together, ≈ **20 tokens
  per parameter** is compute-optimal. But inference cost scales with N, so production
  models are **over-trained** small models (LLaMA-3 8B saw 15T tokens ≈ 1,900 tokens/param).
- Context length is extended late in training (RoPE scaling: position interpolation,
  NTK-aware, YaRN); long-context quality ≠ long-context support ("lost in the middle").
- **Mixture of Experts (MoE)**: the FFN is replaced by N expert FFNs with a router that
  picks top-k (1–2) per token → many parameters, few active FLOPs (Mixtral 8×7B: 47B
  params, ~13B active). Needs load-balancing losses and expert parallelism; memory-heavy,
  compute-light.
- **GQA/MQA** (shared KV heads), **RMSNorm**, **SwiGLU**, **RoPE**, no biases: the modern
  "LLaMA recipe".

## 4. Post-training: from base model to assistant

1. **Supervised fine-tuning (SFT / instruction tuning)**: train on (prompt, ideal response)
   pairs with the same next-token loss (often masking the prompt tokens). Quality ≫ quantity
   (LIMA: 1k great examples). Catastrophic forgetting: low LR, mix in general data.
2. **Preference optimization**: collect comparisons (A better than B).
   **RLHF**: train a reward model on comparisons, then optimize the policy with PPO
   against it with a **KL penalty** to the SFT model (prevents reward hacking / collapse).
   **DPO**: skips the reward model and RL; directly increases the likelihood margin of
   preferred over rejected responses with an implicit KL constraint; simpler, stable,
   standard in open models. Variants: IPO, KTO, ORPO, GRPO (RL with verifiable rewards for
   reasoning models).
3. **Parameter-efficient fine-tuning (PEFT)**: **LoRA** freezes W and learns a low-rank
   update `ΔW = BA` (`r` = 8–64) on attention/FFN projections — ~0.1–1% trainable params,
   optimizer memory drops accordingly, adapters are swappable and can be merged into W at
   inference (zero latency cost). **QLoRA**: 4-bit quantized base + LoRA → fine-tune a 65B
   model on one 48 GB GPU. Prefix/prompt tuning and adapters are the other families.
4. **Distillation**: train a small student on a big teacher's outputs/logits (soft targets
   carry more information than hard labels).

When to fine-tune vs prompt vs RAG: fine-tune for *behavior/format/style/domain
language* and latency/cost; RAG for *knowledge that changes* or must be cited; prompting
first because it's cheapest and often enough.

## 5. Inference and decoding

- **Decoding**: greedy (argmax), beam search (translation), sampling with **temperature**
  (scale logits; <1 sharper, >1 flatter), **top-k**, **top-p/nucleus** (sample from the
  smallest set with cumulative prob ≥ p), repetition penalties. Temperature 0 ≠ deterministic
  across hardware/batches (floating point).
- **Two phases**: **prefill** (process the prompt; one big parallel pass; compute-bound;
  determines TTFT) and **decode** (one token at a time; reads all weights + KV cache per
  token; memory-bound; determines tokens/s).
- **KV cache**, **continuous batching**, **PagedAttention**, **speculative decoding**,
  **quantization** (GPTQ, AWQ, int8 SmoothQuant, FP8), **prefix caching**, **tensor
  parallel**: module 06 §11 covers each.
- Cost mental math: a 70B bf16 model needs ≥140 GB → 2×H100 minimum (TP=2), realistic
  serving at 4–8 GPUs; a 7B model fits one GPU with room for batch.
- Structured output: constrained decoding / grammars (JSON mode); function/tool calling =
  model emits a structured call, system executes it, result appended → **agents** are
  loops of this.

## 6. Retrieval-augmented generation (RAG)

Pipeline: documents → **chunking** (size/overlap; respect structure) → **embedding model**
→ vector index (FAISS/HNSW, Milvus, pgvector, Pinecone) → at query time embed the query,
retrieve top-k (**dense**), often **hybrid** with BM25 (**sparse**) via reciprocal rank
fusion → **rerank** with a cross-encoder (query and passage attended jointly; far more
accurate, far more expensive — same retrieval→ranking funnel as recsys) → stuff into the
prompt with instructions to cite → generate.

Failure modes an experienced person lists: bad chunk boundaries, embedding model trained
on a different domain, stale index after document updates, query–document asymmetry
(use instruction-tuned embeddings or HyDE), too many/few chunks (lost in the middle),
no metadata filtering (permissions!), evaluation only by vibes. Evaluate retrieval
(recall@k on labeled queries) **separately** from generation (faithfulness, answer
relevance; RAGAS-style or human).

## 7. Evaluation of LLMs

- **Perplexity** (pretraining health), **benchmarks** (MMLU knowledge, GSM8K math,
  HumanEval code, MT-Bench/Arena for chat) with the standing caveat of **contamination**
  (benchmark in the training data), **LLM-as-judge** (cheap, biased toward length,
  position, and its own outputs; calibrate against humans), **human preference** (gold,
  expensive), **task-specific** metrics (exact match, F1, pass@k).
- Production: offline eval set with rubric → online A/B on user outcomes; monitor refusal
  rate, latency, cost per request, hallucination reports; guardrails for prompt injection
  and PII.

## 8. Vision, speech, multimodal (what to recognize)

- **ResNet** (residual CNN; still the workhorse for small data/latency), **EfficientNet**,
  **ConvNeXt**.
- **ViT**: split the image into 16×16 patches, linearly embed, add positions, run a
  transformer encoder with a `[CLS]` token; needs lots of data or strong augmentation/
  pretraining; scales better than CNNs.
- **CLIP**: image encoder + text encoder trained contrastively on 400M pairs; zero-shot
  classification by comparing image embedding with "a photo of a {class}" embeddings; the
  backbone of multimodal models and image search.
- **Detection** (YOLO single-stage real-time; DETR transformer set prediction),
  **segmentation** (U-Net encoder–decoder with skip connections; SAM promptable
  segmentation), **diffusion** (DDPM denoising; latent diffusion = Stable Diffusion;
  classifier-free guidance; DiT replaces the U-Net with a transformer).
- **Whisper**: encoder–decoder transformer on log-mel spectrograms; robust ASR.
- **Multimodal LLMs** (LLaVA pattern): vision encoder (CLIP/SigLIP) → projector MLP →
  image tokens prepended to the LLM's input; trained in stages.
- **Graph neural networks**: message passing over neighbors; PinSage for Pinterest
  recommendations; knowledge-graph completion; molecules.
- **State-space models** (Mamba): linear-time sequence models competing with attention at
  long contexts; hybrids in production (Jamba).

## 9. Choosing an architecture (how to answer "what model would you use?")

1. Tabular → GBDT (then maybe an MLP with embeddings if you need multi-task/online).
2. Images, small data → pretrained ResNet/ViT fine-tune; large data → ViT/ConvNeXt.
3. Text classification → fine-tuned BERT-class encoder (cheap, fast) before reaching for an
   LLM; generation/extraction/reasoning → instruction-tuned LLM, prompt first, then LoRA.
4. Retrieval/matching → two-tower / bi-encoder + ANN, cross-encoder reranker.
5. Sequences of events (user histories) → transformer over the sequence (SASRec-style).
6. Time series → GBDT with lag features / simple baselines; deep models only at scale.

Always say: baseline first, measure, then the smallest model that meets the latency and
cost budget.

## Experience signals

- "6ND for training FLOPs, 2N per inference token — I use those to sanity-check cost."
- "Chinchilla is compute-optimal, not inference-optimal; production models are
  over-trained small models."
- "DPO for preference tuning unless you have the infra for PPO; LoRA for almost all
  fine-tuning; merge adapters for serving."
- "In RAG I evaluate retrieval recall separately from generation quality — most 'the LLM
  hallucinated' bugs are retrieval failures."
- "Prefill is compute-bound, decode is memory-bound; that's why TTFT and tokens/s are
  tuned differently."

## Follow-up chains

- "How would you build a customer-support assistant on our docs?" → *RAG first; chunking,
  hybrid retrieval, reranking, citations, eval set, guardrails; fine-tune only for tone/
  format later; monitor cost/latency.* → "Why not fine-tune on the docs?" → *knowledge
  changes, no citations, fine-tuning teaches style not facts reliably, update cost.*
- "LoRA vs full fine-tuning?" → "Where do you put the adapters, what rank?" → "Any
  quality loss?" → *small on most tasks; full FT still wins for large distribution shifts.*
- "What is temperature?" → "Why is temperature 0 still non-deterministic?" →
  *floating-point non-associativity, batch composition, kernel selection.*
- "How big a GPU for a 70B model?" → *140 GB in bf16 + KV cache → ≥2×H100 with TP;
  int4 ≈ 35 GB fits one.*
