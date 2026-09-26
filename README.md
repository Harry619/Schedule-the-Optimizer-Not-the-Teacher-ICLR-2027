# Schedule the Optimizer, Not the Teacher

Code and data for the anonymous submission:

> **Schedule the Optimizer, Not the Teacher: Diagnosing and Mitigating Long-Horizon Collapse in On-Policy Self-Distillation**

This repository contains everything needed to reproduce the paper's results:
the modified training stack, all training/evaluation scripts, the full result
ledger (313 records), per-problem scores for every evaluation, the canonical
bootstrap analysis that produces every confidence interval in the paper, and
the figure-generation pipeline.

## Summary

On-policy self-distillation (OPSD,
[Zhao et al.](https://arxiv.org/abs/2601.18734),
[official code](https://github.com/siyan-zhao/OPSD)) lets a language model
teach itself: a frozen copy of the initial policy, conditioned on privileged
ground-truth solutions, provides dense per-token supervision on the student's
own rollouts. The official protocol evaluates only the first 100 training
steps. We extend the horizon and find:

1. **Long-horizon collapse.** Under the default constant learning rate, OPSD
   performance peaks around step 75–100 and then collapses below the base
   model by step 300, at all three scales we tested (Qwen3-1.7B/4B/8B).
2. **The cause is optimization-side, not teacher-side.** Periodic teacher
   refreshes do not prevent collapse (they catastrophically amplify it),
   while a simple cosine LR decay to zero removes most of it. Terminal
   performance is governed primarily by the cumulative LR dose
   $B=\int\eta\,dt$.
3. **An EMA teacher is not an escape.** It matches the best early score of
   the frozen-teacher family, yet ends the horizon below the frozen teacher's
   own endpoint at every rate tested (32.5–32.9 vs. 35.1 across EMA
   0.99–0.9995).
4. Secondary studies: difficulty-aware weighting (DAW) and an RLVR hybrid
   objective are statistically indistinguishable from the baseline within
   budget and do not affect the collapse.

## Repository layout

```
├── opsd_train.py          # entry point (extends official repo with the flags below)
├── opsd_trainer.py        # trainer: generalized JSD, DAW, hybrid, refresh, EMA, fp32, sym-clip
├── data_collator.py       # collator (+ ground-truth answer passthrough for the hybrid objective)
├── grpo_train.py          # GRPO baseline (verbatim from the official repo)
├── sft_train.py           # SFT baseline (verbatim from the official repo)
├── accelerate.yaml        # official 4-GPU DeepSpeed config
├── environment.yml        # conda environment (verbatim from the official repo)
├── requirements.txt       # pip alternative to environment.yml (same pinned versions)
├── eval/
│   ├── evaluate_math.py   # evaluation harness (official repo) — AIME24/25, HMMT25, MATH500, ...
│   └── run_eval.sh
├── scripts/
│   ├── train/             # one script per experiment family (see REPRODUCING.md)
│   ├── eval/              # checkpoint / base-model / MATH500 evaluation drivers
│   └── tests/             # CPU-runnable unit tests (reward parsing, refresh save)
├── analysis/
│   ├── compact_results.json         # full result ledger, 313 records (append-only)
│   ├── per_problem_scores.json      # per-problem correctness for all 128 evaluation files
│   ├── bootstrap_canonical.py       # reproduces ALL 24 confidence intervals in the paper
│   ├── bootstrap_canonical_0915.json# canonical CI output (single source of truth)
│   ├── gen_figures.py               # regenerates every matplotlib figure in the paper
│   ├── logs/                        # training logs behind the loss/divergence figures
│   ├── extract_per_problem.py       # raw eval JSONs -> per_problem_scores.json
│   ├── compact_results.py           # raw eval JSONs -> compact_results.json records
│   ├── analysis_0915.py             # original canonical run (needs raw eval JSONs)
│   ├── e6_divergence_full300.json   # 300-step unclipped-divergence trajectory
│   └── format_rate_table.py / estimate_compute.py
└── REPRODUCING.md         # paper claim -> command mapping
```

## Setup

```bash
# option A: conda (verbatim from the official repo)
conda env create -f environment.yml
conda activate opsd

# option B: plain pip
pip install -r requirements.txt
# note: flash-attn 2.8.3 usually needs a prebuilt wheel matching your torch/CUDA
# (see https://github.com/Dao-AILab/flash-attention/releases)
```

Verified stack: Python 3.10, torch 2.8.0, trl 0.26.0, vllm 0.11.0,
deepspeed 0.18.2, flash-attn 2.8.3, on 2–8×A800-80GB.

Training data: [`siyanzhao/Openthoughts_math_30k_opsd`](https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd)
(loaded automatically by `opsd_train.py`). Evaluation benchmarks are fetched
by `eval/evaluate_math.py` (AIME 2024/2025, HMMT-Feb-2025, MATH500, ...).
Base models: Qwen3-1.7B / Qwen3-4B / Qwen3-8B (Base).

All scripts take paths from `$OPSD_ROOT` (default `/root/autodl-tmp`, the
AutoDL container layout we used). Set `OPSD_ROOT` and place models under
`$OPSD_ROOT/models/` to adapt to your machine.

## Quick start

```bash
# 1) 100-step baseline reproduction (official protocol), 4 GPUs
bash scripts/train/run_opsd_1b.sh

# 2) 300-step long-horizon run with constant LR (collapse) vs cosine decay (fix)
bash scripts/train/long_run_300.sh baseline_long 0,1,2,3 none                       # constant LR
bash scripts/train/long_run_300.sh lrdecay      0,1,2,3 none --lr_scheduler_type cosine --max_steps 300

# 3) evaluate a checkpoint on AIME24/AIME25/HMMT25 (paper protocol: Avg@12, temp 1.0)
bash scripts/eval/eval_ckpt.sh <checkpoint_dir> <tag> 0,1,2,3

# 4) reproduce every CI in the paper from the shipped per-problem scores
cd analysis && python bootstrap_canonical.py

# 5) regenerate every figure
python gen_figures.py
```

See [REPRODUCING.md](REPRODUCING.md) for the complete claim-by-claim mapping.

## Compute accounting

`analysis/estimate_compute.py` recomputes the server-side total from the
shipped ledger: about 1,343 A800 GPU-hours (68 run families, 297 competition
evaluations, 16 MATH500 evaluations). The paper's compute appendix reports
about 1,850 GPU-hours (88 runs, 451 evaluations); that fuller ledger
additionally counts external-machine replications, which the paper prices by
same-type A800 anchors and discloses as analog-estimated. The two numbers
differ in scope, not in error: the repository script covers what the shipped
artifacts alone can recompute.

## New trainer flags introduced by this work

| Flag | Purpose |
|---|---|
| `--difficulty_aware {bandpass,hard_gate,easy_gate,hard_emph}` | per-sample difficulty-aware weighting (DAW) by global-batch divergence percentile |
| `--daw_eps` | numerical floor for DAW weights |
| `--hybrid_alpha` | mix an RLVR (verifiable-reward) term into the OPSD loss |
| `--teacher_refresh_every N` | merge the student LoRA into the base every N steps and reset the teacher to the current student |
| `--use_ema_teacher --ema_decay D` | replace the frozen teacher with an EMA of the student weights |
| `--jsd_symmetric_clip` | also clip negative per-vocab-item divergences (mechanism probe) |
| `--kl_fp32` | upcast logits to fp32 before the divergence (bf16-defect control) |

All new flags are configuration-gated and default to the official behavior.

## Relationship to the official OPSD repository

This repository is a fork of the official OPSD code
([github.com/siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD),
commit `7448751`). Files marked "verbatim" are unmodified; all other changes
are gated behind the flags above, so the default code path is bit-identical
to the official implementation. `grpo_train.py`, `sft_train.py`,
`environment.yml`, `accelerate.yaml` and `eval/` are unmodified upstream files.

## License

MIT (see [LICENSE](LICENSE)). Upstream files remain the property of the
original authors; please cite the original OPSD paper if you use this code.
