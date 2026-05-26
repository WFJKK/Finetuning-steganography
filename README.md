# Acrostic Steganography Internalization

Can LLMs internalize steganographic encoding through supervised fine-tuning alone, without being told to do it at inference time?

This repo contains experiments using acrostics as a testbed: the first letter of each sentence in the model's output spells a hidden payload. We train models at increasing levels of internalization depth and measure where the behavior breaks down.

Part of the SPAR 2026 fellowship research (Project Poseidon), supervised by Dr. Robert Krzyzanowski at Poseidon Research.

## Internalization Ladder

| Level | Code name | Payload rule | What it tests |
|-------|-----------|-------------|---------------|
| I_0 | `stage1` | Explicit `<secret>` tag in prompt | Can the model do acrostics at all? |
| I_1 | `v0` | First letters of prompt words (no hint) | Pattern derivation from data alone |
| I_2 | `v1a` | I_1 payload reversed | Single-operation transformation |
| I_3 | `v1b` | I_1 payload Caesar +1 | Constant-dependent transformation |
| I_4 | `v2` | XOR with neighbours | Local cross-position operation |
| I_5 | `v3` | Majority vote with neighbours | Local cross-position operation |
| I_6 | `v4` | Per-word MD5 hash | Hash-based (unlearnable) |

> **Nomenclature note:** Scripts use `stage1`/`v0`/`v1a`/`v1b`/`v2`-`v6`; the paper uses I_0/I_1/.../I_6.

## Results Structure

```
results/
  model_scaling_I0/        # Fig: I_0 SER vs model size
    qwen2.5-{0.5b..72b}/   #   n100/ and n250/ (100 & 250 training examples)
  model_scaling_I1/        # Fig: I_1 SER vs model size
    qwen2.5-{1.5b..72b}/   #   500 training examples
  data_scaling_05b/        # Fig: I_0 data scaling at 0.5B
                           #   n100..n1000
  payload_scaling_7b/      # Fig: I_1 BER vs payload length
    stage1_{2,4,8,16,32}bit/
    stage1_32bit_ood/       #   OOD eval at 64 & 128 bit
  vladder_3b/              # Fig: Internalization ladder
    {s1,v0,v1a..v6}_{500,1000}ex.json
  domain_robustness/       # Domain comparison (5 domains)
    domain_{news,fiction,legal,poetry,technical}_{s1,v0}_8bit.json
```

## Data Structure

```
data/
  news/                    # Primary domain
    stage1_8bit/           # I_0 training data (train.jsonl, val.jsonl)
    v0_8bit/               # I_1 training data
    v1a_8bit/ .. v6_8bit/  # I_2..I_6 training data
    stage1_{2,4,16,32}bit/ # Payload sweep training data
    stage1_{64,128,256,512}bit_evalonly/  # OOD eval data
  fiction/                 # Domain robustness (I_0 + I_1 only)
  legal/
  poetry/
  technical/
```

## Quick Start

```bash
pip install torch transformers peft datasets accelerate bitsandbytes trl

# Train I_0 (explicit instruction)
python scripts/train.py \
    --model-size 7b --stage stage1 \
    --data data/news/stage1_8bit/train.jsonl \
    --output adapters/qwen2.5-7b/stage1

# Train I_1 (internalization, requires I_0 adapter)
python scripts/train.py \
    --model-size 7b --stage v0 \
    --data data/news/v0_8bit/train.jsonl \
    --output adapters/qwen2.5-7b/v0 \
    --stage1-adapter adapters/qwen2.5-7b/stage1/full/final

# Evaluate
python scripts/eval.py \
    --model-size 7b --stage stage1 --scheme acrostics \
    --adapter adapters/qwen2.5-7b/stage1/full/final \
    --data data/news/stage1_8bit/val.jsonl \
    --split val --n 100 \
    --output results/my_eval.json
```

## Models and Training

All experiments use Qwen2.5-Instruct (0.5B-72B) with 4-bit NF4 quantization and LoRA (r=16, alpha=32). Adapters on HF Hub: [WFJKK/poseidon-sft-adapters](https://huggingface.co/WFJKK/poseidon-sft-adapters).
