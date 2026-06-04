import json, os, argparse, time
from pathlib import Path

DEC_SYS = ("You are given a list of entities whose ordering encodes a secret number. "
           "Output the secret as a binary string.")

def load(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows

def build_text(system, user, think, bits):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{think}\n</think>\n\n{bits}<|im_end|>\n")

def has_ckpt(d):
    p = Path(d)
    return p.exists() and any(c.name.startswith("checkpoint-") for c in p.iterdir() if c.is_dir())

def train(a):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    ex = load(a.data, a.limit)
    print(f"Training Qwen3 trace decoder on {len(ex)} traces, {a.epochs} epochs")

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = []
    for e in ex:
        rows.append({"text": build_text(DEC_SYS, e["cover"], e["think"], e["payload_str"])})
    ds = Dataset.from_list(rows)

    print("\n" + "=" * 60)
    print("FIRST RENDERED TRAINING EXAMPLE:")
    print("=" * 60)
    print(ds[0]["text"][:600])
    print("..." if len(ds[0]["text"]) > 600 else "")
    print("=" * 60 + "\n")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb,
                                                 device_map="auto", trust_remote_code=True)
    lora = LoraConfig(r=16, lora_alpha=32,
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = SFTConfig(output_dir=a.output_dir, num_train_epochs=a.epochs,
                     per_device_train_batch_size=a.batch_size,
                     gradient_accumulation_steps=a.gradient_accumulation,
                     learning_rate=2e-4, warmup_ratio=0.03, logging_steps=10,
                     save_strategy="epoch", bf16=True, optim="paged_adamw_8bit",
                     report_to="none", max_grad_norm=0.3, lr_scheduler_type="cosine",
                     max_length=a.max_length)
    tr = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)

    resume = a.resume and has_ckpt(a.output_dir)
    print("Resuming from checkpoint" if resume else "Starting fresh")
    t = time.time()
    tr.train(resume_from_checkpoint=resume)
    print(f"Training complete in {(time.time()-t)/60:.1f} minutes")
    tr.save_model(a.output_dir)
    tok.save_pretrained(a.output_dir)
    print(f"Adapter saved to {a.output_dir}")

def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="command")
    pt = s.add_parser("train")
    pt.add_argument("--data", required=True)
    pt.add_argument("--output-dir", required=True)
    pt.add_argument("--model", default="Qwen/Qwen3-8B")
    pt.add_argument("--epochs", type=int, default=3)
    pt.add_argument("--batch-size", type=int, default=1)
    pt.add_argument("--gradient-accumulation", type=int, default=8)
    pt.add_argument("--max-length", type=int, default=2048)
    pt.add_argument("--limit", type=int, default=None)
    pt.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.command == "train":
        train(a)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
