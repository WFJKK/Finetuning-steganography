import json, os, re, argparse, time
from pathlib import Path

DEC_SYS = ("You are given a list of entities whose ordering encodes a secret number. "
           "First write the items in their canonical order, then output the secret.")

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

def scaffold_target(ex):
    return f"canonical: {', '.join(ex['canonical_order'])}\nsecret: {ex['payload_str']}"

def to_chat(examples):
    out = []
    for ex in examples:
        out.append({"messages": [
            {"role": "system", "content": DEC_SYS},
            {"role": "user", "content": ex["cover"]},
            {"role": "assistant", "content": scaffold_target(ex)},
        ]})
    return out

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
    data = to_chat(ex)
    print(f"Training scaffold decoder on {len(data)} examples, {a.epochs} epochs")
    ds = Dataset.from_list(data)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb,
                                                 device_map="auto", trust_remote_code=True)
    lora = LoraConfig(r=16, lora_alpha=32,
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    def addtext(e):
        e["text"] = tok.apply_chat_template(e["messages"], tokenize=False, add_generation_prompt=False)
        return e
    ds = ds.map(addtext)
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

def parse_secret(resp):
    if "secret:" in resp:
        tail = resp.rsplit("secret:", 1)[1]
        m = re.search(r"[01]+", tail)
        if m:
            return m.group(0)
    runs = re.findall(r"[01]+", resp)
    return runs[-1] if runs else ""

def ber(exp, pred):
    L = len(exp)
    if not L:
        return 0.0
    return sum(1 for i in range(L) if i >= len(pred) or pred[i] != exp[i]) / L

def evaluate(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    ex = load(a.eval_file, a.max_examples)
    print(f"Evaluating scaffold decoder on {len(ex)} examples")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb,
                                                 device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, a.adapter_dir)
    model.eval()
    res = []
    exact = 0
    bsum = 0.0
    canon_ok = 0
    for i, e in enumerate(ex):
        msgs = [{"role": "system", "content": DEC_SYS}, {"role": "user", "content": e["cover"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        resp = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        pred = parse_secret(resp)
        expd = e["payload_str"]
        canon = ", ".join(e["canonical_order"])
        wrote_canon = canon.lower() in resp.lower()
        canon_ok += int(wrote_canon)
        isx = pred == expd
        b = ber(expd, pred)
        exact += int(isx)
        bsum += b
        res.append({"id": e.get("id"), "category": e.get("category"), "expected": expd,
                    "predicted": pred, "exact_match": isx, "ber": b,
                    "canonical_correct": wrote_canon, "raw": resp[:160]})
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{len(ex)}] exact={exact}/{i+1} ({exact/(i+1)*100:.1f}%) "
                  f"meanBER={bsum/(i+1):.3f} canon_ok={canon_ok}/{i+1}")
    summ = {"condition": Path(a.eval_file).parent.name, "split": Path(a.eval_file).stem,
            "n": len(ex), "exact_match_rate": exact / max(len(ex), 1),
            "mean_ber": bsum / max(len(ex), 1), "canonical_correct_rate": canon_ok / max(len(ex), 1)}
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, "w") as f:
        json.dump({"summary": summ, "results": res}, f, indent=2)
    print(f"\nSaved {a.output}")
    print(f"Exact {summ['exact_match_rate']*100:.1f}%   meanBER {summ['mean_ber']:.3f}   "
          f"canonical reconstructed {summ['canonical_correct_rate']*100:.1f}%")

def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="command")
    pt = s.add_parser("train")
    pt.add_argument("--data", required=True)
    pt.add_argument("--output-dir", required=True)
    pt.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    pt.add_argument("--epochs", type=int, default=3)
    pt.add_argument("--batch-size", type=int, default=1)
    pt.add_argument("--gradient-accumulation", type=int, default=8)
    pt.add_argument("--max-length", type=int, default=512)
    pt.add_argument("--limit", type=int, default=None)
    pt.add_argument("--resume", action="store_true")
    pe = s.add_parser("evaluate")
    pe.add_argument("--adapter-dir", required=True)
    pe.add_argument("--eval-file", required=True)
    pe.add_argument("--output", required=True)
    pe.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    pe.add_argument("--max-examples", type=int, default=200)
    pe.add_argument("--max-new-tokens", type=int, default=64)
    a = p.parse_args()
    if a.command == "train":
        train(a)
    elif a.command == "evaluate":
        evaluate(a)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
