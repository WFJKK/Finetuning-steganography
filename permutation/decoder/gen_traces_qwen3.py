import json, os, re, argparse, time
from pathlib import Path

DEC_GEN_SYS = (
 "You decode a secret binary string hidden in the ordering of a list of items.\n"
 "Definitions:\n"
 "- canonical order = the items sorted alphabetically.\n"
 "- The given list is the canonical order with some adjacent pairs swapped.\n"
 "- Pairs are fixed by canonical position: pair 1 = canonical items 1 and 2, pair 2 = canonical items 3 and 4.\n"
 "Steps:\n"
 "1. Sort the items alphabetically (canonical order).\n"
 "2. For pair 1, find canonical items 1 and 2 in the GIVEN list. If they appear in alphabetical order, bit=0; if reversed, bit=1.\n"
 "3. Do the same for pair 2 (canonical items 3 and 4).\n"
 "4. The secret is written LAST pair first: bit for pair 2, then bit for pair 1.\n"
 "Keep your reasoning short: do the sort, read each pair once, do not second-guess. "
 "Then end with exactly:\nsecret: <bits>"
)

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

def split_think(resp):
    if "</think>" in resp:
        i = resp.index("</think>")
        think = resp[:i].replace("<think>", "").strip()
        return think, resp[i+len("</think>"):], True
    return resp.strip(), "", False

def answer_bits(ans):
    if "secret:" in ans:
        tail = ans.rsplit("secret:", 1)[1]
        m = re.search(r"[01]+", tail)
        if m:
            return m.group(0)
    runs = re.findall(r"[01]+", ans)
    return runs[-1] if runs else ""

def done_ids(path):
    ids = set()
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                ids.add(json.loads(line).get("id"))
    return ids

def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--verbose-first", type=int, default=2)
    a = ap.parse_args()

    ex = load(a.data, a.limit)
    already = done_ids(a.output)
    ex = [e for e in ex if e.get("id") not in already]
    print(f"{len(ex)} examples to process ({len(already)} already done)")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb,
                                                 device_map="auto", trust_remote_code=True)
    model.eval()

    kept = 0
    seen = 0
    not_closed = 0
    fout = open(a.output, "a")
    for i, e in enumerate(ex):
        msgs = [{"role": "system", "content": DEC_GEN_SYS}, {"role": "user", "content": e["cover"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=True,
                                 temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                                 pad_token_id=tok.pad_token_id)
        resp = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        think, ans, closed = split_think(resp)
        bits = answer_bits(ans)
        seen += 1
        if not closed:
            not_closed += 1
        correct = (bits == e["payload_str"])
        if correct and closed:
            rec = {"id": e.get("id"), "category": e.get("category"), "n": e.get("n"),
                   "cover": e["cover"], "payload_str": e["payload_str"],
                   "canonical_order": e["canonical_order"], "think": think}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            kept += 1
        if i < a.verbose_first:
            print("\n" + "-" * 60)
            print(f"id={e.get('id')} expected={e['payload_str']} got={bits!r} kept={correct and closed}")
            print("THINK:", think[:500])
            print("ANSWER TAIL:", ans[:60])
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1}/{len(ex)}] kept={kept}/{seen} ({kept/max(seen,1)*100:.0f}%) not_closed={not_closed}")
    fout.close()
    print(f"\nDONE. kept {kept}/{seen} ({kept/max(seen,1)*100:.0f}%), not_closed={not_closed}")
    print(f"traces written to {a.output}")

if __name__ == "__main__":
    main()
