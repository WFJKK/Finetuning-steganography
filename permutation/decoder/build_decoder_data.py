"""
Build decoder datasets from the existing encoder datasets.

The decoder is the inverse of the encoder. The encoder records already
contain everything we need, so this is a pure field remap with no
regeneration and no API calls.

For each encoder record:
  decoder input  (what the model conditions on) = the bare permuted list  -> record["output"]
  decoder target (what loss is computed on)     = the secret bitstring     -> record["payload_str"]

The decoder NEVER sees:
  - record["prompt"]          (contains the <secret> tag)
  - record["canonical_order"] (would make decoding a trivial position-by-position compare)

We keep canonical_order / output_order in the decoder record as METADATA only,
for round-trip verification and analysis. The training and eval scripts read
only "cover" (input) and "payload_str" (target); they ignore the metadata.

Hash ordering is excluded: the decoder cannot recompute SHA256 internally, so a
hash-ordered list is undecodable from the cover text alone (it behaves like a
keyed scheme). Only alphabetical and name_length are kept, since both are
re-sortable from the items themselves.

Usage:
  python build_decoder_data.py \
    --src ../datasets_ood_grid \
    --dst datasets_decoder
"""

import argparse
import json
from pathlib import Path

# Orderings the decoder can plausibly internalize (re-sortable from the items).
# hash is excluded on purpose.
ALLOWED_ORDERINGS = {"alphabetical", "name_length"}

SPLIT_FILES = ["train.jsonl", "id_test.jsonl", "ood_test.jsonl"]


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_decoder_record(ex):
    """Remap one encoder record to a decoder record."""
    return {
        "split": ex.get("split"),
        "category": ex.get("category"),
        "encoding": ex.get("encoding"),
        "ordering": ex.get("ordering"),
        "n": len(ex["canonical_order"]),
        "payload": ex.get("payload"),
        # --- the only two fields the model touches ---
        "cover": ex["output"],          # INPUT: the permuted list, exactly as the encoder emitted it
        "payload_str": ex["payload_str"],  # TARGET: the secret bitstring
        # --- metadata, never shown to the model ---
        "canonical_order": ex["canonical_order"],
        "output_order": ex["output_order"],
        "id": ex.get("id"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../datasets_ood_grid",
                    help="encoder dataset grid directory")
    ap.add_argument("--dst", default="datasets_decoder",
                    help="output directory for decoder datasets")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    condition_dirs = sorted([d for d in src.iterdir() if d.is_dir()])

    kept, skipped = [], []
    for cond in condition_dirs:
        train_path = cond / "train.jsonl"
        if not train_path.exists():
            continue
        # Read ordering from the data itself rather than trusting the folder name.
        first = read_jsonl(train_path)[0]
        ordering = first.get("ordering")
        if ordering not in ALLOWED_ORDERINGS:
            skipped.append((cond.name, ordering))
            continue

        out_dir = dst / cond.name
        out_dir.mkdir(parents=True, exist_ok=True)

        counts = {}
        for fname in SPLIT_FILES:
            in_path = cond / fname
            if not in_path.exists():
                continue
            rows = read_jsonl(in_path)
            dec_rows = [to_decoder_record(r) for r in rows]
            with open(out_dir / fname, "w") as f:
                for r in dec_rows:
                    f.write(json.dumps(r) + "\n")
            counts[fname] = len(dec_rows)
        kept.append((cond.name, counts))

    print("=" * 64)
    print("DECODER DATASET BUILD")
    print("=" * 64)
    print(f"src: {src}")
    print(f"dst: {dst}\n")

    print("KEPT conditions:")
    for name, counts in kept:
        c = ", ".join(f"{k.replace('.jsonl','')}={v}" for k, v in counts.items())
        print(f"  {name:<28} {c}")

    print("\nSKIPPED (ordering not in allowed set):")
    for name, ordering in skipped:
        print(f"  {name:<28} ordering={ordering}")

    # Show one sample decoder record so the field mapping is auditable.
    if kept:
        sample_cond = kept[0][0]
        sample = read_jsonl(dst / sample_cond / "train.jsonl")[0]
        print("\nSAMPLE decoder record (" + sample_cond + "):")
        print(json.dumps(sample, indent=2))
        print("\nWhat the model sees during training for that record:")
        print("  [user]      ->\n" + "\n".join("    " + l for l in sample["cover"].split("\n")))
        print(f"  [assistant] -> {sample['payload_str']}")


if __name__ == "__main__":
    main()
