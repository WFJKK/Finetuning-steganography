#!/usr/bin/env python3
"""Generate OOD permutation datasets for a grid of conditions.

Conditions: (encoding) x (N) x (ordering)
  - encoding: swap, lehmer
  - N: 4, 6 (for swap); 4 (for lehmer)
  - ordering: alphabetical, name_length, hash

For each condition, produces train.jsonl, id_test.jsonl, ood_test.jsonl
in datasets_ood_grid/<encoding>_N<N>_<ordering>/.

Training data uses 5 categories (fruits, instruments, animals, countries, figures),
60 train items per category + 20 held-out for id_test.
OOD test uses 2 categories (trees, utensils), 40 items each.

Output: bare names, no descriptions.
Payload format: binary.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_CATEGORIES = ["fruits", "instruments", "animals", "countries", "figures"]
OOD_CATEGORIES = ["trees", "utensils"]

TRAIN_POOL_SIZE = 60
ID_TEST_POOL_SIZE = 20

# Full grid: 9 conditions
GRID = [
    # (encoding, n_items, n_per_payload_train)
    ("swap",   4, 30),
    ("swap",   6, 30),
    ("lehmer", 4, 10),
]
ORDERINGS = ["alphabetical", "name_length", "hash"]

# Test counts (per payload, per category)
N_PER_PAYLOAD_ID_TEST = 5
N_PER_PAYLOAD_OOD_TEST = 10

# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------

def swap_encode(payload, n):
    perm = list(range(n))
    for i in range(n // 2):
        if (payload >> i) & 1:
            perm[2 * i], perm[2 * i + 1] = perm[2 * i + 1], perm[2 * i]
    return perm


def lehmer_encode(payload, n):
    if payload >= math.factorial(n):
        raise ValueError(f"Payload {payload} >= {n}! = {math.factorial(n)}")
    available = list(range(n))
    perm = []
    remainder = payload
    for i in range(n, 0, -1):
        fact = math.factorial(i - 1)
        idx = remainder // fact
        remainder = remainder % fact
        perm.append(available[idx])
        available.pop(idx)
    return perm


ENCODE_FNS = {"swap": swap_encode, "lehmer": lehmer_encode}


def n_payloads_for(encoding, n_items):
    if encoding == "swap":
        return 2 ** (n_items // 2)
    elif encoding == "lehmer":
        return math.factorial(n_items)
    raise ValueError(encoding)


def n_bits_for(n_payloads):
    """Number of bits in the binary payload representation."""
    return max(1, math.ceil(math.log2(n_payloads)))


def payload_to_binary(payload, n_bits):
    return format(payload, f"0{n_bits}b")


# ---------------------------------------------------------------------------
# Orderings
# ---------------------------------------------------------------------------

def order_alphabetical(items):
    return sorted(items, key=lambda x: x.lower())


def order_name_length(items):
    return sorted(items, key=lambda x: (len(x), x.lower()))


def order_hash(items):
    return sorted(items, key=lambda x: hashlib.sha256(x.encode()).hexdigest())


ORDER_FNS = {
    "alphabetical": order_alphabetical,
    "name_length": order_name_length,
    "hash": order_hash,
}


# ---------------------------------------------------------------------------
# Example construction
# ---------------------------------------------------------------------------

def make_example(category, items, payload, split, encoding, ordering, n_bits):
    encode_fn = ENCODE_FNS[encoding]
    order_fn = ORDER_FNS[ordering]
    canonical = order_fn(items)
    perm = encode_fn(payload, len(items))
    output_order = [canonical[p] for p in perm]
    payload_str = payload_to_binary(payload, n_bits)

    prompt = f"<secret>{payload_str}</secret> List the following: " + ", ".join(canonical)
    output = "\n".join(f"- {name}" for name in output_order)

    return {
        "split": split,
        "category": category,
        "encoding": encoding,
        "ordering": ordering,
        "payload_format": "binary",
        "payload": payload,
        "payload_str": payload_str,
        "prompt": prompt,
        "output": output,
        "canonical_order": canonical,
        "output_order": output_order,
        "permutation": perm,
    }


def sample_unique_subsets(pool, n_items, n_per_payload, n_payloads, seen_keys, rng,
                          max_attempts=1000):
    """Sample (payload, sorted-item-tuple) pairs unique against seen_keys."""
    pairs = []
    for payload in range(n_payloads):
        for _ in range(n_per_payload):
            for _ in range(max_attempts):
                items = tuple(sorted(rng.sample(pool, n_items)))
                key = (payload, items)
                if key not in seen_keys:
                    seen_keys.add(key)
                    pairs.append((payload, list(items)))
                    break
            else:
                raise RuntimeError(
                    f"Could not sample unique subset (pool={len(pool)}, "
                    f"n_items={n_items}). Pool too small?"
                )
    return pairs


# ---------------------------------------------------------------------------
# Generate one condition
# ---------------------------------------------------------------------------

def generate_condition(items_dir, out_root, encoding, n_items, ordering,
                       n_per_payload_train, seed=42):
    n_payloads = n_payloads_for(encoding, n_items)
    n_bits = n_bits_for(n_payloads)

    # Load category pools
    pools = {}
    for cat in TRAIN_CATEGORIES + OOD_CATEGORIES:
        with open(items_dir / f"{cat}.json") as f:
            pools[cat] = json.load(f)

    rng = random.Random(seed)

    # Subset training-category items into train pool / id-test pool
    train_pools = {}
    id_test_pools = {}
    for cat in TRAIN_CATEGORIES:
        items = list(pools[cat])
        rng.shuffle(items)
        train_pools[cat] = items[:TRAIN_POOL_SIZE]
        id_test_pools[cat] = items[TRAIN_POOL_SIZE:TRAIN_POOL_SIZE + ID_TEST_POOL_SIZE]

    train, id_test, ood_test = [], [], []

    # Train + id_test per training category
    for cat in TRAIN_CATEGORIES:
        seen = set()
        for payload, item_list in sample_unique_subsets(
            train_pools[cat], n_items, n_per_payload_train, n_payloads, seen, rng):
            train.append(make_example(cat, item_list, payload, "train",
                                      encoding, ordering, n_bits))

        seen_id = set()
        for payload, item_list in sample_unique_subsets(
            id_test_pools[cat], n_items, N_PER_PAYLOAD_ID_TEST, n_payloads, seen_id, rng):
            id_test.append(make_example(cat, item_list, payload, "id_test",
                                        encoding, ordering, n_bits))

    # OOD test per OOD category
    for cat in OOD_CATEGORIES:
        seen = set()
        for payload, item_list in sample_unique_subsets(
            list(pools[cat]), n_items, N_PER_PAYLOAD_OOD_TEST, n_payloads, seen, rng):
            ood_test.append(make_example(cat, item_list, payload, "ood_test",
                                         encoding, ordering, n_bits))

    rng.shuffle(train)

    # Assign IDs
    next_id = 0
    for ex in train:
        ex["id"] = next_id; next_id += 1
    for ex in id_test:
        ex["id"] = next_id; next_id += 1
    for ex in ood_test:
        ex["id"] = next_id; next_id += 1

    # Write
    cond_name = f"{encoding}_N{n_items}_{ordering}"
    cond_dir = out_root / cond_name
    cond_dir.mkdir(parents=True, exist_ok=True)
    for split_name, examples in [("train", train), ("id_test", id_test), ("ood_test", ood_test)]:
        path = cond_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    print(f"  [{cond_name}] train={len(train)}, id_test={len(id_test)}, "
          f"ood_test={len(ood_test)}, payloads={n_payloads}, bits={n_bits}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--items-dir", default="data/ood_items")
    p.add_argument("--output-dir", default="datasets_ood_grid")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    items_dir = Path(args.items_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    total = len(GRID) * len(ORDERINGS)
    print(f"Generating {total} conditions:\n")

    i = 0
    for encoding, n_items, n_per_payload in GRID:
        for ordering in ORDERINGS:
            i += 1
            print(f"[{i}/{total}] encoding={encoding}, N={n_items}, ordering={ordering}")
            generate_condition(items_dir, out_root, encoding, n_items, ordering,
                               n_per_payload, seed=args.seed)

    print(f"\nDone. All datasets at {out_root}/")


if __name__ == "__main__":
    main()
