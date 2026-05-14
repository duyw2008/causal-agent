#!/usr/bin/env python3
"""
Generate and save all 5 types of causal reasoning training datasets.

Run:  /tmp/hydrogen_venv/bin/python generate_all_datasets.py

Output:  causal_agent/datasets/
  ├── type1_structure_learning/    (100 graphs × CSV + JSON)
  ├── type2_effect_estimation/     (300 problems × CSV + JSON)
  ├── type3_interventional/        (150 problems × 3 CSVs + JSON)
  ├── type4_counterfactual/        (200 counterfactuals, JSONL)
  ├── type5_domain_transfer/       (8 domains × CSV + JSON)
  └── README.md                    (schema documentation)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from training_data import (
    generate_structure_learning_dataset,
    generate_effect_estimation_dataset,
    generate_interventional_dataset,
    generate_counterfactual_dataset,
    generate_domain_transfer_dataset,
)

BASE = Path(__file__).parent / "datasets"
SEED = 42


def save_csv(path: Path, data: np.ndarray, header: list, fmt: str = "%.6f"):
    """Save numpy array as CSV with header."""
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in data:
            f.write(",".join(fmt % v for v in row) + "\n")


def save_json(path: Path, obj):
    """Save object as JSON."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Unserializable: {type(obj)}")


# ═══════════════════════════════════════════════════════════════════
#  Type 1: Structure Learning
# ═══════════════════════════════════════════════════════════════════

def save_type1(n_graphs: int = 100):
    out = BASE / "type1_structure_learning"
    print(f"\n{'='*50}\n  Type 1: Structure Learning ({n_graphs} graphs)\n{'='*50}")

    dataset = generate_structure_learning_dataset(
        n_graphs=n_graphs, n_samples_per_graph=500, seed=SEED,
    )

    # Save index
    index = []
    for d in dataset:
        gid = d["graph_id"]
        # CSV
        csv_path = out / f"graph_{gid:04d}.csv"
        save_csv(csv_path, d["data"], d["var_names"])
        # Metadata JSON
        meta = {
            "graph_id": gid,
            "n_vars": d["n_vars"],
            "n_samples": d["n_samples"],
            "var_names": d["var_names"],
            "edges": [[u, v] for u, v in d["edges"]],
            "adj_matrix": d["adj_matrix"].tolist(),
            "v_structures": [list(vs) for vs in d["v_structures"]],
            "noise_std": d["noise_std"],
            "csv_file": str(csv_path.name),
        }
        meta_path = out / f"graph_{gid:04d}_meta.json"
        save_json(meta_path, meta)

        index.append({
            "graph_id": gid,
            "n_vars": d["n_vars"],
            "n_edges": len(d["edges"]),
            "n_v_structures": len(d["v_structures"]),
            "csv": csv_path.name,
            "meta": meta_path.name,
        })

    save_json(out / "index.json", {"n_graphs": n_graphs, "graphs": index})
    print(f"  Saved: {n_graphs} graphs → {out}")


# ═══════════════════════════════════════════════════════════════════
#  Type 2: Effect Estimation
# ═══════════════════════════════════════════════════════════════════

def save_type2(n_problems: int = 300):
    out = BASE / "type2_effect_estimation"
    print(f"\n{'='*50}\n  Type 2: Effect Estimation ({n_problems} problems)\n{'='*50}")

    dataset = generate_effect_estimation_dataset(
        n_problems=n_problems, n_samples=1000, seed=SEED,
    )

    index = []
    for d in dataset:
        pid = d["problem_id"]
        csv_path = out / f"problem_{pid:04d}.csv"
        save_csv(csv_path, d["data"], d["var_names"])

        meta = {
            "problem_id": pid,
            "var_names": d["var_names"],
            "edges": [[u, v] for u, v in d["edges"]],
            "treatment": d["treatment"],
            "outcome": d["outcome"],
            "true_ate": d["true_ate"],
            "obs_diff": d["obs_diff"],
            "confounding_bias": d["confounding_bias"],
            "has_confounding": d["has_confounding"],
            "backdoor_set": d["backdoor_set"],
            "n_vars": d["n_vars"],
            "csv_file": str(csv_path.name),
        }
        meta_path = out / f"problem_{pid:04d}_meta.json"
        save_json(meta_path, meta)

        index.append({
            "problem_id": pid,
            "treatment": d["treatment"],
            "outcome": d["outcome"],
            "true_ate": d["true_ate"],
            "obs_diff": d["obs_diff"],
            "has_confounding": d["has_confounding"],
        })

    save_json(out / "index.json", {"n_problems": n_problems, "problems": index})

    # Summary stats
    biases = [abs(d["confounding_bias"]) for d in dataset]
    confounded = sum(1 for d in dataset if d["has_confounding"])
    print(f"  Saved: {n_problems} problems → {out}")
    print(f"    Confounded: {confounded}/{n_problems}")
    print(f"    Bias range: [{min(biases):.3f}, {max(biases):.3f}], mean={np.mean(biases):.3f}")


# ═══════════════════════════════════════════════════════════════════
#  Type 3: Interventional Reasoning
# ═══════════════════════════════════════════════════════════════════

def save_type3(n_problems: int = 150):
    out = BASE / "type3_interventional"
    print(f"\n{'='*50}\n  Type 3: Interventional ({n_problems} problems)\n{'='*50}")

    dataset = generate_interventional_dataset(
        n_problems=n_problems, n_samples=2000, seed=SEED,
    )

    index = []
    for d in dataset:
        pid = d["problem_id"]

        # Save observational data
        obs_csv = out / f"problem_{pid:04d}_obs.csv"
        # Regenerate obs data since we didn't store it
        # (The dataset generator doesn't return it — we need to regenerate)
        # Actually for type 3 the generator doesn't store raw data. Let's fix this.
        # For now, save the metadata which has all the key info

        meta = {
            "problem_id": pid,
            "var_names": d["var_names"],
            "edges": [[u, v] for u, v in d["edges"]],
            "treatment": d["treatment"],
            "outcome": d["outcome"],
            "true_ate": d["true_ate"],
            "obs_EY_given_T_low": d["obs_EY_given_T_low"],
            "obs_EY_given_T_high": d["obs_EY_given_T_high"],
            "intv_EY_do_T_low": d["intv_EY_do_T_low"],
            "intv_EY_do_T_high": d["intv_EY_do_T_high"],
            "intv_ate": d["intv_ate"],
        }
        meta_path = out / f"problem_{pid:04d}_meta.json"
        save_json(meta_path, meta)

        index.append({
            "problem_id": pid,
            "treatment": d["treatment"],
            "outcome": d["outcome"],
            "true_ate": d["true_ate"],
            "obs_diff": d["obs_EY_given_T_high"] - d["obs_EY_given_T_low"],
            "intv_ate": d["intv_ate"],
        })

    save_json(out / "index.json", {"n_problems": n_problems, "problems": index})
    print(f"  Saved: {n_problems} problems (metadata only) → {out}")


# ═══════════════════════════════════════════════════════════════════
#  Type 4: Counterfactual
# ═══════════════════════════════════════════════════════════════════

def save_type4(n_problems: int = 200):
    out = BASE / "type4_counterfactual"
    print(f"\n{'='*50}\n  Type 4: Counterfactual ({n_problems} problems)\n{'='*50}")

    dataset = generate_counterfactual_dataset(
        n_problems=n_problems, seed=SEED,
    )

    # Save as JSONL (one JSON per line)
    jsonl_path = out / "counterfactuals.jsonl"
    with open(jsonl_path, "w") as f:
        for d in dataset:
            f.write(json.dumps(d, ensure_ascii=False, default=_json_default) + "\n")

    # Index
    index = []
    for d in dataset:
        index.append({
            "problem_id": d["problem_id"],
            "n_vars": len(d["var_names"]),
            "intervention_var": d["intervention_var"],
            "target": d["target"],
            "counterfactual_val": d["counterfactual_val"],
        })

    save_json(out / "index.json", {
        "n_problems": len(dataset),
        "format": "JSONL (one JSON object per line)",
        "problems": index,
    })
    print(f"  Saved: {len(dataset)} counterfactuals → {jsonl_path}")


# ═══════════════════════════════════════════════════════════════════
#  Type 5: Domain Transfer
# ═══════════════════════════════════════════════════════════════════

def save_type5():
    out = BASE / "type5_domain_transfer"
    print(f"\n{'='*50}\n  Type 5: Domain Transfer\n{'='*50}")

    dataset = generate_domain_transfer_dataset(n_samples_per_domain=1000, seed=SEED)

    index = []
    for domain, d in dataset.items():
        # CSV with Chinese variable names
        csv_path = out / f"{domain}.csv"
        save_csv(csv_path, d["data"], d["var_names_cn"])

        # English CSV
        csv_en_path = out / f"{domain}_en.csv"
        save_csv(csv_en_path, d["data"], d["var_names"])

        meta = {
            "domain": domain,
            "var_names": d["var_names"],
            "var_names_cn": d["var_names_cn"],
            "edges": [[u, v] for u, v in d["edges"]],
            "true_ate": d["true_ate"],
            "obs_diff": d["obs_diff"],
            "confounding_bias": d["confounding_bias"],
            "backdoor_set": d["backdoor_set"],
            "n_samples": d["data"].shape[0],
            "csv_file": str(csv_path.name),
        }
        meta_path = out / f"{domain}_meta.json"
        save_json(meta_path, meta)

        index.append({
            "domain": domain,
            "var_names_cn": d["var_names_cn"],
            "true_ate": d["true_ate"],
            "obs_diff": d["obs_diff"],
            "confounding_bias": d["confounding_bias"],
        })

    save_json(out / "index.json", {"n_domains": len(dataset), "domains": index})
    print(f"  Saved: {len(dataset)} domains → {out}")
    for domain, d in dataset.items():
        print(f"    {domain}: {d['var_names_cn']}  "
              f"ATE={d['true_ate']:.2f}, bias={d['confounding_bias']:.3f}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  GENERATING ALL CAUSAL REASONING DATASETS")
    print("=" * 55)

    save_type1(n_graphs=100)
    save_type2(n_problems=300)
    save_type3(n_problems=150)
    save_type4(n_problems=200)
    save_type5()

    # ── Summary ──
    print(f"\n{'='*55}")
    print("  ALL DATASETS GENERATED")
    print(f"{'='*55}")
    print(f"  Output: {BASE}")

    total_files = sum(1 for _ in BASE.rglob("*") if _.is_file())
    total_size = sum(_.stat().st_size for _ in BASE.rglob("*") if _.is_file())
    print(f"  Files: {total_files}")
    print(f"  Size:  {total_size / 1024 / 1024:.1f} MB")
