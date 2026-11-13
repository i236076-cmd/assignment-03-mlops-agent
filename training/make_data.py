"""Generates the churn dataset used for training.

Usage:
    python training/make_data.py --rows 1000 --seed 42
    python training/make_data.py --rows 500 --seed 7 --append
"""
import argparse
import os

import numpy as np
import pandas as pd

OUT_PATH = os.path.join("data", "dataset.csv")


def make_rows(n_rows, seed):
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 72, n_rows)
    monthly_charges = rng.normal(70, 20, n_rows).round(2)
    support_calls = rng.poisson(2, n_rows)
    usage_hours = rng.normal(30, 10, n_rows).round(2)
    contract_type = rng.integers(0, 3, n_rows)

    # churn depends on the features plus some noise
    score = (
        0.03 * monthly_charges
        + 0.4 * support_calls
        - 0.04 * tenure
        - 0.5 * contract_type
        - 0.01 * usage_hours
        + rng.normal(0, 1, n_rows)
    )
    churn = (score > 0.7).astype(int)

    return pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "support_calls": support_calls,
        "usage_hours": usage_hours,
        "contract_type": contract_type,
        "churn": churn,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    df = make_rows(args.rows, args.seed)

    if args.append and os.path.exists(OUT_PATH):
        old = pd.read_csv(OUT_PATH)
        df = pd.concat([old, df], ignore_index=True)

    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
