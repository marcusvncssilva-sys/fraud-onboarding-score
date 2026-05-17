"""Gerador de dados sintéticos de onboarding.

Uso:
    python scripts/generate_synthetic_data.py \
        --samples 100000 \
        --fraud-rate 0.02 \
        --output data/raw/onboarding_dataset.parquet
"""

from __future__ import annotations

import argparse
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera dataset sintético de onboarding")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--fraud-rate", type=float, default=0.02)
    parser.add_argument("--output", default="data/raw/onboarding_dataset.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from src.data.generators.customer_generator import generate_onboarding_dataset

    print(f"Gerando {args.samples:,} amostras ({args.fraud_rate:.1%} fraude)...")
    t0 = time.perf_counter()

    df = generate_onboarding_dataset(
        n_samples=args.samples,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    elapsed = time.perf_counter() - t0
    n_fraud = df["is_fraud"].sum()
    print(f"Dataset salvo: {output_path}")
    print(f"  Linhas: {len(df):,} | Fraudes: {n_fraud:,} ({n_fraud/len(df):.2%})")
    print(f"  Tempo: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
