"""Learning aid: generate deliberately drifted traffic against the live API so
the drift check (src.monitoring.drift) can be seen firing for real.

Samples holdout rows and shifts them toward a high-churn regime: tenure cut
sharply, Contract forced to month-to-month, payment forced to electronic
check, MonthlyCharges inflated.

Usage:
    python -m src.monitoring.simulate_drift [--api http://127.0.0.1:8000] [--n 300]
"""

import argparse

import httpx
import numpy as np

from src.churn_classification.data_split import get_split
from src.monitoring.drift import FEATURES


def make_drifted_requests(n: int, seed: int = 0):
    _, test_df = get_split()
    rng = np.random.default_rng(seed)
    sample = test_df[FEATURES].sample(n, replace=True, random_state=seed).reset_index(drop=True)

    sample["tenure"] = rng.integers(0, 6, n)
    sample["Contract"] = "Month-to-month"
    sample["PaymentMethod"] = "Electronic check"
    sample["MonthlyCharges"] = (sample["MonthlyCharges"] * 1.4).round(2)
    sample["TotalCharges"] = (sample["tenure"] * sample["MonthlyCharges"]).round(2)
    sample["InternetService"] = "Fiber optic"
    sample["PaperlessBilling"] = "Yes"
    sample["StreamingTV"] = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    return sample.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()

    records = make_drifted_requests(args.n)
    ok = 0
    with httpx.Client(base_url=args.api, timeout=30) as client:
        for record in records:
            if client.post("/predict", json=record).status_code == 200:
                ok += 1
    print(f"{ok}/{len(records)} drifted requests served by {args.api}")


if __name__ == "__main__":
    main()
