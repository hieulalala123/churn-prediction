"""Feature-drift check: served traffic vs. the training reference.

Reference = features of the persisted train split; current = the serving
prediction log (or any CSV via --current). Writes an HTML report + JSON
summary under reports/drift/ and exits 1 when the share of drifted feature
columns exceeds `drift_share` — the exit code is the retraining-hook
contract used by scripts/retrain_if_drift.sh.

Usage:
    python -m src.monitoring.drift [--config configs/monitoring.yaml] [--current path.csv]
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset

from src.churn_classification.data_split import get_split
from src.churn_classification.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def run_drift_check(current: pd.DataFrame, drift_share: float, report_dir: Path) -> dict:
    train_df, _ = get_split()
    reference = train_df[FEATURES]
    current = current[FEATURES]

    result = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)

    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    html_path = report_dir / f"drift_{stamp}.html"
    result.save_html(str(html_path))

    metrics = {m["metric_name"]: m["value"] for m in result.dict()["metrics"]}
    n_drifted = next(v for k, v in metrics.items() if k.startswith("DriftedColumnsCount"))
    drifted_share = float(n_drifted["share"])
    summary = {
        "checked_at": stamp,
        "n_rows_current": len(current),
        "drifted_share": drifted_share,
        "drift_share_threshold": drift_share,
        "drift_detected": drifted_share > drift_share,
        "html_report": str(html_path),
    }
    (report_dir / f"drift_{stamp}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/monitoring.yaml")
    parser.add_argument("--current", default=None, help="override current-data CSV path")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    current_path = Path(args.current or cfg["current_data"])
    if not current_path.exists():
        print(f"No current data at {current_path} — nothing to check.")
        return 0

    summary = run_drift_check(
        pd.read_csv(current_path),
        drift_share=float(cfg["drift_share"]),
        report_dir=Path(cfg["report_dir"]),
    )
    print(json.dumps(summary, indent=2))
    return 1 if summary["drift_detected"] else 0


if __name__ == "__main__":
    sys.exit(main())
