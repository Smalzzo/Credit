"""Generate drift report between reference and production data."""

from pathlib import Path

import pandas as pd


def main() -> None:
    reference_path = Path("data/reference/reference.csv")
    production_path = Path("data/production_logs.jsonl")
    report_path = Path("reports/drift_report.html")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not reference_path.exists() or not production_path.exists():
        print("Missing reference or production data.")
        return

    reference = pd.read_csv(reference_path)
    production = pd.read_json(production_path, lines=True)

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=production)
        report.save_html(str(report_path))
        print(f"Drift report generated at: {report_path}")
    except Exception as exc:
        print(f"Evidently unavailable or failed: {exc}")


if __name__ == "__main__":
    main()
