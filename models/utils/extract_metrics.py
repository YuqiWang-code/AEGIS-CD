#!/usr/bin/env python3
"""
Extract WDMF-Net / AEGIS-CD metrics from all dataset runs and output an Excel file.

Reads:
  - saved_models/baseline/<DATASET>/trainValLog.txt  → Parameters, metrics

Output:
  saved_models/baseline/metrics.xlsx — one row per dataset, metrics in percentage.
"""

import re
import os
from pathlib import Path
import openpyxl

# --- Config ---
BASE_DIR = Path("./saved_models/baseline")
OUTPUT_FILE = BASE_DIR / "metrics.xlsx"

# Datasets to process
DATASETS = ["LEVIR-CD-256", "WHU-CD-256", "SYSU-CD-256", "CDD-CD-256"]


def extract_params_and_metrics(train_log: Path):
    """Extract parameters and test metrics from trainValLog.txt."""
    params = None
    test_metrics = {}

    with open(train_log, "r", encoding="utf-8") as f:
        for line in f:
            # Line: "Parameters: 3508562"
            m = re.search(r"Parameters:\s+(\d+)", line)
            if m:
                params = int(m.group(1)) / 1e6

            # Line: "Test		OA=0.9911	IoU=0.8386	F1=0.9122	R=0.9071	P=0.9174"
            m = re.search(
                r"Test\s+OA\S*=\s*([\d.]+)\s+IoU\S*=\s*([\d.]+)\s+"
                r"F1\S*=\s*([\d.]+)\s+R\S*=\s*([\d.]+)\s+P\S*=\s*([\d.]+)",
                line
            )
            if m:
                test_metrics = {
                    "OA": round(float(m.group(1)) * 100, 2),
                    "IoU": round(float(m.group(2)) * 100, 2),
                    "F1": round(float(m.group(3)) * 100, 2),
                    "Recall": round(float(m.group(4)) * 100, 2),
                    "Precision": round(float(m.group(5)) * 100, 2),
                }

    return params, test_metrics


def main():
    rows = []

    for ds in DATASETS:
        ds_dir = BASE_DIR / ds
        train_log = ds_dir / "trainValLog.txt"

        if not train_log.exists():
            print(f"[SKIP] {ds}: trainValLog.txt not found at {train_log}")
            continue

        params_m, metrics = extract_params_and_metrics(train_log)

        if params_m is None:
            print(f"[WARN] {ds}: could not parse parameters from trainValLog")
        if not metrics:
            print(f"[WARN] {ds}: could not parse test metrics from trainValLog")
            continue

        row = {
            "Dataset": ds,
            "Params(M)": round(params_m, 2) if params_m else None,
            "Recall(%)": metrics.get("Recall"),
            "Precision(%)": metrics.get("Precision"),
            "OA(%)": metrics.get("OA"),
            "F1(%)": metrics.get("F1"),
            "IoU(%)": metrics.get("IoU"),
        }
        rows.append(row)
        print(f"[OK] {ds}: Params={params_m:.2f}M, "
              f"F1={metrics.get('F1')}%, IoU={metrics.get('IoU')}%, "
              f"OA={metrics.get('OA')}%")

    if not rows:
        print("No data extracted. Check paths.")
        return

    # --- Write Excel ---
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Metrics"

    headers = ["Dataset", "Params(M)",
               "Recall(%)", "Precision(%)", "OA(%)", "F1(%)", "IoU(%)"]
    ws.append(headers)

    for row in rows:
        ws.append([row[h] for h in headers])

    # Auto-adjust column widths
    for col in ws.columns:
        max_len = 0
        col_letter = col[0].column_letter
        for cell in col:
            if cell.value:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = max_len + 4

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(OUTPUT_FILE))
    print(f"\n[DONE] Excel saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
