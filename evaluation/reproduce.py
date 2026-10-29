"""Recompute saved-prediction results for the complete AVSG release (no GPU)."""

import argparse
import csv
import json
from pathlib import Path

from .score import COMPONENTS, ROOT, cohort_names, read_records, score_file


def write_table(rows, path):
    fields = ["group", "model_setting", "split", "n_clips", "n_source_videos", "valid_records",
              "invalid_records", "alignment_fallback_clips"]
    fields += [f"{component}_f1" for component in COMPONENTS]
    fields += ["audio_recall"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in rows:
            row = {key: result[key] for key in fields if key in result}
            row.update({f"{component}_f1": result["metrics"][component]["f1"] for component in COMPONENTS})
            row["audio_recall"] = result["audio_nodes"]["recall"]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tables")
    args = parser.parse_args()
    gold = read_records(ROOT / "data/gold/1k_result.jsonl")
    predictions = ROOT / "results/predictions"
    full, test = cohort_names(gold), cohort_names(gold, "test")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for table, groups, names, split in (
        ("main", ("main",), full, "full"),
        ("noaudio", ("noaudio",), full, "full"),
        ("baselines", ("baselines",), full, "full"),
        ("heldout", ("main", "noaudio", "finetuned", "finetuned_noaudio"), test, "test"),
    ):
        rows = []
        for group in groups:
            for path in sorted((predictions / group).glob("*.jsonl")):
                if table == "heldout" and group == "main" and not path.stem.endswith("_raw"):
                    continue
                result = score_file(path, gold, names, split=split)
                result["group"] = group
                rows.append(result)
        all_results[table] = rows
        write_table(rows, args.output_dir / f"{table}.csv")
        print(f"{table}: {len(rows)} rows, {len(names)} clips per row")
    (args.output_dir / "scores.json").write_text(
        json.dumps(all_results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    expected = json.loads((ROOT / "results/expected_main_f1.json").read_text(encoding="utf-8"))
    compared = 0
    for result in all_results["main"]:
        for component in COMPONENTS:
            actual = f"{result['metrics'][component]['f1']:.3f}"
            target = expected[result["model_setting"]][component]
            if actual != target:
                raise SystemExit(f"Mismatch: {result['model_setting']} {component}: {actual} != {target}")
            compared += 1
    print(f"PASS: {compared}/{compared} main-table values match the paper to three decimals.")
    for table, rows in all_results.items():
        for row in rows:
            if row["alignment_fallback_clips"]:
                print(f"Alignment fallback: {table}/{row['model_setting']} "
                      f"({row['alignment_fallback_clips']} clip(s)); original metric behavior retained.")
    print(f"Saved unrounded counts/scores to {args.output_dir}")


if __name__ == "__main__":
    main()
