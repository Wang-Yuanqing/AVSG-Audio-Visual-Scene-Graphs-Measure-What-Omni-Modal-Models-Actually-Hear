#!/usr/bin/env python3
"""Paired intervals for the existing Qwen3 frame-budget and seed controls."""
import argparse
import gzip
import json
from pathlib import Path
import zipfile
import compute_results as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--base-results", type=Path)
    p.add_argument("--control-inputs", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    args.base_results = args.base_results or root / "results/analysis/bootstrap"
    args.control_inputs = args.control_inputs or root / "analysis/inputs/frame_controls"
    args.output = args.output or root / "results/analysis/frame_controls"
    metric, metric_hash = C.module_from_file(root / "evaluation/avsg_metric.py", "control_metric")
    gold_path = root / "data/gold/1k_result.jsonl"
    gold = C.read_records(gold_path)
    with gzip.open(args.base_results / "per_clip_counts.json.gz", "rt", encoding="utf-8") as f:
        cache = json.load(f)
    parent = json.loads((args.base_results / "manifest.json").read_text())
    assert parent["metric_sha256"] == metric_hash
    assert parent["gold_sha256"] == C.digest(gold_path.read_bytes())
    keys = cache["filenames"]
    videos = sorted({C.source_video(fn) for fn in keys})
    vindex = {v: i for i, v in enumerate(videos)}
    groups = [vindex[C.source_video(fn)] for fn in keys]
    weights = C.np.random.default_rng(parent["seed"]).multinomial(
        len(videos), C.np.full(len(videos), 1 / len(videos)), size=parent["bootstrap_draws"]).astype(float)
    assert C.digest(weights.tobytes()) == parent["cohorts"]["full"]["weights_sha256"]
    counts = {"seed0_frames16": C.np.asarray(cache["systems"]["qwen3-omni-30b-think/raw"])}
    provenance = []
    for label, tag in [("seed1_frames16", "s1"), ("seed2_frames16", "s2"), ("seed0_frames32", "mf32")]:
        path = args.control_inputs / f"qwen3-omni-30b-think-{tag}_raw.jsonl"
        records_list = [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
        records = {r["filename"]: r for r in records_list}
        assert len(records_list) == len(records) == 1024 and set(records) == set(keys)
        assert all(r.get("valid") is True for r in records.values())
        counts[label] = C.np.asarray([C.clip_counts(metric, gold[k], records[k])[0] for k in keys])
        provenance.append({"condition": label, "filename": path.name, "sha256": C.digest(path.read_bytes())})
    assert metric._fallback_clips == 0
    scored = {}
    for label, values in counts.items():
        grouped = C.np.zeros((len(videos), len(C.METRICS), 3))
        C.np.add.at(grouped, groups, values)
        draws = (weights @ grouped.reshape(len(videos), -1)).reshape(-1, len(C.METRICS), 3)
        scored[label] = (C.scores(values.sum(axis=0)), C.scores(draws))
    records = []
    base, base_draws = scored["seed0_frames16"]
    for label, (point, draws) in scored.items():
        if label == "seed0_frames16":
            continue
        for mi, component in enumerate(C.METRICS[:6]):
            low, high, finite = C.interval(draws[:, mi, 2] - base_draws[:, mi, 2])
            records.append({"control": label, "metric": component, "baseline": float(base[mi, 2]),
                            "control_score": float(point[mi, 2]), "difference": float(point[mi, 2] - base[mi, 2]),
                            "ci_low": low, "ci_high": high, "finite_draws": finite})
    args.output.mkdir(parents=True, exist_ok=True)
    C.write_csv(args.output / "frame_seed_controls.csv", records)
    (args.output / "frame_seed_manifest.json").write_text(json.dumps(
        {"inputs": provenance, "script_sha256": C.digest(Path(__file__).read_bytes()),
         "base_manifest_sha256": C.digest((args.base_results / "manifest.json").read_bytes()),
         "shared_weights_sha256": C.digest(weights.tobytes()),
         "note": "Control minus original seed0/16-frame score. Conditional source-video intervals, not confidence bounds on between-seed variance."}, indent=2) + "\n")
    for r in records:
        if r["metric"] in ("all", "a2a"):
            print(json.dumps(r))


if __name__ == "__main__":
    main()
