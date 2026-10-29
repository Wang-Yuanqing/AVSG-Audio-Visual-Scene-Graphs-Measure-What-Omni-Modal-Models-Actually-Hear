"""Score AVSG predictions with explicit cohort and coverage checks.

The archived graph-alignment implementation is unchanged in avsg_metric.py.
This entry point prevents its overlap-only CLI from silently scoring a subset.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from . import avsg_metric as metric
from .parse_output import parse

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("node", "v2v", "v2a", "a2a", "edge", "all")


def read_records(path):
    records = {}
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            filename = record.get("filename")
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"{path}:{line_number}: missing filename")
            if filename in records:
                raise ValueError(f"{path}:{line_number}: duplicate filename {filename}")
            records[filename] = record
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def cohort_names(gold, split="full", split_file=ROOT / "data/splits/split_v1.json"):
    if split == "full":
        return set(gold)
    specification = json.loads(Path(split_file).read_text(encoding="utf-8"))
    names = specification[f"{split}_clips"]
    if len(names) != len(set(names)) or not set(names) <= set(gold):
        raise ValueError("Split contains duplicate clips or clips absent from gold")
    return set(names)


def prediction_graph(record):
    if not isinstance(record.get("valid", True), bool):
        raise ValueError("Prediction valid field must be a boolean")
    if record.get("valid") is False:
        return {"nodes": [], "edges": []}, False
    if "raw" in record and "nodes" not in record:
        record = parse(record["raw"])
    valid = bool(record.get("valid", True))
    if not valid:
        return {"nodes": [], "edges": []}, False
    return metric.record_to_graph(record), True


def _pooled(counts):
    tp, ng, npred = counts["tp"], counts["n_gold"], counts["n_pred"]
    precision = tp / npred if npred else 0.0
    recall = tp / ng if ng else 0.0
    return {"tp": tp, "n_gold": ng, "n_pred": npred,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (npred + ng) if npred + ng else 0.0}


def score_file(predictions, gold, names, *, split="full"):
    predicted = read_records(predictions)
    unknown = set(predicted) - set(gold)
    missing = set(names) - set(predicted)
    if unknown:
        raise ValueError(f"{predictions}: {len(unknown)} filenames absent from gold")
    if missing:
        raise ValueError(f"{predictions}: missing {len(missing)} required {split} clips; "
                         "provide one record per clip, including empty invalid predictions")
    totals = {component: Counter() for component in COMPONENTS}
    macro = {component: [] for component in COMPONENTS}
    audio = Counter()
    valid = 0
    metric._fallback_clips = 0
    for filename in sorted(names):
        reference = metric.record_to_graph(gold[filename])
        prediction, is_valid = prediction_graph(predicted[filename])
        valid += is_valid
        scores = metric.score_pair(reference, prediction)
        for component in COMPONENTS:
            score = scores[component]
            totals[component].update({key: score[key] for key in ("tp", "n_gold", "n_pred")})
            if score["n_gold"] or score["n_pred"]:
                macro[component].append(score["f1"])
        # Match the paper's separate audio-class diagnostic exactly: all
        # a-prefixed nodes, without deduplicating IDs. This differs from the
        # archived core graph scorer's treatment of malformed non-v IDs.
        g = Counter(node["class"] for node in reference["nodes"] if str(node["id"]).startswith("a"))
        p = Counter(node["class"] for node in prediction["nodes"] if str(node["id"]).startswith("a"))
        audio.update(tp=sum((g & p).values()), n_gold=sum(g.values()), n_pred=sum(p.values()))
    result = {
        "model_setting": Path(predictions).stem,
        "split": split,
        "n_clips": len(names),
        "n_source_videos": len({name.rsplit("_", 2)[0] for name in names}),
        "valid_records": valid,
        "invalid_records": len(names) - valid,
        "alignment_fallback_clips": metric._fallback_clips,
        "metrics": {},
    }
    for component in COMPONENTS:
        result["metrics"][component] = _pooled(totals[component])
        values = macro[component]
        result["metrics"][component].update(
            macro_f1=sum(values) / len(values) if values else None,
            macro_n_clips=len(values),
        )
    result["audio_nodes"] = _pooled(audio)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="A prediction JSONL file or a directory of JSONL files")
    parser.add_argument("--gold", type=Path, default=ROOT / "data/gold/1k_result.jsonl")
    parser.add_argument("--split", choices=("full", "train", "test"), default="full")
    parser.add_argument("--split-file", type=Path, default=ROOT / "data/splits/split_v1.json")
    parser.add_argument("--output", type=Path, help="Optional unrounded JSON results")
    args = parser.parse_args()
    gold = read_records(args.gold)
    names = cohort_names(gold, args.split, args.split_file)
    files = sorted(args.predictions.glob("*.jsonl")) if args.predictions.is_dir() else [args.predictions]
    if not files:
        parser.error("No prediction files found")
    try:
        results = [score_file(path, gold, names, split=args.split) for path in files]
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Cohort: {args.split}; {len(names)} clips. Scores: corpus micro-F1.")
    print(f"{'model_setting':44} {'n':>4} {'valid':>5} " + " ".join(f"{c:>6}" for c in COMPONENTS))
    for row in results:
        values = " ".join(f"{row['metrics'][c]['f1']:6.3f}" for c in COMPONENTS)
        print(f"{row['model_setting']:44} {row['n_clips']:4} {row['valid_records']:5} {values}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
