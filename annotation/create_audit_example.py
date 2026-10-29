"""Create a fresh two-rater demonstration study from one released graph."""
import argparse
import json
from pathlib import Path

from .audit.rules import rule_for

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "annotation/example.jsonl")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory; existing studies are never overwritten.")
    record = json.loads(args.input.read_text(encoding="utf-8").splitlines()[0])
    nodes = {node["id"]: node for node in record["ground_truth_nodes"]}
    tasks = []
    for index, edge in enumerate(record["ground_truth_edges"], 1):
        task = {"task_id": f"t{index:04d}", "source": nodes[edge["source"]],
                "target": nodes[edge["target"]], "relation": edge["edge"],
                "family": edge["source"][0] + "2" + edge["target"][0]}
        task["rule"] = rule_for(task)
        tasks.append(task)
    study = {"study_id": "avsg_public_ui_demo", "schema_version": 2, "seed": 42,
             "annotators": ["rater1", "rater2"],
             "clips": [{"clip_id": "c001", "filename": record["filename"], "tasks": tasks}],
             "pairs": [{"raters": ["rater1", "rater2"], "clip_ids": ["c001"]}],
             "assignments": {"c001": ["rater1", "rater2"]},
             "orders": {"rater1": ["c001"], "rater2": ["c001"]}}
    args.output.mkdir(parents=True)
    (args.output / "study.json").write_text(json.dumps(study, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Created demonstration study: {len(tasks)} edges, two raters. Media: {record['filename']}")


if __name__ == "__main__":
    main()
