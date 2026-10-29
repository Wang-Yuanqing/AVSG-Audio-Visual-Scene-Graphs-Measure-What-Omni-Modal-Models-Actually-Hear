"""Rebuild the campaign's ms-swift training records from the official split."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from prompt import SYSTEM_PROMPT, build_user_text


def build_records(gold, split, clips_dir, model):
    train = set(split["train_clips"])
    test = set(split["test_clips"])
    if train & test:
        raise ValueError("Train and test clips overlap.")
    by_name = {r["filename"]: r for r in gold}
    if train | test != set(by_name):
        raise ValueError("Split does not exactly cover gold.")
    source = lambda name: name.removesuffix(".mp4").rsplit("_", 2)[0]
    if {source(n) for n in train} & {source(n) for n in test}:
        raise ValueError("Source videos overlap across train and test.")
    records = []
    # Preserve the saved split order, target serialization and prompt bytes.
    for filename in split["train_clips"]:
        record = by_name[filename]
        media = str(clips_dir / record["filename"])
        target = {"nodes": record["ground_truth_nodes"], "edges": record["ground_truth_edges"]}
        prefix = "<video><audio>" if model == "qwen25" else "<video>"
        item = {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prefix + build_user_text()},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
        ], "videos": [media]}
        if model == "qwen25":
            # The original loader extracts audio from the same MP4 separately.
            item["audios"] = [media]
        records.append(item)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen25", "qwen3"], required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=ROOT / "data/gold/1k_result.jsonl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/split_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-media-check", action="store_true", help="Prepare records without local media (for inspection only).")
    args = parser.parse_args()
    gold = [json.loads(line) for line in args.gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = build_records(gold, json.loads(args.split.read_text(encoding="utf-8")), args.clips_dir.resolve(), args.model)
    if not args.skip_media_check:
        missing = [r["videos"][0] for r in records if not Path(r["videos"][0]).is_file()]
        if missing:
            parser.error(f"Missing {len(missing)} training clips; first: {missing[0]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(f"Wrote {len(records)} {args.model} training records to {args.output}")


if __name__ == "__main__":
    main()
