"""Build the historical Vertex batch requests; credentials are never embedded."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from prompt import SYSTEM_PROMPT, build_user_text


def make_request(filename, caption, gcs_dir):
    return {"key": filename, "request": {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [
            {"text": build_user_text(caption)},
            {"file_data": {"mime_type": "video/mp4", "file_uri": f"{gcs_dir.rstrip('/')}/{filename}"}}
        ]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcs-dir", required=True)
    parser.add_argument("--gold", type=Path, default=ROOT / "data/gold/1k_result.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/requests")
    args = parser.parse_args()
    if not args.gcs_dir.startswith("gs://"):
        parser.error("--gcs-dir must be a gs:// location containing your MP4 clips.")
    records = [json.loads(s) for s in args.gold.read_text(encoding="utf-8").splitlines() if s.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for setting in ("raw", "caption"):
        rows = [make_request(r["filename"], r.get("caption", "") if setting == "caption" else None, args.gcs_dir) for r in records]
        target = args.output_dir / f"{setting}.jsonl"
        target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{setting}: {len(rows)} requests -> {target}")


if __name__ == "__main__":
    main()
