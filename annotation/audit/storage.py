"""Durable, per-rater answers with revision checks and no shared judgments."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

VERDICTS = {"correct", "incorrect", "uncertain"}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # tempfile.mkstemp retries PermissionError on Windows when os.access reports
    # a directory writable. Under a restricted token that check can be wrong,
    # causing a very long retry loop. Retry genuine name collisions only.
    for _ in range(10):
        tmp = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError("Could not allocate a unique temporary answer file.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class AnswerStore:
    def __init__(self, study_dir):
        self.root = Path(study_dir).resolve()
        raw = (self.root / "study.json").read_bytes()
        self.study = json.loads(raw)
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.raters = set(self.study["annotators"])
        if any(not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name)
               for name in self.raters):
            raise ValueError("Annotator IDs must use lowercase letters, digits, '-' or '_'.")
        self.tasks = {t["task_id"] for c in self.study["clips"] for t in c["tasks"]}
        if len(self.tasks) != sum(len(c["tasks"]) for c in self.study["clips"]):
            raise ValueError("Duplicate task IDs.")
        clips = {c["clip_id"]: c for c in self.study["clips"]}
        self.assigned = {rater: {t["task_id"] for cid in self.study["orders"][rater]
                                  for t in clips[cid]["tasks"]} for rater in self.raters}

    def path(self, annotator):
        if annotator not in self.raters:
            raise ValueError("Unknown annotator.")
        return self.root / ("judgments_" + annotator + ".json")

    def _read(self, annotator):
        path = self.path(annotator)
        if not path.exists():
            return {"study_id": self.study["study_id"], "study_sha256": self.sha256,
                    "annotator": annotator, "answers": {}, "history": []}
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data["study_id"] != self.study["study_id"] or
                data.get("study_sha256") != self.sha256 or data["annotator"] != annotator):
            raise ValueError("Study changed or answer file belongs to another rater. Restore the frozen study.")
        if set(data["answers"]) - self.assigned[annotator]:
            raise ValueError("Answer file contains relations outside this rater's assignment.")
        return data

    def read(self, annotator):
        path = self.path(annotator)
        with FileLock(str(path) + ".lock", timeout=10):
            return self._read(annotator)

    def save(self, annotator, task_id, verdict, note="", reason="none",
             expected_revision=0, elapsed_seconds=None, media_watched=False):
        self.path(annotator)
        if task_id not in self.assigned[annotator]:
            raise ValueError("This relation is not assigned to your account.")
        if verdict not in VERDICTS or reason not in {"none", "media_issue"}:
            raise ValueError("Choose Correct, Incorrect, or Cannot tell.")
        if reason == "media_issue" and verdict != "uncertain":
            raise ValueError("Media problems must be recorded as Cannot tell.")
        if not media_watched and reason != "media_issue":
            raise ValueError("Watch and listen to this clip before saving.")
        if not isinstance(note, str) or len(note) > 3000:
            raise ValueError("Notes must be at most 3,000 characters.")
        path = self.path(annotator)
        with FileLock(str(path) + ".lock", timeout=10):
            data = self._read(annotator)
            current = data["answers"].get(task_id)
            revision = current["revision"] if current else 0
            if revision != expected_revision:
                raise ValueError("This answer changed in another tab. Reload before saving.")
            if current:
                data["history"].append(dict(current))
            answer = {"task_id": task_id, "verdict": verdict,
                      "reason": reason, "note": note.strip(), "revision": revision + 1,
                      "saved_at": datetime.now(timezone.utc).isoformat(),
                      "elapsed_seconds": elapsed_seconds, "media_watched": bool(media_watched)}
            data["answers"][task_id] = answer
            data["updated_at"] = answer["saved_at"]
            atomic_json(path, data)
        return answer
