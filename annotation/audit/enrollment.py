"""Password-free, persistent ownership of the study's fixed annotator slots."""
from datetime import datetime, timezone
import hashlib
import json
import re
import secrets

from filelock import FileLock

from .storage import AnswerStore, atomic_json


class ParticipantStore:
    """Issue opaque resume tokens and keep participant names as private metadata.

    Constructing the store creates only enrollment metadata, never a claim.
    Tokens belong in the volunteer's browser storage; only their SHA-256 hashes
    are kept in participants.json. The Gradio browser-storage encryption secret
    is persisted separately in the same private file, so a restart can resume.
    """

    def __init__(self, study_dir):
        self.answers = AnswerStore(study_dir)
        self.path = self.answers.root / "participants.json"
        self.lock_path = str(self.path) + ".lock"
        self.order = self._allocation_order()
        with FileLock(self.lock_path, timeout=10):
            if not self.path.exists():
                atomic_json(self.path, {
                    "schema_version": 1,
                    "study_id": self.answers.study["study_id"],
                    "study_sha256": self.answers.sha256,
                    "browser_secret": secrets.token_urlsafe(32),
                    "participants": {},
                })
            self.browser_secret = self._read()["browser_secret"]

    def _allocation_order(self):
        # Fill one position in each pair first, then the second positions.
        # This spreads recruitment across the assigned clip groups.
        groups = [pair["raters"] for pair in self.answers.study.get("pairs", [])]
        order = []
        for position in range(max((len(group) for group in groups), default=0)):
            for group in groups:
                if position < len(group):
                    rater = group[position]
                    if rater not in self.answers.raters:
                        raise ValueError("Enrollment pair contains an unknown annotator.")
                    if rater not in order:
                        order.append(rater)
        order.extend(rater for rater in self.answers.study["annotators"] if rater not in order)
        return order

    def _read(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        current_sha = hashlib.sha256((self.answers.root / "study.json").read_bytes()).hexdigest()
        if (data.get("schema_version") != 1 or
                data.get("study_id") != self.answers.study["study_id"] or
                data.get("study_sha256") != self.answers.sha256 or
                current_sha != self.answers.sha256):
            raise ValueError("Study changed or enrollment belongs to another study. Restore the frozen study.")
        participants = data.get("participants")
        secret = data.get("browser_secret")
        if not isinstance(participants, dict) or not isinstance(secret, str) or len(secret) < 32:
            raise ValueError("Invalid enrollment data.")
        occupied = set()
        for token_hash, entry in participants.items():
            if (not re.fullmatch(r"[0-9a-f]{64}", token_hash) or
                    not isinstance(entry, dict) or
                    entry.get("rater") not in self.answers.raters or
                    entry["rater"] in occupied or
                    not isinstance(entry.get("claimed_at"), str)):
                raise ValueError("Invalid enrollment data.")
            if "name" in entry and self._name(entry["name"]) != entry["name"]:
                raise ValueError("Invalid participant name in enrollment data.")
            occupied.add(entry["rater"])
        return data

    @staticmethod
    def _token_hash(token):
        if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{43}", token):
            raise ValueError("Your saved volunteer session is not recognized.")
        return hashlib.sha256(token.encode("ascii")).hexdigest()

    def _lookup(self, data, token):
        entry = data["participants"].get(self._token_hash(token))
        if entry is None:
            raise ValueError("Your saved volunteer session is not recognized.")
        return entry["rater"]

    def lookup(self, token):
        """Resolve a previously issued token without claiming or changing a slot."""
        with FileLock(self.lock_path, timeout=10):
            return self._lookup(self._read(), token)

    @staticmethod
    def _name(name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Please enter your name before starting.")
        name = name.strip()
        if len(name) > 100 or not name.isprintable():
            raise ValueError("Please enter a name of at most 100 characters on one line.")
        return name

    def get_name(self, token):
        """Read only the name belonging to the supplied browser token."""
        with FileLock(self.lock_path, timeout=10):
            data = self._read()
            self._lookup(data, token)
            return data["participants"][self._token_hash(token)].get("name", "")

    def claim(self, token=None, name=None):
        """Resume a known token, or claim one unused slot when no token exists."""
        with FileLock(self.lock_path, timeout=10):
            data = self._read()
            if token is not None and token != "":
                rater = self._lookup(data, token)
                entry = data["participants"][self._token_hash(token)]
                # A returning browser need not retype its name. Names label
                # results; only the token authorizes access to existing work.
                saved_name = entry.get("name", "")
                entered_name = saved_name if name is None or name == "" else name
                entered_name = self._name(entered_name)
                if entered_name != saved_name:
                    entry["name"] = entered_name
                    atomic_json(self.path, data)
                return rater, token
            name = self._name(name)
            occupied = {entry["rater"] for entry in data["participants"].values()}
            # Existing legacy/QA answers have no anonymous owner. Never hand
            # them to a newcomer, including an empty pre-existing answer file.
            rater = next((r for r in self.order if r not in occupied and
                          not self.answers.path(r).exists()), None)
            if rater is None:
                raise ValueError("All volunteer places are taken. Thank you for your interest!")
            while True:
                new_token = secrets.token_urlsafe(32)
                token_hash = self._token_hash(new_token)
                if token_hash not in data["participants"]:
                    break
            data["participants"][token_hash] = {
                "rater": rater,
                "name": name,
                "claimed_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(self.path, data)
            return rater, new_token
