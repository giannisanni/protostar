"""Knowledge registry: what was written, when, and on what evidence.

Design principle from the CallMeMaybe discussion (2026-08-02): a system that
modifies itself needs structural self-knowledge, not behavioural probing. The
sleep job must be able to ask "do I already know this?" and "what must I
re-check after this write?" by LOOKING IT UP, not by generating text and
squinting at the output.

Append-only JSONL. Every consolidation records its own evidence, so a rolled
back write leaves a trace too — the failures are the interesting rows.
"""
import hashlib
import json
from pathlib import Path


def content_key(fact):
    """Stable id for a fact's CONTENT, independent of surface form."""
    raw = f"{fact['subject']}|{fact['template']}|{fact['object']}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class KnowledgeRegistry:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._rows.append(json.loads(line))

    def append(self, row):
        self._rows.append(row)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def knows(self, key):
        """Dedup: has this exact content been COMMITTED before?"""
        return any(r["key"] == key and r["committed"] for r in self._rows)

    def committed_facts(self):
        """Everything successfully written — the retention suite probes these."""
        return [r for r in self._rows if r["committed"]]

    def cycles(self):
        return sorted({r["cycle"] for r in self._rows})

    def summary(self):
        ok = sum(r["committed"] for r in self._rows)
        return f"{len(self._rows)} writes, {ok} committed, {len(self._rows) - ok} rolled back"
