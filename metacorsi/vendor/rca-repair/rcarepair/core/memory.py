"""The two long-term memories and the episodic log.

LTM 1 (case base): bug signature -> RCA method, root cause, knowledge column, why chain (deck page 1 + cases).
LTM 2 (compounds): knowledge column x signature -> element combination, parameter pattern, generalisation query,
success record (periodic table + compounds).
"""
import json
import os


def _load(path):
    if path and os.path.exists(path):
        return json.load(open(path))
    return []


def _save(path, rows):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        json.dump(rows, open(path, "w"), ensure_ascii=False, indent=1)


def sig_similarity(a, b):
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    return sum(1 for k in keys if a.get(k) == b.get(k)) / len(keys)


def sig_covers(general, sig):
    if not general:
        return 0.0
    return sum(1 for k, v in general.items() if sig.get(k) == v) / len(general)


class LTM1:
    def __init__(self, path=None):
        self.path = path
        self.cases = _load(path)

    def match(self, sig, tau=0.8):
        best, conf = None, 0.0
        for c in self.cases:
            s = sig_similarity(sig, c["signature"])
            if s > conf:
                best, conf = c, s
        return (best, conf) if conf >= tau else (None, conf)

    def add(self, case):
        self.cases.append(case)
        _save(self.path, self.cases)


class LTM2:
    def __init__(self, path=None):
        self.path = path
        self.entries = _load(path)

    def lookup(self, k, sig, tau=0.8):
        """entries keep a class-level signature (a few keys); an entry applies when the bug agrees on those keys"""
        rows = [e for e in self.entries if e["k"] == k and sig_covers(e["signature"], sig) >= tau]
        return sorted(rows, key=lambda e: -e.get("success", 0) / max(1, e.get("tries", 1)))

    def add_or_update(self, entry):
        for e in self.entries:
            if e["k"] == entry["k"] and e["compound"] == entry["compound"] and e["signature"] == entry["signature"]:
                e["tries"] = e.get("tries", 0) + entry.get("tries", 1)
                e["success"] = e.get("success", 0) + entry.get("success", 0)
                e.setdefault("sites_seen", []).extend(entry.get("sites_seen", []))
                _save(self.path, self.entries)
                return e
        self.entries.append(entry)
        _save(self.path, self.entries)
        return entry


class EpisodeLog:
    def __init__(self, path=None):
        self.path = path
        self.rows = []

    def add(self, ep):
        from dataclasses import asdict
        self.rows.append(asdict(ep))
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(asdict(ep), ensure_ascii=False) + "\n")
