"""Parse an AddressSanitizer report into the failure signal the RCA starts from."""
import re

FRAME = re.compile(r"^\s*#(\d+)\s+0x[0-9a-f]+\s+in\s+(\S+)\s+(\S+?):(\d+)(?::(\d+))?\s*$")


def parse(stderr):
    if "ERROR: AddressSanitizer" not in stderr:
        return None
    lines = stderr.splitlines()
    rep = {"bug": None, "access": None, "size": None, "frames": [], "region": None, "alloc_frames": []}
    for i, ln in enumerate(lines):
        m = re.search(r"ERROR: AddressSanitizer: ([\w-]+) on address (0x[0-9a-f]+)", ln)
        if m:
            rep["bug"], rep["addr"] = m.group(1), int(m.group(2), 16)
        m = re.match(r"^(READ|WRITE) of size (\d+) at (0x[0-9a-f]+)", ln)
        if m:
            rep["access"], rep["size"] = m.group(1), int(m.group(2))
        m = re.search(r"is located (\d+) bytes (to the right|to the left|inside) of (\d+)-byte region \[(0x[0-9a-f]+),(0x[0-9a-f]+)\)", ln)
        if m:
            rep["region"] = {"offset": int(m.group(1)), "side": m.group(2), "size": int(m.group(3)),
                             "start": int(m.group(4), 16), "end": int(m.group(5), 16)}
    section = "frames"
    for ln in lines:
        if "allocated by thread" in ln:
            section = "alloc_frames"
        elif ln.startswith("SUMMARY") or "freed by thread" in ln:
            section = None
        m = FRAME.match(ln)
        if m and section:
            rep[section].append({"func": m.group(2), "file": m.group(3), "line": int(m.group(4))})
    return rep


def is_error(stderr):
    return "ERROR: AddressSanitizer" in stderr
