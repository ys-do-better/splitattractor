"""Records which box of the two flowcharts each step went through, with the real values, so a run can be read back
(and drawn) step by step."""
import json
import os
import time


class RunLog:
    def __init__(self, path=None, echo=True):
        self.path = path
        self.echo = echo
        self.steps = []

    def step(self, node, zh, **data):
        """node: flowchart box id (e.g. rca.case_match); zh: one line for people; data: the values behind it"""
        row = {"i": len(self.steps), "node": node, "zh": zh, "data": data, "t": round(time.time(), 3)}
        self.steps.append(row)
        if self.echo:
            print(f"[{node}] {zh}", flush=True)
        self.save()
        return row

    def save(self):
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            json.dump(self.steps, open(self.path, "w"), ensure_ascii=False, indent=1, default=str)
