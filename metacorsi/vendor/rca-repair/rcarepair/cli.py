"""usage: python -m rcarepair run EXAMPLE_DIR [--out DIR] [--memory DIR] [--fresh]

Runs the RCA agent on the example's failure, hands the diagnosis to the repair agent, and writes
runlog.json (every flowchart box passed, with values), diagnosis.json, final.diff, upstream.json, episodes.jsonl,
summary.md, and the two long-term memories.
"""
import argparse
import json
import os
import shutil
import time

from .core.memory import LTM1, LTM2, EpisodeLog
from .core.rca import RCAAgent
from .core.repair import RepairAgent
from .core.runlog import RunLog
from .targets.c_asan.target import CTarget


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rcarepair")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("example")
    r.add_argument("--out", default=None)
    r.add_argument("--memory", default="memory")
    r.add_argument("--fresh", action="store_true", help="start with empty LTM 1 / LTM 2")
    a = ap.parse_args(argv)
    name = os.path.basename(os.path.normpath(a.example))
    out = os.path.abspath(a.out or os.path.join("runs", name))
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    if a.fresh and os.path.exists(a.memory):
        shutil.rmtree(a.memory)
    os.makedirs(a.memory, exist_ok=True)
    ltm1, ltm2 = LTM1(os.path.join(a.memory, "ltm1.json")), LTM2(os.path.join(a.memory, "ltm2.json"))
    episodes = EpisodeLog(os.path.join(out, "episodes.jsonl"))
    log = RunLog(os.path.join(out, "runlog.json"))
    t0 = time.time()
    target = CTarget(a.example, os.path.join(out, "work"))
    log.step("setup", "目标系统 %s：%d 个被分析的源文件，编译通过；回归测试 %d 个" % (
        target.name, len(target.code), len(target.tests["regression"])), example=os.path.abspath(a.example))
    D, ev = RCAAgent(ltm1, log, episodes).run(target, target.scenario.get("access", "white"))
    for e in ev.episodes:
        episodes.add(e)
    json.dump(D.to_json(), open(os.path.join(out, "diagnosis.json"), "w"), ensure_ascii=False, indent=1, default=str)
    res = RepairAgent(ltm2, log).run(target, D, ev)
    open(os.path.join(out, "final.diff"), "w").write(target.final_diff())
    up = target.compare_upstream()
    if up is not None:
        json.dump(up, open(os.path.join(out, "upstream.json"), "w"), indent=1)
        log.step("compare.upstream", "和上游合入的修复比：%s" % (
            "逐字一致" if up.get("identical") else ("不一致" if up.get("applied") else "上游补丁打不上")), upstream=up)
    log.step("done", "完成，用时 %.0f 秒，编译 %d 次" % (time.time() - t0, target.nbuild),
             seconds=round(time.time() - t0, 1), builds=target.nbuild, patched=res.get("patched"))
    for f in ("ltm1.json", "ltm2.json"):
        if os.path.exists(os.path.join(a.memory, f)):
            shutil.copy(os.path.join(a.memory, f), os.path.join(out, f))
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("# %s\n\n" % target.name)
        for s in log.steps:
            f.write("- `%s` %s\n" % (s["node"], s["zh"]))
    print("output:", out)


if __name__ == "__main__":
    main()
