"""Measurements by re-execution under gdb (the recorded failure is replayed with the same inputs)."""
import json
import os
import subprocess
import tempfile

DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdb_driver.py")


def run_plan(binary, args, plan, env=None, timeout=300):
    with tempfile.TemporaryDirectory() as d:
        pp, op = os.path.join(d, "plan.json"), os.path.join(d, "out.json")
        json.dump(plan, open(pp, "w"))
        e = dict(os.environ, RR_PLAN=pp, RR_OUT=op, ASAN_OPTIONS="detect_leaks=0")
        if env:
            e.update(env)
        subprocess.run(["gdb", "-q", "-batch", "-nx", "-x", DRIVER, "--args", binary] + list(args),
                       env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        if not os.path.exists(op):
            raise RuntimeError("gdb produced no result for plan %s" % plan["mode"])
        return json.load(open(op))


def crash_state(binary, args, queries):
    return run_plan(binary, args, {"mode": "crash_state", "queries": queries})


def last_writer(binary, args, start_func, addr):
    return run_plan(binary, args, {"mode": "last_writer", "start": start_func, "addr": addr})


def reach(binary, args, locations, exprs=(), max_hits=50):
    return run_plan(binary, args, {"mode": "reach", "locations": list(locations), "exprs": list(exprs),
                                   "max_hits": max_hits})


def enforce(binary, args, locations, expr, bound, max_hits=50):
    return run_plan(binary, args, {"mode": "enforce", "locations": list(locations), "expr": expr, "bound": bound,
                                   "max_hits": max_hits})
