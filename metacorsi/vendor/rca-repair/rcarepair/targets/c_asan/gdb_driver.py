"""Runs inside gdb (gdb -batch -x gdb_driver.py). Reads a JSON plan from $RR_PLAN, writes JSON results to $RR_OUT.

Modes
  crash_state  run to the first ASan report; return the stack and evaluate {frame, expr} queries there
  last_writer  run to the first entry of plan.start; watch the byte at plan.addr; record every write until the ASan
               report (or exit); the last one before the report is the writer the RCA needs
  reach        stop at plan.locations (file:line); evaluate plan.exprs at each hit; report hits and whether ASan fired
  enforce      stop at plan.locations; if plan.expr is above plan.bound there, set it to plan.bound (the rule held at
               that point); report hits (value before / after) and whether ASan still fired
"""
import json
import os

import gdb

plan = json.load(open(os.environ["RR_PLAN"]))
result = {"mode": plan["mode"], "asan": False, "exited": False, "frames": [], "values": {}, "writes": [], "hits": []}
ASAN_FUNCS = ["__asan_report_%s%s" % (k, n) for k in ("store", "load") for n in ("1", "2", "4", "8", "16", "_n")]
last_stop = {"event": None}


def on_stop(ev):
    last_stop["event"] = ev


gdb.events.stop.connect(on_stop)
for cmd in ("set pagination off", "set confirm off", "set breakpoint pending on", "set print pretty off",
            "set disable-randomization on"):
    gdb.execute(cmd, to_string=True)
for fn in ASAN_FUNCS:
    gdb.execute("break " + fn, to_string=True)


def alive():
    inf = gdb.selected_inferior()
    return inf is not None and inf.pid != 0


def frames():
    out = []
    f = gdb.newest_frame()
    while f is not None:
        sal = f.find_sal()
        out.append({"func": f.name(), "file": sal.symtab.fullname() if sal.symtab else None, "line": sal.line})
        f = f.older()
    return out


def at_asan():
    try:
        name = gdb.newest_frame().name() or ""
    except gdb.error:
        return False
    return name.startswith("__asan_report_")


def value_of(v):
    t = v.type.strip_typedefs()
    if t.code == gdb.TYPE_CODE_PTR:
        return {"ptr": int(v.cast(gdb.lookup_type("unsigned long")))}
    if t.code in (gdb.TYPE_CODE_INT, gdb.TYPE_CODE_CHAR, gdb.TYPE_CODE_BOOL, gdb.TYPE_CODE_ENUM):
        return {"int": int(v)}
    return {"str": str(v)}


def nth_frame(n):
    f = gdb.newest_frame()
    for _ in range(n):
        f = f.older()
    return f


def evaluate(frame_index, expr):
    try:
        nth_frame(frame_index).select()
        return value_of(gdb.parse_and_eval(expr))
    except gdb.error as e:
        return {"error": str(e)}


def run():
    gdb.execute("run", to_string=True)


def cont():
    try:
        gdb.execute("continue", to_string=True)
    except gdb.error:
        pass


mode = plan["mode"]
if mode == "crash_state":
    run()
    if alive() and at_asan():
        result["asan"] = True
        result["frames"] = frames()
        for q in plan.get("queries", []):
            result["values"]["%d|%s" % (q["frame"], q["expr"])] = evaluate(q["frame"], q["expr"])
elif mode == "last_writer":
    bp = gdb.Breakpoint(plan["start"], internal=False)
    run()
    if alive() and not at_asan():
        bp.delete()
        addr = int(plan["addr"])
        cur = int(gdb.parse_and_eval("*(unsigned char *)%d" % addr))
        gdb.execute("watch -l *(unsigned char *)%d" % addr, to_string=True)
        while True:
            cont()
            if not alive():
                result["exited"] = True
                break
            if at_asan():
                result["asan"] = True
                result["frames"] = frames()
                break
            new = int(gdb.parse_and_eval("*(unsigned char *)%d" % addr))
            fr = frames()
            # the watchpoint stops after the store; the byte before the pc belongs to the storing statement
            fr[0]["write_line"] = gdb.find_pc_line(int(gdb.newest_frame().pc()) - 1).line
            result["writes"].append({"old": cur, "new": new, "frames": fr})
            cur = new
elif mode == "reach":
    for loc in plan["locations"]:
        gdb.Breakpoint(loc)
    run()
    while alive():
        if at_asan():
            result["asan"] = True
            result["frames"] = frames()
            break
        fr = frames()
        hit = {"func": fr[0]["func"], "file": fr[0]["file"], "line": fr[0]["line"], "values": {}}
        for e in plan.get("exprs", []):
            hit["values"][e] = evaluate(0, e)
        result["hits"].append(hit)
        if len(result["hits"]) >= plan.get("max_hits", 50):
            break
        cont()
    if not alive():
        result["exited"] = True
elif mode == "enforce":
    for loc in plan["locations"]:
        gdb.Breakpoint(loc)
    run()
    while alive():
        if at_asan():
            result["asan"] = True
            result["frames"] = frames()
            break
        fr = frames()
        v = evaluate(0, plan["expr"]).get("int")
        hit = {"func": fr[0]["func"], "line": fr[0]["line"], "before": v, "after": v}
        if v is not None and v > plan["bound"]:
            gdb.execute("set var %s = %d" % (plan["expr"], plan["bound"]), to_string=True)
            hit["after"] = evaluate(0, plan["expr"]).get("int")
        result["hits"].append(hit)
        if len(result["hits"]) >= plan.get("max_hits", 50):
            break
        cont()
    if not alive():
        result["exited"] = True
json.dump(result, open(os.environ["RR_OUT"], "w"))
gdb.execute("kill", to_string=True) if alive() else None
