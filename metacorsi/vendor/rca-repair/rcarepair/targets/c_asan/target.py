"""A C program checked by AddressSanitizer: build, run, replay under gdb, patch, validate.

The example directory supplies target.yaml (how to build and run, which files are under repair, tests, what the
operator can do on the running system), knowledge.yaml (where outside data enters, which calls copy LEN bytes) and
scenario.yaml (is the system running, can it stop, what access we have).
"""
import collections
import difflib
import itertools
import os
import re
import shutil
import subprocess

import yaml

from ...core.model import Episode, Evidence, Signal
from . import asan, gdbprobe
from .analysis import Analysis
from .cast import CFile, norm

RUN_ENV = {"ASAN_OPTIONS": "detect_leaks=0:exitcode=86"}
ORIGIN_ZH = {"device": "设备"}
OP_LINE = re.compile(r"^OP \d+ (\S+) addr=(0x[0-9a-f]+)\s*(.*)$")


class CTarget:
    deterministic = True  # the same inputs replay the same way, so every question is asked once

    def __init__(self, example_dir, workdir):
        self.ex = os.path.abspath(example_dir)
        cfg = yaml.safe_load(open(os.path.join(self.ex, "target.yaml")))
        self.cfg = cfg
        self.name = cfg["name"]
        self.knowledge = yaml.safe_load(open(os.path.join(self.ex, cfg["knowledge"])))
        self.scenario = yaml.safe_load(open(os.path.join(self.ex, cfg["scenario"])))
        self.workdir = os.path.abspath(workdir)
        src = os.path.join(self.ex, cfg["root"])
        self.orig, self.work = os.path.join(self.workdir, "orig"), os.path.join(self.workdir, "work")
        for d in (self.orig, self.work):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(src, d)
        self.code = cfg["code"]
        self.tests = {"reproducer": self._ex(cfg["tests"]["reproducer"]),
                      "regression": [self._ex(t) for t in cfg["tests"]["regression"]],
                      "workload": self._ex(cfg["tests"]["workload"])}
        self.ana_orig = Analysis(self.orig, self.code, self.knowledge, [os.path.join(self.orig, d) for d in cfg["include"]])
        self.ana = Analysis(self.work, self.code, self.knowledge, [os.path.join(self.work, d) for d in cfg["include"]])
        self.edits = []
        self.nbuild = 0
        self.bin_orig = self.build(self.orig, "orig")
        self.bin = self.bin_orig
        self.baseline = {t: self.ops(self.run(self.bin_orig, t)) for t in self.tests["regression"]}
        self.config_path = os.path.join(self.workdir, "system.conf")

    def _ex(self, p):
        return os.path.join(self.ex, p)

    # ---- build and run ------------------------------------------------------------------------------------------
    def build(self, root, tag):
        out = os.path.join(self.workdir, "build", tag)
        os.makedirs(out, exist_ok=True)
        binary = os.path.join(out, "system")
        cmd = ([self.cfg["cc"]] + self.cfg["cflags"] + ["-I" + os.path.join(root, d) for d in self.cfg["include"]]
               + [os.path.join(root, f) for f in self.cfg["sources"]] + ["-o", binary])
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            raise RuntimeError("build failed:\n" + r.stderr[-3000:])
        self.nbuild += 1
        return binary

    def run(self, binary, scenario, config=None, timeout=60):
        args = [binary, scenario] + (["--config", config] if config else [])
        r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                           env=dict(os.environ, **RUN_ENV), timeout=timeout)
        return {"exit": r.returncode, "stdout": r.stdout, "stderr": r.stderr, "asan": asan.parse(r.stderr)}

    @staticmethod
    def ops(run):
        return [ln.strip() for ln in run["stdout"].splitlines() if ln.startswith("OP ")]

    @staticmethod
    def last_op(run):
        lines = [ln for ln in run["stdout"].splitlines() if ln.startswith("OP ")]
        if not lines:
            return None
        m = OP_LINE.match(lines[-1].split("=" * 10)[0].strip())
        return {"op": m.group(1), "addr": int(m.group(2), 16)} if m else None

    # ---- RCA side -------------------------------------------------------------------------------------------------
    def failure(self):
        r = self.run(self.bin_orig, self.tests["reproducer"])
        rep = r["asan"]
        if rep is None:
            raise RuntimeError("the reproducer does not fail")
        f0 = rep["frames"][0]
        op = self.last_op(r)
        sig = Signal("%s %s" % (rep["bug"], rep["access"]), "%s:%d %s" % (os.path.basename(f0["file"]), f0["line"], f0["func"]),
                     rep, self.tests["reproducer"])
        ev = Evidence()
        ev.add("crash_report", rep)
        ev.add("failing_op", op)
        ev.episodes.append(Episode(o="%s addr=0x%02x" % (op["op"], op["addr"]), a="run the reproducer",
                                   f="ASan %s %s of size %s" % (rep["bug"], rep["access"], rep["size"]),
                                   t="%s:%d" % (os.path.basename(f0["file"]), f0["line"]), meta={"frames": rep["frames"][:6]}))
        return sig, ev

    def signature(self, sig, ev):
        rep = sig.report
        fr = [f["func"] for f in rep["frames"] if self.ana_orig.is_code(f["file"])]
        return {"bug": rep["bug"], "access": rep["access"], "sink": fr[0] if fr else None,
                "path": " <- ".join(fr[:3]), "system": self.cfg.get("family", self.name)}

    def remind(self, sig):
        return []

    def units(self, access):
        return {"black": "组件、输入、配置、改动", "white": "语句、(变量, 时刻)", "spec": "规则、需求"}[access]

    def in_hand(self, ev):
        have = set()
        if ev.has("crash_report"):
            have.add("failing_trace")
        if ev.has("spectra"):
            have.add("spectra")
        if self.knowledge.get("component_model"):
            have.add("component_model")
        if self.scenario.get("rerun") == "controllable":
            have.add("controllable_rerun")
        return have

    def _values(self, st):
        return {(int(k.split("|", 1)[0]), norm(k.split("|", 1)[1])): v for k, v in st["values"].items()}

    def backtrack(self, ev):
        rep = ev.get("crash_report")
        st = ev.get("crash_state")
        if st is None:
            st = gdbprobe.crash_state(self.bin_orig, [self.tests["reproducer"]], [])
            ev.add("crash_state", st)
        values = ev.get("crash_values") or {}
        memdeps = ev.get("memdeps") or {}
        for _ in range(6):  # each round asks gdb (one replay) for the values the walk still needs
            hops, missing, need = self.ana_orig.backtrack(rep, st["frames"], values, memdeps)
            need = [q for q in need if (q["frame"], norm(q["expr"])) not in values]
            if not need:
                break
            got = gdbprobe.crash_state(self.bin_orig, [self.tests["reproducer"]], need)
            values.update(self._values(got))
        ev.add("crash_values", values)
        return hops, missing

    def measure(self, missing, ev):
        """acquire the next piece of evidence the method asked for; returns (key, Chinese description)"""
        if missing and missing["need"] == "last_writer":
            r = gdbprobe.last_writer(self.bin_orig, [self.tests["reproducer"]], missing["start"], missing["addr"])
            writes = r["writes"]
            memdeps = ev.get("memdeps") or {}
            if writes:
                memdeps[str(missing["addr"])] = writes[-1]
            ev.add("memdeps", memdeps)
            w = writes[-1] if writes else None
            ep = Episode(o="watch %s" % missing["expr"], a="replay the failure under gdb, watchpoint on the byte",
                         f=("written %d time(s); last %d -> %d by %s" % (len(writes), w["old"], w["new"], w["frames"][0]["func"]))
                         if w else "never written", t=missing.get("where", ""), meta={"writes": len(writes)})
            ev.episodes.append(ep)
            return "memdeps", {"expr": missing["expr"], "writes": len(writes), "writer": w["frames"][0]["func"] if w else None,
                               "writer_line": w["frames"][0].get("write_line") if w else None,
                               "via": [f["func"] for f in w["frames"][1:4]] if w else [], "new": w["new"] if w else None}
        raise RuntimeError("no way to measure %s" % (missing,))

    def localization(self, hops):
        return self.ana_orig.localization(hops)

    def guards_on(self, hops):
        return self.ana_orig.guards_on(hops)

    def later_checks(self, hops, L, ev):
        """checks on the same memory location that run only after the write (static candidates, then gdb compares
        their addresses with the location the bad value was read from)"""
        st = ev.get("crash_state")
        cands = self.ana_orig.late_check_candidates(st["frames"], L.extra["frame"])
        if not cands:
            return []
        got = gdbprobe.crash_state(self.bin_orig, [self.tests["reproducer"]],
                                   [{"frame": c["frame"], "expr": "&(%s)" % c["expr"]} for c in cands])
        vals = self._values(got)
        target_addr = next(h.extra["addr"] for h in hops if h.kind == "write")
        out = []
        for c in cands:
            v = vals.get((c["frame"], norm("&(%s)" % c["expr"])), {})
            if v.get("ptr") == target_addr:
                c["bound_value"] = self.ana_orig.bound_value(c["bound"])
                out.append(c)
        return out

    def bound_value(self, bound):
        return self.ana_orig.bound_value(bound)

    # ---- RCA side: answers for the walk up the build steps (core/walk.py) ----------------------------------------------
    def outside_value(self, hops):
        """the value that came in from outside the system on this run, and who sent it"""
        origin = next((h for h in hops if h.kind == "origin" and h.extra.get("kind") == "external"), None)
        if origin is None:
            return None
        write = next((h for h in hops if h.kind == "write"), None)
        who = origin.extra.get("origin")
        return {"value": write.value if write else origin.value, "who": who, "who_zh": ORIGIN_ZH.get(who, who),
                "via": origin.func, "where": "%s:%d" % ((write or origin).file, (write or origin).line)}

    def state_fits(self, iota, hops):
        """the schema side of the state the logic writes into: does the buffer hold the longest legal value?"""
        region = hops[0].extra.get("region") or {}
        size, bv = region.get("size"), iota.get("bound_value")
        ok = size is not None and bv is not None and size > bv
        return {"ok": ok, "size": size, "bound_value": bv,
                "zh": "缓冲区 %s 字节，%s合法的最大值（%s）" % (size, "装得下" if ok else "装不下", bv)}

    def logic_under(self, iota, L):
        """existing tests that pass through L with the value inside the assumption: does the logic stay correct?"""
        loc = "%s:%d" % (os.path.basename(L["file"]), L["stmt_line"])
        bv = iota.get("bound_value")
        used = []
        for t in self.tests["regression"]:
            r = gdbprobe.reach(self.bin_orig, [t], [loc], [L["expr_text"]])
            vals = [h["values"][L["expr_text"]].get("int") for h in r["hits"]]
            if vals and all(v is not None and bv is not None and v <= bv for v in vals):
                used.append({"test": os.path.basename(t), "values": sorted(set(vals)), "asan": r["asan"]})
        if not used:
            return {"ok": None, "tests": [], "values": [], "zh": "没有现成的测试经过这里"}
        ok = not any(u["asan"] for u in used)
        vals = sorted({v for u in used for v in u["values"]})
        return {"ok": ok, "tests": used, "values": vals,
                "zh": "已有 %d 个测试经过这里，值为 %s，都在假设里：%s" % (len(used), "、".join(map(str, vals)),
                                                            "都没出错" if ok else "也出错")}

    def rule_known(self, iota):
        """is the rule the assumption needs already written somewhere in the system?"""
        late = iota.get("late_check")
        if late:
            at = "%s:%d" % (late["file"].split("/")[-1], late["line"])
            return {"ok": True, "where": at, "zh": "有：%s 查同一个值 ≤ %s" % (at, late["bound_value"]),
                    "not_at_boundary_zh": "规则在 %s，写完之后才查；值一进来就被用了，边界上没查" % at}
        return {"ok": False, "zh": "系统里找不到这条规则"}

    def case_expected(self, iota):
        """does the design expect the outside to break the rule, i.e. is there an error path for it?"""
        late = iota.get("late_check")
        if late and late.get("exits"):
            return {"ok": True, "zh": "预料到了：越过上限就报 %s" % late.get("errno")}
        return {"ok": False, "zh": "设计里没有处理这种情况的路径"}

    def replay_as_is(self, root, iota, L):
        """the control for walking back: replay this failure under the same harness with nothing changed"""
        loc = "%s:%d" % (os.path.basename(L["file"]), L["stmt_line"])
        r = gdbprobe.reach(self.bin_orig, [self.tests["reproducer"]], [loc], [L["expr_text"]])
        vals = [h["values"][L["expr_text"]].get("int") for h in r["hits"]]
        return {"fails": bool(r["asan"]), "values": vals, "asan": r["asan"], "loc": loc,
                "zh": "原样重放，%s 处 %s 为 %s：%s" % (loc, L["expr_text"], "、".join(map(str, vals)) or "没经过",
                                                "越界" if r["asan"] else "没越界")}

    def replay_with_root_right(self, root, iota, L):
        """replay this failure with the root step done right; at the boundary: the rule holds where the value is used"""
        if root["step"] != "boundary":
            return {"ok": None, "hits": [], "zh": "这个系统还不能这样重放"}
        loc = "%s:%d" % (os.path.basename(L["file"]), L["stmt_line"])
        r = gdbprobe.enforce(self.bin_orig, [self.tests["reproducer"]], [loc], L["expr_text"], iota["bound_value"])
        changed = [h for h in r["hits"] if h["before"] != h["after"]]
        ok = not r["asan"]
        return {"ok": ok, "hits": r["hits"], "asan": r["asan"], "loc": loc,
                "zh": "重放这次失败，在 %s 守住 %s ≤ %s（%s）：%s" % (
                    loc, L["expr_text"], iota["bound_value"],
                    "、".join("%s → %s" % (h["before"], h["after"]) for h in changed) or "值本来就在里面",
                    "不再越界" if ok else "还是越界")}

    def class_sites(self, root, iota, L):
        """the places the same step made: every value written by the device used as a copy length, unchecked before"""
        return [dict(s, this=os.path.basename(s["file"]) == os.path.basename(L["file"]) and s["stmt_line"] == L["stmt_line"])
                for s in self.sites(original=True)]

    def class_test(self, site):
        return self.site_test(site)

    # ---- repair side: stop-gap on the running system ---------------------------------------------------------------
    def recurring(self):
        r = self.run(self.bin, self.tests["workload"])
        return r["asan"] is not None, r

    def blackbox(self, sym, ev):
        h = self.cfg.get("blackbox", {}).get(sym)
        if not h:
            return None
        h = dict(h, sym=sym)
        if h["how"] == "config":
            op = ev.get("failing_op")
            h["text"] = h["line"].format(addr="0x%02x" % op["addr"])
        return h

    def apply_blackbox(self, h):
        if h["how"] == "config":
            open(self.config_path, "w").write(h["text"] + "\n")

    def undo_blackbox(self, h):
        if h["how"] == "config" and os.path.exists(self.config_path):
            os.remove(self.config_path)

    def stable(self, h=None):
        """does the failure stay away for a while: run the workload (a polling loop) on the current build"""
        cfg = self.config_path if (h and h["how"] == "config" and os.path.exists(self.config_path)) else None
        r = self.run(self.bin, self.tests["workload"], config=cfg)
        ops = self.ops(r)
        counts = collections.Counter()
        for ln in ops:
            m = OP_LINE.match(ln)
            if m:
                ret = re.search(r"ret=(-?\d+)", m.group(3))
                counts[(m.group(1), m.group(2), "denied" if "denied" in ln else
                        ("error %s" % ret.group(1) if ret and int(ret.group(1)) < 0 else "ok"))] += 1
        return r["asan"] is None and r["exit"] == 0, {"ops": len(ops), "asan": r["asan"] is not None,
                                                     "outcomes": {" ".join(k): v for k, v in counts.items()}}

    # ---- repair side: patches ---------------------------------------------------------------------------------------
    def params(self, sym, D, ev):
        """fill an element's parameters; yields theta dicts. Only elements this target knows how to write are offered."""
        if sym != "Cd":
            return None
        return self._params_cd(D, D.L["file"], D.L["stmt_line"], D.L["expr_text"])

    def _idiom(self, rel, line):
        cf = self.ana.cfile(os.path.join(self.work, rel))
        fn = cf.function_at(line)
        exits = cf.error_exits(fn)
        if not exits:
            return None, None, []
        lhs, leave = collections.Counter((e[0], e[2]) for e in exits).most_common(1)[0][0]
        errnos = collections.Counter(e[1] for e in exits)
        lines = [e[3] for e in exits if (e[0], e[2]) == (lhs, leave)]
        return (lhs, leave), errnos, lines

    def _params_cd(self, D, rel, stmt_line, expr_text):
        idiom, errnos, lines = self._idiom(rel, stmt_line)
        late = D.iota.get("late_check")
        cands = []
        if late and late.get("errno"):
            cands.append((late["errno"], "errno of the later check (%s:%d)" % (late["file"], late["line"]),
                          "错误码借核心层那条检查的 %s" % late["errno"]))
        for e, _ in (errnos or {}).most_common():
            if e not in [c[0] for c in cands]:
                cands.append((e, "most common errno in this function", "错误码用本函数最常见的 %s" % e))
        for errno, why, why_zh in cands:
            body = ["%s = %s;" % (idiom[0], errno), idiom[1]] if idiom else ["return %s;" % errno]
            yield {"sym": "Cd", "file": rel, "before_line": stmt_line, "cond": "%s > %s" % (expr_text, D.iota["bound"]),
                   "body": body, "expr_text": expr_text,
                   "why": {"cond": "the invariant iota, negated", "body": "error exit borrowed from lines %s" % lines,
                           "errno": why},
                   "why_zh": {"cond": "条件由不变式 ι 取反：%s 不能超过 %s" % (expr_text, D.iota["bound"]),
                              "body": "出错写法借同函数第 %s 行的「%s = …; %s」" % ("、".join(map(str, lines)), idiom[0], idiom[1])
                              if idiom else "本函数没有出错写法，直接 return",
                              "errno": why_zh}}

    def apply(self, theta):
        path = os.path.join(self.work, theta["file"])
        lines = open(path).read().split("\n")
        ln = theta["before_line"]
        ind = lines[ln - 1][: len(lines[ln - 1]) - len(lines[ln - 1].lstrip())]
        new = [ind + "if (%s) {" % theta["cond"]] + [ind + "\t" + b for b in theta["body"]] + [ind + "}"]
        lines[ln - 1:ln - 1] = new
        open(path, "w").write("\n".join(lines))
        theta["inserted"] = len(new)
        self.edits.append(theta)
        self.ana.reset()
        try:
            self.bin = self.build(self.work, "patch%d" % len(self.edits))
            return True, None
        except RuntimeError as e:
            return False, str(e)

    def revert(self, theta):
        path = os.path.join(self.work, theta["file"])
        lines = open(path).read().split("\n")
        ln = theta["before_line"]
        del lines[ln - 1: ln - 1 + theta["inserted"]]
        open(path, "w").write("\n".join(lines))
        self.edits.remove(theta)
        self.ana.reset()
        self.bin = self.build(self.work, "revert%d" % self.nbuild)

    def validate(self, theta, D, extra_tests=()):
        """tests, then the reproducer, then the invariant at the patched call"""
        steps = []
        r = self.run(self.bin, self.tests["reproducer"])
        steps.append({"what": "reproducer", "zh": "复现", "ok": r["asan"] is None and r["exit"] == 0,
                      "detail": (self.ops(r) or ["-"])[-1]})
        for t in extra_tests:
            r = self.run(self.bin, t)
            steps.append({"what": "generated test %s" % os.path.basename(t), "zh": "生成的测试", "ok": r["asan"] is None and r["exit"] == 0,
                          "detail": (self.ops(r) or ["-"])[-1]})
        same = all(self.ops(self.run(self.bin, t)) == self.baseline[t] for t in self.tests["regression"])
        steps.append({"what": "regression", "zh": "回归测试", "ok": same,
                      "detail": "%d tests, outputs identical to the original build" % len(self.tests["regression"])})
        call_line = theta["before_line"] + theta["inserted"]
        bv = D.iota.get("bound_value")
        seen = []
        for t in [self.tests["reproducer"]] + list(extra_tests) + self.tests["regression"]:
            res = gdbprobe.reach(self.bin, [t], ["%s:%d" % (os.path.basename(theta["file"]), call_line)],
                                 [theta["expr_text"]])
            seen += [h["values"][theta["expr_text"]].get("int") for h in res["hits"]]
        inv_ok = all(v is not None and bv is not None and v <= bv for v in seen)
        steps.append({"what": "invariant", "zh": "不变式 ι", "ok": inv_ok,
                      "detail": "%s at the call: %s (bound %s)" % (theta["expr_text"], sorted(set(seen)), bv)})
        return all(s["ok"] for s in steps), steps

    # ---- repair side: generalisation -----------------------------------------------------------------------------------
    def lift(self, theta, D):
        return {"sym": theta["sym"], "bound": D.iota["bound"], "errno": theta["body"][0].split("=")[-1].strip(" ;")
                if "=" in theta["body"][0] else None,
                "role": "a length argument of a copy whose value was written by the device (%s)" % D.iota.get("origin"),
                "pattern": "if (<device-written length> > %s) { <error exit of that function> }" % D.iota["bound"]}

    def sites(self, original=False):
        return (self.ana_orig if original else self.ana).sites()

    def instantiate(self, ts, site):
        idiom, errnos, lines = self._idiom(site["file"], site["stmt_line"])
        errno = ts["errno"] or (errnos.most_common(1)[0][0] if errnos else "-EINVAL")
        body = ["%s = %s;" % (idiom[0], errno), idiom[1]] if idiom else ["return %s;" % errno]
        return {"sym": ts["sym"], "file": site["file"], "before_line": site["stmt_line"],
                "cond": "%s > %s" % (site["len_text"], ts["bound"]), "body": body, "expr_text": site["len_text"]}

    def site_test(self, site):
        """search the declared input space for a test that reaches this site and, on the original build, fails there"""
        tg = self.cfg["testgen"]
        os.makedirs(os.path.join(self.workdir, "gen"), exist_ok=True)
        base = os.path.basename(site["file"])
        for k, (op, ln) in enumerate(itertools.product(tg["ops"], tg["device_lengths"])):
            path = os.path.join(self.workdir, "gen", "site%d_%d.scn" % (site["line"], k))
            open(path, "w").write(tg["template"].format(op=op, len=ln))
            r = self.run(self.bin_orig, path)
            if r["asan"] and any(os.path.basename(f["file"]) == base and f["line"] == site["line"] for f in r["asan"]["frames"]):
                return path, {"op": op, "len": ln, "asan": "%s %s" % (r["asan"]["bug"], r["asan"]["access"])}
        return None, None

    def drop_stopgap(self, h):
        self.undo_blackbox(h)
        return self.stable(None)

    def final_diff(self):
        out = []
        for f in self.code:
            a = open(os.path.join(self.orig, f)).read().split("\n")
            b = open(os.path.join(self.work, f)).read().split("\n")
            rel = f.split("kernel/", 1)[-1]
            out += [ln.rstrip("\n") for ln in difflib.unified_diff(a, b, "a/" + rel, "b/" + rel, lineterm="")]
        return "\n".join(out) + "\n"

    def compare_upstream(self):
        up = self.cfg.get("upstream_patch")
        if not up:
            return None
        tmp = os.path.join(self.workdir, "upstream")
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        shutil.copytree(os.path.join(self.orig, "kernel"), tmp)
        r = subprocess.run(["patch", "-p1", "-s", "-d", tmp, "-i", self._ex(up)], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            return {"applied": False, "error": r.stdout + r.stderr}
        same = all(open(os.path.join(tmp, f.split("kernel/", 1)[-1])).read() == open(os.path.join(self.work, f)).read()
                   for f in self.code)
        return {"applied": True, "identical": same, "patch": up}
