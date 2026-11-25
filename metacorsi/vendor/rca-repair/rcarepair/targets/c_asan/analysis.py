"""Program facts for C code: the causal chain behind a sanitizer report, guards on the chain, checks that come too
late, the origin of the bad value, and the static query that finds every site of the same kind.

Everything is driven by the code and by the knowledge file (where outside data enters, which calls copy LEN bytes);
nothing here names the example.
"""
import os
import re

from ...core.model import Hop
from .cast import CFile, norm, upper_bounds

IDENT = re.compile(r"^[A-Za-z_]\w*$")


def plain(cond):
    """condition text as written, without the outer parentheses of the if"""
    c = " ".join(cond.split())
    return c[1:-1].strip() if c.startswith("(") and c.endswith(")") else c


def strip_addr(e):
    e = norm(e)
    while e.startswith("&"):
        e = norm(e[1:])
    return e


def base_prefixes(e):
    """msgs[i+1].buf[0] -> [msgs[i+1].buf[0], msgs[i+1].buf, msgs[i+1], msgs]"""
    e = strip_addr(e)
    out = [e]
    while True:
        m = re.match(r"^(.*)(\[[^\[\]]*\]|\.[A-Za-z_]\w*|->[A-Za-z_]\w*)$", e)
        if not m or not m.group(1):
            break
        e = m.group(1)
        out.append(e)
    return out


def buffer_of(arg):
    """the buffer a call writes into: &(msgs[i+1].buf[1]) -> msgs[i+1].buf, msgs[i].buf -> msgs[i].buf"""
    e = strip_addr(arg)
    m = re.match(r"^(.*)\[[^\[\]]*\]$", e)
    return m.group(1) if m and arg.strip().startswith("&") else e


def subexprs(e):
    """tainted-candidate subexpressions of a length expression: msg[1].buf[0] + 1 -> [msg[1].buf[0] + 1, msg[1].buf[0]]"""
    e = norm(e)
    parts = [e] + [p for p in re.split(r"[+\-*/]", e) if p and p != e]
    return parts


class Analysis:
    def __init__(self, root, code_files, knowledge, include_dirs):
        self.root = root
        self.code_files = code_files
        self.k = knowledge
        self.include_dirs = include_dirs
        self._cache = {}

    def cfile(self, path):
        path = os.path.realpath(path)
        if path not in self._cache:
            self._cache[path] = CFile(path)
        return self._cache[path]

    def reset(self):
        self._cache = {}

    def is_code(self, path):
        if not path:
            return False
        rp = os.path.realpath(path)
        return any(rp == os.path.realpath(os.path.join(self.root, f)) for f in self.code_files)

    def rel(self, path):
        return os.path.relpath(os.path.realpath(path), os.path.realpath(self.root))

    # ---- causal chain tracing: walk back from the violation along data dependences ----------------------------
    def backtrack(self, report, frames, values, memdeps):
        """frames: gdb stack at the report (index = gdb frame number); values: {(frame, expr): value};
        memdeps: {addr: writer record}. Returns (hops, missing, queries) where queries are values still needed."""
        code = [(i, f) for i, f in enumerate(frames) if self.is_code(f["file"])]
        if not code:
            return [], {"need": "code_frames"}, []
        need = []

        def val(fi, expr):
            key = (fi, norm(expr))
            if key not in values:
                need.append({"frame": fi, "expr": expr})
                return None
            v = values[key]
            return v.get("int", v.get("ptr", v.get("str")))

        fi, f0 = code[0]
        cf = self.cfile(f0["file"])
        stmt = cf.statement_at(f0["line"])
        lhs_rhs = cf.store_target(stmt) if stmt is not None else None
        hops = []
        region = report.get("region") or {}
        if lhs_rhs is None:
            return hops, {"need": "store_statement", "where": f0}, need
        lhs, _ = lhs_rhs
        arr, idx = cf.subscript_index(lhs)
        hops.append(Hop("violation", f0["func"], self.rel(f0["file"]), f0["line"], norm(lhs) + " = ...",
                        value=val(fi, idx) if idx else None,
                        note="%s %s past a %s-byte object" % (report.get("access", "access").lower(),
                                                              report.get("bug"), region.get("size", "?")),
                        extra={"region": region, "index": idx, "array": arr}))
        bound, loop_line = cf.loop_bound(stmt, idx) if idx else (None, None)
        if bound is None:
            return hops, {"need": "length_operand", "where": f0}, need
        hops.append(Hop("bound", f0["func"], self.rel(f0["file"]), loop_line, bound, value=val(fi, bound),
                        note="loop bound of the write"))
        expr, k = bound, 0
        while True:
            fi, fr = code[k]
            cf = self.cfile(fr["file"])
            fn = cf.function_at(fr["line"])
            e = norm(expr)
            if IDENT.match(e) and e in fn.params and k + 1 < len(code):
                pidx = fn.params.index(e)
                ci, caller = code[k + 1]
                ccf = self.cfile(caller["file"])
                call = ccf.call_at(caller["line"], fn.name)
                if call is None or pidx >= len(call.args):
                    return hops, {"need": "call_site", "where": caller}, need
                arg = call.args[pidx]
                stmt_line = ccf.statement_of(call.node).start_point[0] + 1
                hops.append(Hop("arg", caller["func"], self.rel(caller["file"]), call.line, norm(arg),
                                value=val(ci, arg), note="passed as %s of %s" % (e, fn.name),
                                extra={"callee": fn.name, "param": e, "arg_index": pidx, "stmt_line": stmt_line,
                                       "frame": ci, "arg_text": arg.strip()}))
                expr, k = arg, k + 1
                continue
            if IDENT.match(e):
                return hops, {"need": "reaching_definition", "expr": e, "where": fr}, need
            # a memory read: who wrote this location last?
            addr = val(fi, "&(%s)" % e)
            if addr is None:
                return hops, {"need": "address", "expr": e}, need
            w = memdeps.get(str(addr))
            if w is None:
                return hops, {"need": "last_writer", "expr": e, "addr": addr, "start": fr["func"],
                              "where": "%s:%d" % (self.rel(fr["file"]), fr["line"])}, need
            wf = w["frames"][0]
            wcf = self.cfile(wf["file"])
            wline = wf.get("write_line") or wf["line"]
            wst = wcf.statement_at(wline)
            store = wcf.store_target(wst) if wst is not None else None
            via = [{"func": x["func"], "file": self.rel(x["file"]), "line": x["line"]}
                   for x in w["frames"][1:] if self.is_code(x["file"])]
            hops.append(Hop("write", wf["func"], self.rel(wf["file"]), wline,
                            (store[0] + " = " + store[1]) if store else "?", value=w["new"],
                            note="last write to %s before the use" % e, extra={"via": via, "addr": addr}))
            rhs = store[1] if store else ""
            src = self.source_of(rhs)
            if src:
                hops.append(Hop("origin", src["func"], hops[-1].file, wline, norm(rhs), value=w["new"],
                                note="data from %s" % src["origin"], extra=src))
                return hops, None, need
            return hops, {"need": "origin_of", "expr": rhs}, need

    def source_of(self, rhs):
        m = re.match(r"^\s*([A-Za-z_]\w*)\s*\(", rhs or "")
        if m and m.group(1) in self.k.get("sources", {}):
            s = self.k["sources"][m.group(1)]
            if "returns" in s:
                return {"func": m.group(1), "origin": s["returns"], "kind": "external"}
        if re.match(r"^\s*-?(0x[0-9a-fA-F]+|\d+)\s*$", rhs or ""):
            return {"func": None, "origin": "constant", "kind": "constant"}
        return None

    @staticmethod
    def localization(hops):
        """L: the call where the value read from memory is first used as a length (the last arg hop)"""
        args = [h for h in hops if h.kind == "arg"]
        return args[-1] if args else None

    # ---- facts for the why chain ------------------------------------------------------------------------------
    def guards_on(self, hops):
        """checks on the tracked value inside each function it passes through, before its use"""
        out = []
        for h in hops:
            if h.kind != "arg":
                continue
            callee = h.extra["callee"]
            for path in self.code_files:
                cf = self.cfile(os.path.join(self.root, path))
                fn = cf.functions.get(callee)
                if fn is None:
                    continue
                p = h.extra["param"]
                for g in cf.guards(fn):
                    mentions = [c for c in g.comparisons if p in (c[0], c[2])]
                    if mentions:
                        out.append({"func": callee, "file": path, "line": g.line, "cond": plain(g.cond),
                                    "param": p, "upper_bound": bool(upper_bounds(g.comparisons, p))})
        return out

    def late_check_candidates(self, frames, L_frame_index):
        """upper-bound checks that run after the call in each caller above L (static part)"""
        out = []
        for fi in range(L_frame_index + 1, len(frames)):
            fr = frames[fi]
            if not self.is_code(fr["file"]):
                continue
            cf = self.cfile(fr["file"])
            fn = cf.function_at(fr["line"])
            if fn is None:
                continue
            exits = {e[3]: e for e in cf.error_exits(fn)}
            for g in cf.guards_after(fn, fr["line"]):
                for lhs, op, rhs in g.comparisons:
                    if op in (">", ">="):
                        expr, bound = lhs, rhs
                    elif op in ("<", "<="):
                        expr, bound = rhs, lhs
                    else:
                        continue
                    if not re.search(r"[\[\.]|->", expr):  # only memory reads can be the same location
                        continue
                    out.append({"frame": fi, "func": fn.name, "file": self.rel(fr["file"]), "line": g.line,
                                "after_line": fr["line"], "cond": plain(g.cond), "expr": expr, "op": op,
                                "bound": bound, "errno": exits.get(g.line, (None, None))[1], "exits": g.exits})
        return out

    def bound_value(self, bound):
        if re.match(r"^\d+$", bound):
            return int(bound)
        return self.resolve_define(bound)

    def resolve_define(self, name):
        from .cast import resolve_define
        return resolve_define(self.include_dirs, name)

    # ---- the static query of the generalisation step -----------------------------------------------------------
    def sink_summaries(self):
        sinks = {n: (s["dst_arg"], s["len_arg"]) for n, s in self.k.get("sinks", {}).items()}
        writers = {n: s["writes_arg"] for n, s in self.k.get("sources", {}).items() if "writes_arg" in s}
        funcs = {}
        for path in self.code_files:
            cf = self.cfile(os.path.join(self.root, path))
            for name, fn in cf.functions.items():
                funcs[name] = (path, cf, fn)
        changed = True
        while changed:
            changed = False
            for name, (path, cf, fn) in funcs.items():
                for call in cf.calls(fn):
                    if call.callee in sinks and name not in sinks:
                        d, l = sinks[call.callee]
                        if l < len(call.args) and d < len(call.args):
                            la, da = norm(call.args[l]), base_prefixes(call.args[d])[-1]
                            if la in fn.params and da in fn.params and not self._guarded(cf, fn, call.line, la):
                                sinks[name] = (fn.params.index(da), fn.params.index(la))
                                changed = True
                    if call.callee in writers and name not in writers:
                        w = writers[call.callee]
                        if w < len(call.args):
                            da = base_prefixes(call.args[w])[-1]
                            if da in fn.params:
                                writers[name] = fn.params.index(da)
                                changed = True
        return sinks, writers, funcs

    def _guarded(self, cf, fn, line, expr):
        for g in cf.guards(fn, before_line=line):
            for sub in subexprs(expr):
                if upper_bounds(g.comparisons, sub) and g.exits:
                    return True
        return False

    def sites(self):
        """every call that uses a value written by the device as the length of a copy, with no upper-bound check
        on it before the call"""
        sinks, writers, funcs = self.sink_summaries()
        returns = {n for n, s in self.k.get("sources", {}).items() if "returns" in s}
        out = []
        for name, (path, cf, fn) in funcs.items():
            calls = sorted(cf.calls(fn), key=lambda c: c.line)
            tainted = {}
            for call in calls:
                if call.callee in sinks and name not in self.k.get("sinks", {}):
                    d, l = sinks[call.callee]
                    if l < len(call.args):
                        le = call.args[l]
                        hit = None
                        for sub in subexprs(le):
                            for pre in base_prefixes(sub)[1:]:  # most specific buffer first
                                if pre in tainted and tainted[pre] < call.line:
                                    hit = (sub, pre, tainted[pre])
                                    break
                            if hit is None and returns and re.search(r"\b(%s)\s*\(" % "|".join(map(re.escape, returns)), sub):
                                hit = (sub, "call", call.line)
                            if hit:
                                break
                        if hit and not self._guarded(cf, fn, call.line, le):
                            st = cf.statement_of(call.node)
                            out.append({"file": path, "func": name, "line": call.line,
                                        "stmt_line": st.start_point[0] + 1, "call": call.callee,
                                        "len_expr": norm(le), "len_text": le.strip(), "tainted_by_line": hit[2],
                                        "tainted_buffer": hit[1]})
                if call.callee in writers:
                    w = writers[call.callee]
                    if w < len(call.args):
                        tainted[buffer_of(call.args[w])] = call.line
        return sorted(out, key=lambda s: (s["file"], s["line"]))
