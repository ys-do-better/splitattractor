"""From 'where' to 'why': ask why up the causal chain until the answer no longer changes whether this class of bug
happens, then read the knowledge column off the top (repair column) and the bottom (symptom column).

The questions are fixed and generic; the answers come from the target:
  1. what went wrong at the violation                    -> the bad value       (bottom)
  2. what checked the value on its way to the use        -> checks that cannot catch it
  3. is the expectation written anywhere, and when        -> a check that runs too late
  4. where does the value come from                       -> its origin; outside the system = stop
"""
from .model import Episode, WhyStep

# kind of the step -> column of the periodic table (pages 1-3)
KIND_COLUMN = {
    "bad_value": "Factual", "wrong_constant": "Factual", "wrong_config": "Factual", "bad_state": "Factual",
    "untrusted_source": "Conceptual", "missing_rule": "Conceptual", "wrong_rule": "Conceptual",
    "insufficient_guard": "Conceptual", "wrong_ownership": "Conceptual",
    "late_check": "Sequential", "missing_step": "Sequential", "extra_step": "Sequential", "wrong_order": "Sequential",
    "race": "Concurrent", "interleaving": "Concurrent",
    "nondeterministic": "Probabilistic", "learned": "Probabilistic",
}


def build(target, ev, hops):
    L = target.localization(hops)
    viol = hops[0]
    bound = next((h for h in hops if h.kind == "bound"), None)
    write = next((h for h in hops if h.kind == "write"), None)
    origin = hops[-1] if hops[-1].kind == "origin" else None
    region = viol.extra.get("region") or {}
    steps = []
    steps.append(WhyStep(
        "bad_value",
        "%s writes past a %s-byte buffer; the length is %s" % (viol.func, region.get("size"), bound.value if bound else "?"),
        "越界写：长度 %s，缓冲区只有 %s 字节" % (bound.value if bound else "?", region.get("size")),
        "%s:%d" % (viol.file, viol.line),
        {"region": region, "index": viol.extra.get("index"), "index_value": viol.value}))
    guards = target.guards_on(hops)
    weak = [g for g in guards if not g["upper_bound"]]
    if guards and not any(g["upper_bound"] for g in guards):
        g = weak[0]
        steps.append(WhyStep(
            "insufficient_guard", "%s only rejects %s" % (g["func"], g["cond"]),
            "驱动只拒绝 %s" % g["cond"], "%s:%d" % (g["file"], g["line"]), {"guards": guards}))
    late = target.later_checks(hops, L, ev) if L is not None else []
    if late:
        c = late[0]
        steps.append(WhyStep(
            "late_check", "%s checks %s, but only after the transfer at line %d has returned" % (c["func"], c["cond"], c["after_line"]),
            "核心层的检查在写完之后", "%s:%d" % (c["file"], c["line"]), {"check": c}))
    stop = None
    if origin is not None and origin.extra.get("kind") == "external":
        stop = "above this is the %s itself, outside the system: why it sent %s does not change whether this bug class happens" % (
            origin.extra.get("origin"), origin.value)
        steps.append(WhyStep(
            "untrusted_source",
            "the length comes from the %s (%s) and is used without being treated as untrusted" % (origin.extra.get("origin"), origin.func),
            "设备给的长度没当不可信", "%s:%d" % (origin.file, origin.line),
            {"source": origin.func, "origin": origin.extra.get("origin"), "stop": stop}))
    elif origin is not None and origin.extra.get("kind") == "constant":
        steps.append(WhyStep("wrong_constant", "the length is a constant (%s)" % origin.expr,
                             "长度是写死的常量 %s" % origin.expr, "%s:%d" % (origin.file, origin.line), {}))
    k, k_s = KIND_COLUMN[steps[-1].kind], KIND_COLUMN[steps[0].kind]
    if late:
        bnd, bv, frm = late[0]["bound"], late[0]["bound_value"], "the later check %s:%d" % (late[0]["file"], late[0]["line"])
    else:
        bnd = str(region.get("size", 0) - 1)
        bv, frm = int(bnd), "the buffer capacity"
    iota = {"expr": L.expr if L else None, "op": "<=", "bound": bnd, "bound_value": bv, "from": frm,
            "late_check": late[0] if late else None, "origin": origin.extra.get("origin") if origin else None,
            "text": "%s <= %s" % (L.extra.get("arg_text", L.expr) if L else "?", bnd)}
    delta = None
    if write is not None:
        delta = Episode(o="the length byte read from the %s" % (origin.extra.get("origin") if origin else "source"),
                        a="replay with a watchpoint on it", f="written by %s" % write.func,
                        t="%s:%d" % (write.file, write.line), expected="<= %s (%s)" % (bnd, bv), actual=write.value,
                        delta="%s > %s" % (write.value, bv) if bv is not None and write.value is not None and write.value > bv else None)
    return steps, iota, L, delta, k, k_s, stop
