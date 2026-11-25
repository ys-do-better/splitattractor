"""Ask why along the steps that built the system (Liu, 2026-10-30), then walk back down before handing over.

A program or an agent is built in steps, and each part is made from parts made earlier:

    (1) requirements (FDE)       from the users and the scenario
    (2) domain knowledge         the periodic table of the domain
    (3) schema (working memory)  made from (2) and (1)
    (4) assumptions at the boundary, what the system takes for granted about the outside: made from (2) and (1)
    (5) logic                    made on (3) and (4)

The failure shows up in (5). At every step the walk asks one question: are the inputs of this step right?
A wrong input moves the walk to the step that made it; that move is the next "why". A step whose inputs are right
but whose output is wrong is the root: asking further up would not change it. At the boundary the input is an
assumption about the outside, so it is checked against what the outside really did. Users and the outside world are
reality, so the walk always ends at one of the five steps.

Walking back: (a) with the root step done right, does this failure go away, compared with replaying it as it is;
(b) the same step made other places as well: predict that they fail the same way, then check. Only then is the
understanding taken as right.

Agents and people answer differently from run to run (deck v17, page 6). A target that says so
(deterministic = False) is asked every question `runs` times: an answer stands only when almost all runs agree, and
this case counts as explained only when the failure is clearly rarer with the root step done right than as it is.

The questions are fixed and generic; the target answers them with evidence.
"""
from math import comb

STEPS = {  # id -> (number on the deck, name for slides, the steps it is made from)
    "fde": (1, "需求（FDE）", ()),
    "domain": (2, "领域知识（周期表）", ()),
    "schema": (3, "schema（WM）", ("domain", "fde")),
    "boundary": (4, "边界上的假设", ("domain", "fde")),
    "logic": (5, "逻辑", ("schema", "boundary")),
}
ROOT_KIND = {"logic": "wrong_rule", "boundary": "missing_rule", "schema": "missing_rule", "domain": "missing_rule",
             "fde": "missing_rule"}
SURE = 0.8    # share of runs that must give the same answer
ALPHA = 0.05  # this case: the failure must be rarer with the root step done right, at this level


def name(step):
    n, zh, _ = STEPS[step]
    return "%s %s" % ("①②③④⑤"[n - 1], zh)


def times(target):
    """how often to ask: once for a deterministic system, `runs` times for one that answers differently each run"""
    return 1 if getattr(target, "deterministic", True) else getattr(target, "runs", 5)


def poll(target, fn, ok=lambda r: r["ok"]):
    """ask the same question several times; the most common answer stands only if almost every run gives it"""
    k = times(target)
    rs = [fn() for _ in range(k)]
    oks = [ok(r) for r in rs]
    top = max(oks, key=oks.count)
    r = dict(rs[oks.index(top)])
    r.update(ok=top, agree=oks.count(top), runs=k, sure=oks.count(top) >= SURE * k)
    return r


def rarer(a, b, k):
    """one-sided Fisher exact test: b failures in k runs with the root step done right against a failures in k runs
    as it is; small means the failure is rarer when the root step is done right"""
    t = a + b
    return sum(comb(t, x) * comb(2 * k - t, k - x) for x in range(b + 1)) / comb(2 * k, k)


def _check(q, r, a):
    c = {"q": q, "ok": r["ok"], "a": a}
    if r["runs"] > 1:
        c.update(agree=r["agree"], runs=r["runs"])
    return c


def _outside(target, hops, bound):
    o = target.outside_value(hops)
    if o is None:
        return {"ok": None, "outside": None}
    return {"ok": o["value"] is not None and bound is not None and o["value"] <= bound, "outside": o}


def _unsure(steps, s, r, what):
    s["verdict"], s["go"] = "unsure", None
    s["a"] = "%s：%d 次里只有 %d 次一样，说不准 → 多跑几次、找别的证据再下结论" % (what, r["runs"], r["agree"])
    return steps, None


def walk(target, ev, hops, iota, L):
    """returns (steps, root); every step is a dict the run log and the deck can show as it is.
    root is None when the evidence does not agree across runs."""
    steps = []
    bv = iota.get("bound_value")
    # ---- (5) logic: the outside input against the assumption, the state against the schema, the logic under both
    oc = poll(target, lambda: _outside(target, hops, bv))
    fits = poll(target, lambda: target.state_fits(iota, hops))
    under = poll(target, lambda: target.logic_under(iota, L))
    outside, a_out = oc["outside"], oc["ok"]
    s = {"step": "logic", "name": name("logic"), "q": "输入都对吗？",
         "checks": [
             _check("外面给的值在假设里吗？", oc,
                    "%s给了 %s，假设是 ≤ %s" % (outside["who_zh"], outside["value"], bv) if outside else "没有来自外面的值"),
             _check("状态装得进 schema 吗？", fits, fits["zh"]),
             _check("假设成立时，逻辑对吗？", under, under["zh"])],
         "evidence": {"outside": outside, "fits": fits, "under": under}}
    steps.append(s)
    if outside is not None and not oc["sure"]:
        return _unsure(steps, s, oc, "外面给的值在不在假设里")
    if outside is not None and not a_out:
        s["verdict"], s["go"] = "input_wrong", "boundary"
        s["a"] = "外面给的值不在假设里 → 往上：造这个假设的那一步"
        cur = "boundary"
    elif not fits["sure"]:
        return _unsure(steps, s, fits, "状态装不装得进 schema")
    elif not fits["ok"]:
        s["verdict"], s["go"] = "input_wrong", "schema"
        s["a"] = "状态装不进 schema → 往上：造 schema 的那一步"
        cur = "schema"
    else:
        s["verdict"], s["go"] = "root", None
        s["a"] = "输入都对，逻辑还错 → 根在逻辑"
        return steps, _root("logic", "输入都对，这段逻辑自己错了")
    # ---- (4) boundary or (3) schema: is the output wrong against reality, and are its own inputs right?
    known = poll(target, lambda: target.rule_known(iota))
    expected = poll(target, lambda: target.case_expected(iota))
    if cur == "boundary":
        q_out = {"q": "假设对得上外面真实给的吗？", "ok": False,
                 "a": "假设 ≤ %s，%s真给了 %s" % (bv, outside["who_zh"], outside["value"])}
    else:
        q_out = {"q": "schema 装得下领域里合法的状态吗？", "ok": False, "a": fits["zh"]}
    s = {"step": cur, "name": name(cur), "q": "输出对吗？输入都对吗？",
         "checks": [q_out,
                    _check("领域知识里有这条规则吗？", known, known["zh"]),
                    _check("需求和设计预料到这种情况吗？", expected, expected["zh"])],
         "evidence": {"known": known, "expected": expected}}
    steps.append(s)
    if not known["sure"]:
        return _unsure(steps, s, known, "领域知识里有没有这条规则")
    if not known["ok"]:
        s["verdict"], s["go"] = "input_wrong", "domain"
        s["a"] = "领域知识里没有这条规则 → 往上：周期表不全"
        steps.append({"step": "domain", "name": name("domain"), "q": "这条知识周期表里有吗？", "verdict": "root",
                      "checks": [], "a": "没有：根在领域知识", "evidence": {}})
        return steps, _root("domain", "周期表里没有这条规则")
    if not expected["sure"]:
        return _unsure(steps, s, expected, "需求和设计有没有预料到")
    if not expected["ok"]:
        s["verdict"], s["go"] = "input_wrong", "fde"
        s["a"] = "需求没预料到这种情况 → 往上：需求（FDE）"
        steps.append({"step": "fde", "name": name("fde"), "q": "出事的场景需求里有吗？", "verdict": "root",
                      "checks": [], "a": "没有：根在需求", "evidence": {}})
        return steps, _root("fde", "需求里没有这个场景")
    s["verdict"], s["go"] = "root", None
    if cur == "boundary":
        s["a"] = "输入都对，输出错 → 根在这一步：规则有，但没放在边界上"
        return steps, _root("boundary", known.get("not_at_boundary_zh") or "规则有，但没放在边界上")
    s["a"] = "输入都对，输出错 → 根在这一步：schema 没把这条规则装进去"
    return steps, _root("schema", "领域里有、需求里有，schema 没装进去")


def _root(step, text_zh):
    return {"step": step, "n": STEPS[step][0], "name": name(step), "text_zh": text_zh, "kind": ROOT_KIND[step]}


def walk_back(target, ev, root, iota, L):
    """from the root back down to the symptom: this case against replaying it as it is, then the class the same step
    made"""
    k = times(target)
    as_is = [target.replay_as_is(root, iota, L) for _ in range(k)]
    fixed = [target.replay_with_root_right(root, iota, L) for _ in range(k)]
    a = sum(1 for r in as_is if r["fails"])
    b = sum(1 for r in fixed if not r["ok"])
    if k == 1:
        p, ok = None, a == 1 and b == 0
    else:
        p = rarer(a, b, k)
        ok = p < ALPHA
    case = dict(fixed[0], ok=ok, runs=k, as_is_fails=a, fixed_fails=b, p=p, as_is=as_is[0],
                zh="%s（原样重放 %d 次出错 %d 次，只改这一步 %d 次出错 %d 次%s）" % (
                    fixed[0]["zh"], k, a, k, b, "，p = %.3f" % p if p is not None else ""))
    sites = target.class_sites(root, iota, L)
    predicted = [x for x in sites if not x.get("this")]
    checked = []
    for x in predicted:
        r = poll(target, lambda: dict(zip(("test", "info"), target.class_test(x))), ok=lambda r: r["test"] is not None)
        checked.append({"site": x, "test": r["test"], "info": r["info"], "fails": r["ok"] and r["sure"],
                        "agree": r["agree"], "runs": r["runs"]})
    return {"case": case, "sites": sites, "checked": checked, "ok": ok and all(c["fails"] for c in checked)}
