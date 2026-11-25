"""RCA agent (deck page 4): case matching -> search loop over the page-1 rows -> where -> ask up the build steps ->
walk back -> column."""
from ..methods.families import PRIORITY, choose
from . import walk, whychain
from .model import Diagnosis

HAVE_ZH = {"failing_trace": "一条失败 trace", "spectra": "很多次运行", "component_model": "规约、模型",
           "controllable_rerun": "能控制地重跑", "llm": "大模型"}


def _agree(c):
    """how many runs gave this answer, shown only when the target was asked more than once"""
    return "（%d 次里 %d 次）" % (c["runs"], c["agree"]) if c.get("runs", 1) > 1 else ""


class RCAAgent:
    def __init__(self, ltm1, log, episodes=None, methods=PRIORITY, tau=0.8, max_rounds=5):
        self.ltm1, self.log, self.episodes = ltm1, log, episodes
        self.methods, self.tau, self.max_rounds = methods, tau, max_rounds

    def run(self, target, access):
        log = self.log
        sig, ev = target.failure()
        log.step("rca.failure", "失败：%s，%s" % (sig.kind, sig.where), signal=sig.__dict__)
        similar = target.remind(sig)
        ev.episodes += similar
        log.step("rca.remind", "取出这次和相似的经历：这次 1 条，相似的 %d 条" % len(similar),
                 episodes=[e.__dict__ for e in ev.episodes])
        sigd = target.signature(sig, ev)
        case, conf = self.ltm1.match(sigd, self.tau)
        if case is not None:
            log.step("rca.case_match", "LTM ① 里见过（相似度 %.2f）→ 直接用案例里的根因" % conf, signature=sigd, case=case)
            d = case["diagnosis"]
            return Diagnosis(**dict(d, from_case=True)), ev
        log.step("rca.case_match", "LTM ① 里没见过（%d 条案例，最高相似度 %.2f）" % (len(self.ltm1.cases), conf),
                 signature=sigd, cases=len(self.ltm1.cases), best=conf)
        log.step("rca.units", "候选的单位：%s" % target.units(access), access=access)
        res, m = None, None
        for r in range(1, self.max_rounds + 1):
            have = target.in_hand(ev)
            m = choose(have, self.methods)
            if m is None:
                log.step("rca.choose", "第 %d 轮：手里什么都用不上" % r, round=r, have=sorted(have))
                break
            log.step("rca.choose", "第 %d 轮：手里有 %s → %s（%s）" % (r, "、".join(HAVE_ZH.get(h, h) for h in sorted(have)),
                                                            m.row, m.does_zh),
                     round=r, have=sorted(have), method=m.key, row=m.row)
            res = m.run(target, ev)
            hops = [h.__dict__ for h in res.hops]
            if res.localized:
                log.step("rca.localized", "定位到了：%s" % res.summary_zh, round=r, hops=hops)
                break
            log.step("rca.not_localized", "没定位到：%s" % res.summary_zh, round=r, hops=hops, missing=res.missing)
            key, got = target.measure(res.missing, ev)
            log.step("rca.measure", "取下一份证据：重放这次失败，在 %s 上设观察点 → 写它的是 %s（第 %s 行），值 %s"
                     % (got["expr"], got["writer"], got["writer_line"], got["new"]), round=r, evidence=key, got=got)
        if res is None or not res.localized:
            raise RuntimeError("RCA did not localize the fault")
        W, iota, L, delta, k, k_s, stop = whychain.build(target, ev, res.hops)
        if delta is not None:
            ev.episodes.append(delta)
            if self.episodes is not None:
                self.episodes.add(delta)
        Ld = {"file": L.file, "line": L.line, "stmt_line": L.extra["stmt_line"], "func": L.func, "call": L.extra["callee"],
              "expr": L.expr, "expr_text": L.extra.get("arg_text", L.expr)}
        log.step("rca.where", "在哪：%s:%d，%s 里把 %s 当作 %s 的长度" % (L.file, L.extra["stmt_line"], L.func,
                                                                  Ld["expr_text"], L.extra["callee"]),
                 L=Ld, delta=delta.__dict__ if delta else None, facts=[s.__dict__ for s in W])
        # ask up the steps that built the system: one question per step, are its inputs right?
        steps, root = walk.walk(target, ev, res.hops, iota, Ld)
        for st in steps:
            log.step("rca.ask", "%s：%s%s" % (st["name"], "".join("%s %s%s；" % (c["q"], c["a"], _agree(c))
                                                                for c in st["checks"]), st["a"]), step=st)
        if root is None:
            log.step("rca.root", "根：说不准，%s" % steps[-1]["a"], root=None)
            raise RuntimeError("the answers disagree across runs at %s; more evidence is needed" % steps[-1]["step"])
        log.step("rca.root", "根：%s，%s" % (root["name"], root["text_zh"]), root=root)
        back = walk.walk_back(target, ev, root, iota, Ld)
        log.step("rca.back.case", "往回推，这一例：%s" % back["case"]["zh"], case=back["case"])
        log.step("rca.back.class", "往回推，这一类：同一步造的地方 %d 处（%s）；预测其余 %d 处也会错 → %s" % (
            len(back["sites"]), "、".join("第 %d 行" % x["line"] for x in back["sites"]), len(back["checked"]),
            "；".join("第 %d 行：生成的测试在原版本上%s%s" % (c["site"]["line"], "也出错" if c["fails"] else "没出错", _agree(c))
                     for c in back["checked"]) or "没有别处"),
                 sites=back["sites"], checked=back["checked"], ok=back["ok"])
        if not back["ok"]:
            raise RuntimeError("walking back from %s does not explain the failure" % root["step"])
        k = whychain.KIND_COLUMN[root["kind"]]
        log.step("rca.column", "根那一步 → %s；症状 → %s" % (k, k_s), k=k, k_s=k_s, iota=iota)
        sigd = dict(sigd, origin=iota.get("origin"))
        D = Diagnosis(L=Ld, k=k, k_s=k_s, W=[s.__dict__ for s in W], iota=iota, conf=1.0, method=m.row,
                      signature=sigd, chain=[h.__dict__ for h in res.hops], root=root, walk=steps, back=back)
        self.ltm1.add({"signature": target.signature(sig, ev), "diagnosis": D.to_json()})
        log.step("rca.record", "root cause 记录：在哪、错在哪一步（%s）、哪类知识（%s）、这一类（%d 处）；写回 LTM ①，交给 Repair agent"
                 % (root["name"], k, len(back["sites"])), ltm1_size=len(self.ltm1.cases))
        return D, ev
