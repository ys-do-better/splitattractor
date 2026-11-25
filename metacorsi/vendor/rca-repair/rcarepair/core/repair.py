"""Repair agent (deck page 5): stop-gap with the black-box row -> correction in the RCA's column -> generalisation."""
import itertools

from . import table


class RepairAgent:
    def __init__(self, ltm2, log, max_pair=2):
        self.ltm2, self.log, self.max_pair = ltm2, log, max_pair

    # ---- stage 0 ----------------------------------------------------------------------------------------------------
    def stopgap(self, target, D, ev):
        log = self.log
        running = bool(target.scenario.get("running"))
        recurring, r = target.recurring() if running else (False, None)
        log.step("repair.running", "在跑、会复发？%s" % ("是：工作负载再跑一遍，同一处又越界" if running and recurring else "否"),
                 running=running, recurring=recurring)
        if not (running and recurring):
            return None
        cands = sorted(table.cell(D.k_s, "black"), key=lambda e: (e.cost, e.shade))
        log.step("repair.stopgap.cell", "止血：症状那一列（%s）的黑盒格：%s，便宜、换得少的先试" % (
            D.k_s, " ".join(e.sym for e in cands)), order=[e.sym for e in cands])
        for e in cands:
            h = target.blackbox(e.sym, ev)
            if h is None:
                log.step("repair.stopgap.skip", "%s %s：这个系统没有对应的操作，跳过" % (e.sym, e.name_zh), sym=e.sym)
                continue
            target.apply_blackbox(h)
            ok, info = target.stable(h)
            log.step("repair.stopgap.try", "%s %s：%s → %s" % (e.sym, e.name_zh, h.get("zh", h.get("text", "")),
                                                              "好了" if ok else "没好，撤销，换大一级"),
                     sym=e.sym, handle=h, ok=ok, info=info)
            if ok:
                log.step("repair.stopgap.keep", "先留着当临时措施，再去根治", sym=e.sym)
                return h
            target.undo_blackbox(h)
        return None

    # ---- stage 1 ----------------------------------------------------------------------------------------------------
    def candidates(self, D, access):
        rows = table.rows_within(access)
        els = [e for r in rows for e in table.cell(D.k, r)]
        out = []
        for row in self.ltm2.lookup(D.k, D.signature):
            out.append({"syms": row["compound"], "from": "LTM2", "success": row["success"] / max(1, row["tries"]),
                        "shade": sum(table.ELEMENTS[s].shade for s in row["compound"]),
                        "cost": sum(table.ELEMENTS[s].cost for s in row["compound"])})
        for n in range(1, self.max_pair + 1):
            for combo in itertools.combinations(els, n):
                if n > 1 and any(e.solo for e in combo):
                    continue
                syms = [e.sym for e in combo]
                if any(c["syms"] == syms for c in out):
                    continue
                out.append({"syms": syms, "from": "new", "success": 0.0, "shade": sum(e.shade for e in combo),
                            "cost": sum(e.cost for e in combo)})
        out.sort(key=lambda c: (-c["success"], c["shade"], c["cost"]))
        return rows, out

    def correct(self, target, D, ev, access):
        log = self.log
        rows, cands = self.candidates(D, access)
        log.step("repair.correct.cell", "修正：列 = RCA 给的 %s，行 = 权限够得着的 %s" % (
            D.k, " / ".join({"black": "黑盒", "white": "白盒", "spec": "规约"}[r] for r in rows)), k=D.k, rows=rows)
        log.step("repair.correct.order", "选组合、排序（LTM ② 里 %d 个，新组合 %d 个）：%s" % (
            sum(c["from"] == "LTM2" for c in cands), sum(c["from"] == "new" for c in cands),
            "，".join("+".join(c["syms"]) for c in cands[:6])), candidates=cands)
        for c in cands:
            if len(c["syms"]) > 1:
                continue  # v1: a target offers parameter filling per element; compounds come from LTM 2 entries
            sym = c["syms"][0]
            gen = target.params(sym, D, ev)
            if gen is None:
                log.step("repair.correct.skip", "%s %s：这个系统里还没有它的填参数方法，跳过" % (sym, table.ELEMENTS[sym].name_zh), sym=sym)
                continue
            for theta in gen:
                built, err = target.apply(theta)
                if not built:
                    log.step("repair.correct.try", "%s：编译不过，换参数" % sym, theta=theta, error=err)
                    target.revert(theta)
                    continue
                ok, steps = target.validate(theta, D)
                log.step("repair.correct.try", "%s %s：%s → 验证%s" % (
                    sym, table.ELEMENTS[sym].name_zh, theta["cond"], "过了" if ok else "没过，换参数"),
                    sym=sym, theta=theta, ok=ok, steps=steps)
                if ok:
                    return sym, theta
                target.revert(theta)
        log.step("repair.correct.none", "这一格和这几行都没修好：往深一行、往暗一级", rows=rows)
        return None, None

    # ---- stage 2 ----------------------------------------------------------------------------------------------------
    def generalise(self, target, D, ev, sym, theta, stopgap):
        log = self.log
        el = table.ELEMENTS[sym]
        if el.row == "black":
            log.step("repair.general.skip", "黑盒元素不泛化")
            return []
        ts = target.lift(theta, D)
        log.step("repair.general.lift", "这一例 → 这一类：%s" % ts["pattern"], theta_star=ts)
        all_sites = target.sites(original=True)
        log.step("repair.general.sites", "找所有同类的地方（静态查询：设备写进来的值当拷贝长度、之前没查上限）：原代码里 %d 处：%s" % (
            len(all_sites), "，".join("%s:%d" % (s["file"].split("/")[-1], s["line"]) for s in all_sites)), sites=all_sites)
        fixed = []
        for s in target.sites(original=False):
            orig = next((o for o in all_sites if o["len_expr"] == s["len_expr"] and o["func"] == s["func"]), s)
            test, info = target.site_test(orig)
            th = target.instantiate(ts, s)
            built, err = target.apply(th)
            ok, steps = target.validate(th, D, extra_tests=[test] if test else [])
            log.step("repair.general.fix", "第 %s 行（原代码）：生成的测试 %s 先在原版本上越界 → 打补丁 → 验证%s" % (
                orig["line"], "「%s，长度 %s」" % (info["op"], info["len"]) if info else "（没找到）", "过了" if ok else "没过"),
                site=orig, test=test, test_info=info, theta=th, ok=ok, steps=steps)
            if ok:
                fixed.append((orig, th))
            else:
                target.revert(th)
        if stopgap is not None:
            ok, info = target.drop_stopgap(stopgap)
            log.step("repair.general.drop_stopgap", "换掉临时措施（撤掉 %s）：工作负载 %s" % (
                stopgap["sym"], "不再越界，坏设备的读返回错误，其他设备照常" if ok else "又出错了"), ok=ok, info=info)
        return fixed

    def run(self, target, D, ev):
        log = self.log
        access = target.scenario.get("access", "white")
        stopgap = self.stopgap(target, D, ev)
        sym, theta = self.correct(target, D, ev, access)
        if sym is None:
            return {"patched": False}
        fixed = self.generalise(target, D, ev, sym, theta, stopgap)
        entry = {"k": D.k, "signature": {k: D.signature.get(k) for k in ("bug", "access", "origin")},
                 "compound": [sym], "name": "%s @ device boundary" % sym if D.iota.get("origin") == "device" else sym,
                 "theta_star": target.lift(theta, D)["pattern"],
                 "sites_query": "value written by the device used as a copy length, no upper-bound check before the call",
                 "tries": 1, "success": 1, "sites_seen": [1 + len(fixed)]}
        self.ltm2.add_or_update(entry)
        log.step("repair.ltm2", "写回 LTM ②：%s（列 %s），附上泛化用的静态查询" % (entry["name"], D.k), entry=entry)
        return {"patched": True, "sym": sym, "theta": theta, "fixed_sites": [f[0] for f in fixed]}
