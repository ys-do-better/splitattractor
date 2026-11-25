"""The five method families of the RCA loop.

Each is written against generic inputs (a target, an evidence store) so that any system that supplies the evidence
can use it. Causal chain tracing asks the target to walk dependences; the others work on plain data structures.
"""
import itertools
import math
import os

from .base import Method, MethodResult


class StatisticalAssociation(Method):
    """Tarantula / Ochiai / CBI: score every unit by how often it shows up in failing vs passing runs"""
    key, row, needs, needs_zh, does_zh = "statistical", "Statistical association", "spectra", "很多次运行", "给每处打分排序"

    @staticmethod
    def ochiai(spectra):
        """spectra: {unit: (ef, ep, nf)} executed-in-failing, executed-in-passing, not-executed-in-failing"""
        out = []
        for u, (ef, ep, nf) in spectra.items():
            d = math.sqrt((ef + nf) * (ef + ep))
            out.append((u, ef / d if d else 0.0))
        return sorted(out, key=lambda x: -x[1])

    def run(self, target, ev):
        rank = self.ochiai(ev.get("spectra"))
        top = rank[0] if rank else (None, 0.0)
        unique = len(rank) < 2 or rank[1][1] < top[1]
        return MethodResult(localized=bool(rank) and unique and top[1] > 0, ranking=rank,
                            summary="top %s (%.2f)" % top, summary_zh="排第一的是 %s（%.2f）" % top)


class ModelBasedDiagnosis(Method):
    """Reiter / GDE: conflicts from the system model, minimal hitting sets are the diagnoses"""
    key, row, needs, needs_zh, does_zh = "model", "Model-based diagnosis", "component_model", "有规约、模型", "找最小冲突集"

    @staticmethod
    def minimal_hitting_sets(conflicts, max_size=3):
        comps = sorted(set().union(*conflicts)) if conflicts else []
        found = []
        for k in range(1, max_size + 1):
            for cand in itertools.combinations(comps, k):
                s = set(cand)
                if all(s & c for c in conflicts) and not any(f <= s for f in found):
                    found.append(s)
        return found

    def run(self, target, ev):
        conflicts = [set(c) for c in target.conflicts(ev)]
        diags = self.minimal_hitting_sets(conflicts)
        return MethodResult(localized=len(diags) == 1, ranking=[(sorted(d), 1.0 / len(diags)) for d in diags],
                            summary="%d minimal diagnoses" % len(diags), summary_zh="%d 个最小诊断" % len(diags))


class CausalChainTracing(Method):
    """dynamic slicing / REPT: from the violation, follow data dependences back to where the bad value came from"""
    key, row, needs, needs_zh, does_zh = "causal", "Causal chain tracing", "failing_trace", "一条失败 trace", "顺着依赖往回追"

    def run(self, target, ev):
        hops, missing = target.backtrack(ev)
        if missing is None:
            o = hops[-1]
            return MethodResult(True, hops=hops, summary="reached the origin %s" % o.expr,
                                summary_zh="追到了源头：%s" % o.expr)
        last = hops[-1] if hops else None
        return MethodResult(False, hops=hops, missing=missing,
                            summary="stuck at %s: %s" % (last.expr if last else "-", missing.get("need")),
                            summary_zh="追到 %s 就断了：不知道%s" % (last.expr if last else "-", {
                                "last_writer": "是谁写的这个值", "reaching_definition": "这个变量在哪赋的值",
                                "origin_of": "这个值从哪来"}.get(missing.get("need"), missing.get("need"))))


class ExperimentalIsolation(Method):
    """delta debugging / cause transitions: shrink the difference between a passing and a failing run"""
    key, row, needs, needs_zh, does_zh = "experimental", "Experimental isolation", "controllable_rerun", "能控制地重跑", "二分缩小差异"

    @staticmethod
    def ddmin(deltas, fails):
        """Zeller's ddmin: smallest subset of deltas for which fails(subset) is True"""
        n = 2
        while len(deltas) >= 2:
            chunk = math.ceil(len(deltas) / n)
            subsets = [deltas[i:i + chunk] for i in range(0, len(deltas), chunk)]
            reduced = False
            for sub in subsets:
                if fails(sub):
                    deltas, n, reduced = sub, 2, True
                    break
            if not reduced:
                for sub in subsets:
                    comp = [d for d in deltas if d not in sub]
                    if comp and fails(comp):
                        deltas, n, reduced = comp, max(n - 1, 2), True
                        break
            if not reduced:
                if n >= len(deltas):
                    break
                n = min(len(deltas), 2 * n)
        return deltas

    def run(self, target, ev):
        deltas, fails = target.deltas(ev)
        m = self.ddmin(deltas, fails)
        return MethodResult(localized=len(m) == 1, ranking=[(d, 1.0) for d in m],
                            summary="minimal cause %s" % m, summary_zh="最小原因：%s" % m)


class HypothesisTesting(Method):
    """AutoSD / RCAgent: propose a hypothesis, predict, run an experiment (needs a language model)"""
    key, row, needs, needs_zh, does_zh = "hypothesis", "Hypothesis testing", "llm", "都没有", "提假设、做实验"

    def applicable(self, have):
        return "llm" in have or bool(os.environ.get("RCAREPAIR_LLM"))

    def run(self, target, ev):
        return MethodResult(False, missing={"need": "llm"}, summary="no language model configured",
                            summary_zh="没有配置大模型")


# the order in which the loop prefers them when several apply: the more a method pins the cause per unit of
# evidence, the earlier (a model gives sound diagnoses; a full trace gives exact dependences; ...)
PRIORITY = [ModelBasedDiagnosis(), CausalChainTracing(), ExperimentalIsolation(), StatisticalAssociation(),
            HypothesisTesting()]
# the order in which page 4 draws them
PAGE4_ORDER = ["statistical", "model", "causal", "experimental", "hypothesis"]


def choose(have, methods=PRIORITY):
    for m in methods:
        if m.applicable(have):
            return m
    return None
