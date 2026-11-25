from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RoutedSkill:
    skill_id: str
    procedure: str
    reason: str
    confidence: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _Rule:
    skill_id: str
    pattern: re.Pattern[str]
    procedure: str


class K7SkillRouter:
    """Auditable K7 semantic routing derived from the frozen handoff."""

    _RULES = (
        _Rule("coding.register_runtime_defaults_at_source",
              re.compile(r"\b(schedule|interval|cron|configuration|config|default)\b", re.I),
              "Register defaults at the authoritative configuration source, not only as a caller fallback."),
        _Rule("coding.propagate_recursive_schema_context",
              re.compile(r"json[ -]?schema|\$ref|\$defs|recurs", re.I),
              "Propagate root definitions and traversal state through nested properties; distinguish cycle termination from false."),
        _Rule("coding.propagate_taint_through_named_expressions",
              re.compile(r"\b(taint|walrus|named expression|saniti[sz]er)\b", re.I),
              "Treat named expressions as assignment plus value propagation; only specified sanitizers terminate taint."),
        _Rule("coding.preserve_shorthand_tokens_and_layer_separators",
              re.compile(r"\b(css|shorthand|border|background)\b", re.I),
              "Preserve parsed shorthand tokens and the repository's canonical separator for multiple layers."),
        _Rule("coding.model_html_implicit_table_closure",
              re.compile(r"\b(html|table|tr|td|th|implicit close)\b", re.I),
              "Model implicit row/cell closure before opening a sibling and preserve repeated siblings as arrays."),
    )

    _SAFETY = RoutedSkill(
        "coding.preserve_best_candidate_before_targeted_repair",
        "Checkpoint the best focused-green candidate before one RCA-authorized repair; restore it on capability loss.",
        "K7 global candidate-safety invariant",
        1.0,
    )

    def route(self, task_text: str) -> tuple[RoutedSkill, ...]:
        selected = [self._SAFETY]
        for rule in self._RULES:
            match = rule.pattern.search(task_text)
            if match:
                selected.append(RoutedSkill(rule.skill_id, rule.procedure,
                                            f"public task text matched {match.group(0)!r}", .95))
        return tuple(selected)

    def prompt(self, task_text: str, include_validation: bool = True) -> str:
        lines = ["K7 FAILURE-ROUTED KNOWLEDGE (hypotheses only; terminal evidence is authoritative):"]
        lines.extend(f"- {item.skill_id}: {item.procedure} [reason={item.reason}]"
                     for item in self.route(task_text))
        if include_validation:
            lines.extend((
                "K7 VALIDATION: T0 reachability -> T1 focused/boundary -> T2 impacted preservation -> T3 full only for broad/high-risk final promotion.",
                "Any source mutation invalidates earlier evidence. Keep validation stages separate.",
                "Convergence checkpoints: skeleton by 24, focused by 50, deliverable by 62, freeze at 70; no MH acceptance and no global contract-completion pressure.",
            ))
        else:
            lines.append(
                "RCA-ONLY CONTROL: use the causal walk and walk-back; no Meta-CoRSI regression or promotion gate is active."
            )
        return "\n".join(lines)
