"""MetaCoRSI's verifier-guided memory evolution over FlashRSI Snapshots."""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from flashrsi.agents.deepseek_harness import DeepSeekHarness
from flashrsi.agents.metacorsi import MetaCoRSITarget, copy_active_memory
from flashrsi.agents.metacorsi.miniagent import MetaCoRSIMiniAgent
from flashrsi.algorithms.metacorsi.episode_adapter import verified_episode
from flashrsi.benchmarks import DeepSWE
from flashrsi.core import Algorithm


class MetaCoRSI(Algorithm):
    name = "metacorsi"
    description = (
        "Distill DeepSWE verifier episodes into complementary long-term memory."
    )
    requires_worker = False
    evolve_scope = {"memory"}
    observation_scope = {"config"}
    _defaults = {"max_steps": 1, "evolution_policy": "grounding_feedback"}
    _policies = {
        "standard",
        "positive_only",
        "quarantine_negative",
        "grounding_feedback",
    }
    _ranges = {"max_steps": (int, 1)}

    def _check(self):
        if not isinstance(
            self.target, (MetaCoRSITarget, MetaCoRSIMiniAgent, DeepSeekHarness)
        ) or not isinstance(self.benchmark, DeepSWE):
            raise ValueError(
                "MetaCoRSI requires a MetaCoRSI or DeepSeek Harness Target and benchmark.name: deepswe"
            )
        if isinstance(self.target, DeepSeekHarness):
            self.target.config["enable_memory"] = True
        if self.config.evolution_policy not in self._policies:
            raise ValueError("Unsupported native MetaCoRSI evolution policy")
        if not all((self.benchmark.train, self.benchmark.dev, self.benchmark.test)):
            raise ValueError("MetaCoRSI requires non-empty train, dev and test splits")

    async def evolve(self):
        if "harbor" not in self.target.config:
            self.target.config["harbor"] = {}
        options = self.target.config["harbor"]
        if options.get("artifacts_dir"):
            return await self._evolve_memory()
        with TemporaryDirectory(prefix="flashrsi-metacorsi-evidence-") as temporary:
            options["artifacts_dir"] = temporary
            try:
                return await self._evolve_memory()
            finally:
                del options["artifacts_dir"]

    async def _evolve_memory(self):
        current = self.target.snapshot
        hosted = isinstance(self.target, (MetaCoRSIMiniAgent, DeepSeekHarness))
        if hosted:
            await self.target.check_native(
                forbidden_tasks=(*self.benchmark.dev, *self.benchmark.test)
            )
        baseline = None
        for step in range(1, self.config.max_steps + 1):
            training = await self._evaluate(
                current, self.benchmark.train, label="train"
            )
            if hosted:
                evidence = {
                    "episodes": [
                        episode
                        for item in training
                        if (episode := verified_episode(item, current.id)) is not None
                    ],
                    "policy": self.config.evolution_policy,
                }
                has_evidence = bool(evidence["episodes"])
            else:
                evidence = {
                    "trials": [
                        item.messages[-1]["trial_dir"]
                        for item in training
                        if item.status == "finished"
                        and item.messages
                        and item.messages[-1]["type"] == "HarborVerifier"
                    ]
                }
                has_evidence = bool(evidence["trials"])
            if not has_evidence:
                raise ValueError(
                    "No independently verified training episodes with complete evidence"
                )
            with current.check_out() as checkout, TemporaryDirectory() as temporary:
                memory = checkout / "memory"
                receipt = await self.target.native(
                    "evolve", {"memory": str(memory), **evidence}
                )
                quarantine_path = memory / "candidates.json"
                quarantine = (
                    json.loads(quarantine_path.read_bytes())
                    if hosted and quarantine_path.is_file()
                    else {}
                )
                staged = Path(temporary) / "memory"
                copy_active_memory(memory, staged)
                shutil.rmtree(memory)
                shutil.copytree(staged, memory)
                candidate = self._commit(current, checkout)
            if hosted:
                receipt["memory"] = await self.target.check_native(
                    snapshot=candidate,
                    forbidden_tasks=(*self.benchmark.dev, *self.benchmark.test),
                )
                if quarantine and self._run_dir is not None:
                    directory = self._run_dir / "quarantine"
                    directory.mkdir(mode=0o700, exist_ok=True)
                    file = directory / f"{step:04d}.json"
                    file.touch(mode=0o600, exist_ok=False)
                    # These patches never enter Snapshot resources or the tool container.
                    file.write_text(
                        json.dumps(
                            {
                                "parent_snapshot_id": current.id,
                                "candidate_snapshot_id": candidate.id,
                                "memory_version": receipt["memory"]["version"],
                                "task_candidates": quarantine,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
            self._record(
                "memory-evolution",
                {"step": step, "receipt": receipt, "candidate_id": candidate.id},
            )
            if candidate.id == current.id:
                break
            if baseline is None:
                baseline = await self._evaluate(
                    current, self.benchmark.dev, label="dev-incumbent"
                )
                if any(item.status != "finished" for item in baseline):
                    raise ValueError(
                        "Cannot validate MetaCoRSI retention: incumbent dev trial failed"
                    )
            results = await self._evaluate(
                candidate, self.benchmark.dev, label="dev-candidate"
            )
            accepted = all(item.status == "finished" for item in results) and all(
                results.per_task[task_id] >= score
                for task_id, score in baseline.per_task.items()
            )
            current = self._promote(
                current,
                candidate if accepted else current,
                evaluated_candidate_id=candidate.id,
                incumbent_score=baseline.score,
                candidate_score=results.score,
                step=step,
            )
            if accepted:
                baseline = results
        return current
