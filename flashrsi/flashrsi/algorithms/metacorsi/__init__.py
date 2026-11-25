"""MetaCoRSI's verifier-guided memory evolution over FlashRSI Snapshots."""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from flashrsi.agents.metacorsi import MetaCoRSITarget, copy_active_memory
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
    _defaults = {"max_steps": 1}
    _ranges = {"max_steps": (int, 1)}

    def _check(self):
        if not isinstance(self.target, MetaCoRSITarget) or not isinstance(
            self.benchmark, DeepSWE
        ):
            raise ValueError(
                "MetaCoRSI requires agent.name: metacorsi and benchmark.name: deepswe"
            )
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
        baseline = None
        for step in range(1, self.config.max_steps + 1):
            training = await self._evaluate(
                current, self.benchmark.train, label="train"
            )
            trials = [
                item.messages[-1]["trial_dir"]
                for item in training
                if item.status == "finished"
                and item.messages[-1]["type"] == "HarborVerifier"
            ]
            if not trials:
                raise ValueError(
                    "No independently verified training episodes; inspect native trial logs"
                )
            with current.check_out() as checkout, TemporaryDirectory() as temporary:
                memory = checkout / "memory"
                receipt = await self.target.native(
                    "evolve", {"memory": str(memory), "trials": trials}
                )
                staged = Path(temporary) / "memory"
                copy_active_memory(memory, staged)
                shutil.rmtree(memory)
                shutil.copytree(staged, memory)
                candidate = self._commit(current, checkout)
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
