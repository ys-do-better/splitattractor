"""Build and run matched MiniAgent pilot arms without writing credentials."""

import argparse
import asyncio
import os
from pathlib import Path

from flashrsi.agents import MetaCoRSIMiniAgent, MiniAgent
from flashrsi.config import load_config
from flashrsi.runner import _evaluate, _prepare, _prepare_evaluation, _run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=("B0", "B1", "B2", "B3"), required=True)
    parser.add_argument(
        "--data", type=Path, help="Override the local DeepSWE data path"
    )
    parser.add_argument(
        "--warm-seed", type=Path, help="Indexed historical knowledge for B2"
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--use-config-test-split",
        action="store_true",
        help="Evaluate the same test split as B3 without learning from train",
    )
    parser.add_argument(
        "--heldout-warm-seed",
        action="store_true",
        help="Reject any known exposure for B2 or warm B3",
    )
    args = parser.parse_args()
    if args.arm == "B2" and args.warm_seed is None:
        parser.error("B2 requires --warm-seed")
    if args.warm_seed and args.arm not in {"B2", "B3"}:
        parser.error("--warm-seed can be used for B2 or B3")
    if args.arm == "B3" and args.warm_seed and not args.heldout_warm_seed:
        parser.error("A warm B3 requires --heldout-warm-seed")
    if args.arm in {"B0", "B1", "B2"} and not (args.data or args.use_config_test_split):
        parser.error("Specify a common --data or --use-config-test-split")

    key = os.environ.get("FLASHRSI_AGENT_API_KEY") or os.environ.get(
        "METACORSI_DSH_API_KEY"
    )
    if not key and not args.check_only:
        parser.error("Set FLASHRSI_AGENT_API_KEY for a real model run")
    cfg = load_config(args.config)
    cfg.agent.llm.api_key = key or "check-only-placeholder"
    if args.data:
        cfg.benchmark.data_path = str(args.data.expanduser().resolve())
    if args.warm_seed:
        cfg.agent.knowledge_dir = str(args.warm_seed.expanduser().resolve())
        cfg.agent.evaluation_scope = (
            "heldout" if args.heldout_warm_seed else "historical"
        )
    if args.arm == "B3":
        if args.check_only:
            asyncio.run(_prepare(cfg))
            print("B3 startup checks passed")
        else:
            run_dir, result = asyncio.run(_run(cfg))
            print(run_dir, result["test_score"])
        return

    if not args.use_config_test_split:
        cfg.benchmark.pop("split_counts", None)
        cfg.benchmark.split = "auto"
    if args.arm == "B0":
        cfg.agent.name = "miniagent"
        target = MiniAgent(cfg.agent)
    else:
        target = MetaCoRSIMiniAgent(cfg.agent)
    output = Path(cfg.run.output_dir) / args.arm
    seed_dir = output / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    path = seed_dir / f"{target.snapshot.id}.json"
    if not path.is_file():
        target.snapshot.save(path)
    cfg.run.output_dir = str(output)
    cfg.run.mode = "evaluate"
    cfg.run.snapshot = str(path)
    cfg = load_config(overrides=cfg)
    if args.check_only:
        _, benchmark = asyncio.run(_prepare_evaluation(cfg))
        print(f"{args.arm} startup checks passed for {len(benchmark.test)} tasks")
    else:
        run_dir, result = asyncio.run(_evaluate(cfg))
        print(run_dir, result["test_score"])


if __name__ == "__main__":
    main()
