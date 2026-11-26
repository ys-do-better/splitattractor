# 单元测试与一致性验证

以下为本次覆盖 MiniAgent、原生 DSH 适配、DeepSWE、算法、Launcher、
CLI、Snapshot、最终回答和技能的命令。实际结果为 104 项通过、4 个子测试通过、
1 项未选中。未选中的是需要另一套模拟器资源的 Tau 启动器测试。

```bash
.venv/bin/python -m pytest -o addopts='' -q \
  -k 'not tau_data_uses_same_revision_as_installed_simulator' \
  tests/test_metacorsi_miniagent.py tests/test_metacorsi.py tests/test_deepswe.py \
  tests/test_algorithm.py tests/test_algorithm_migration.py \
  tests/test_snapshot_evaluation.py tests/test_cli_args.py tests/test_generic_launcher.py \
  tests/test_miniagent_final_answer.py tests/test_miniagent_skills.py
```

原生 MetaCoRSI 从 `../metacorsi/metacorsi` 运行原测试共 75 项，全部通过。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='.:vendor/rca-repair' \
  ../../baseline/metacorsi/.venv/bin/python -m pytest -o addopts='' -q -p no:cacheprovider
```

新增测试覆盖凭据与 Shell 执行分离所依赖的资源约束、可信 verifier 字段、
缺失分项拒绝蒸馏、确定性 K0、两轮更新后的 CLS、历史候选隔离和 FlashRSI
训练循环不读取 dev/test 作为蒸馏输入。K1 与 K2 分别通过了下一次原生 V23
同题检索。测试发现 `datacurve/` 与原生短任务 ID 不一致，修正位于转换层，
原生匹配算法未变。K20、CLS K28、K29 的原子分层校验也通过。
