# MetaCoRSI–FlashRSI 适配验收状态

当前代码已实现同一 MiniAgent 执行器下的 MetaCoRSI 运行和知识更新接口。
已验证本地链路与两轮合成进化。真实 DeepSeek 实验及最终正式验收尚未完成。

| 验收项 | 当前状态 | 证据 |
|---|---|---|
| 两种 Target 注册、共同 MiniAgent、独立 verifier | 已做启动检查和本地 Docker 技术冒烟 | [运行报告](02_benchmark_smoke_test.md) |
| 初始检索、CLS、角色约束、RCA、Grounding | 复用原生 V23，实现运行时注入和离线检索检查 | [运行层](04_runtime_integration.md) |
| K0、K20、K28、K29 | 均通过哈希和版本校验 | [进化](05_evolution_validation.md) |
| VerifiedEpisode、连续 CLS、Snapshot 与候选隔离 | 合成成功和失败记录已验证；真实学习轮次未验证 | [测试](03_unit_and_equivalence_tests.md) |
| 独立测试集、真实成本、至少一组真实对照 | 待授权真实模型单题试验，并冻结训练/开发/测试资源 | [Pilot](06_pilot_experiment_results.md) |

原生共享蒸馏入口单独保存在 `../metacorsi` 的
`flashrsi-native-episode-api` 分支，提交 `e97fe87`。
FlashRSI 的方法与 Hosted MiniAgent 接入位于 `metacorsi-local`。
[启动命令](../docs/metacorsi.md) · [四组配置](../examples/quickstarts/deepswe_metacorsi_miniagent.yaml) · [启动脚本](../scripts/run_metacorsi_miniagent.py)。
