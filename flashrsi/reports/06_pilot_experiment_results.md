# Pilot 对照的当前证据

固定的技术冒烟任务为 `datacurve/abs-stepped-slices`。B0、B1、B2 的单题
配置和 B3 的开发集合分割均通过启动检查。[技术冒烟](02_benchmark_smoke_test.md)
只有 B0、B1 使用本地固定响应完成了工具和测试链路，未产生真实模型对照成绩。

| 组 | 执行器 | 起点与更新 | 真实模型成绩 |
|---|---|---|---|
| B0 | Hosted MiniAgent | 无 MetaCoRSI | N/A |
| B1 | Hosted MiniAgent | K0 通用先验，冻结 | N/A |
| B2 | Hosted MiniAgent | 合法历史包，冻结 | N/A |
| B3 | Hosted MiniAgent | K0 或可审计的合法起点，训练集进化 | N/A |

对 B0 与 B1，使用同一任务、模型、工具、Git 提交要求、verifier 和外层上限。
与 B3 配对时为冻结组使用 `--use-config-test-split` 并使用相同的配置和数据路径。
B2 和 warm B3 需明确记录初始包和来源。K29 与其来源任务仅做历史能力验证。
原来提出的 DeepSWE-20 中 3 道历史通过、2 道历史失败是开发抽样，后续五题仍需
根据本地与正式任务镜像的可用性完成验证。
