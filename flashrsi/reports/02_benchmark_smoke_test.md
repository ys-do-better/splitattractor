# DeepSWE 冒烟证据

原有容器版 MiniAgent 对本地真实 `datacurve/abs-stepped-slices` 的
`--check-only` 通过，检查了 Docker Linux、Harbor 0.23、任务格式与 Snapshot。
容器版运行默认从容器下载安装 uv 和依赖。本次未验证其在线安装成功。

新接入对 B0 MiniAgent 和 B1 MetaCoRSI MiniAgent 运行了同一个本地真实任务，
均由 Harbor 执行 Shell 并启动独立 DeepSWE verifier。模型 API 使用
127.0.0.1 的固定响应，只调用 `pwd && git status --short` 后主动结束。
这项技术检查的目的在于验证链路和数据记录。两组都没有生成补丁。

| 组 | 工具 | 本地 API 响应 | F2P | P2P | Verifier reward |
|---|---:|---:|---:|---:|---:|
| B0 | 1 | 2 | 0/6 | 6/6 | 0 |
| B1 | 1 | 2 | 0/6 | 6/6 | 0 |

本地模拟响应的 token 字段成功进入 Trajectory 统计。这些字段不代表真实
模型用量或成本。未将模拟运行计入性能对照。

- 技术检查脚本位于 `runs/metacorsi-miniagent-pilot/offline_transport.py`。
- B0 结果位于 `runs/metacorsi-miniagent-pilot/offline/evaluation-miniagent-deepswe-20261126-102330-891699`。
- B1 结果位于 `runs/metacorsi-miniagent-pilot/offline/evaluation-metacorsi_miniagent-deepswe-20261126-102459-390923`。
- 11 月 25 日 DSH 的 DeepSeek 单题结果仍是跨执行器参考。
- 自动审批拒绝了可能把真实题面和仓库上下文发送给 DeepSeek 的 MiniAgent 探测。
  本地固定响应和只读预检已完成。实际 DeepSeek 测试和性能比较待专项授权。
