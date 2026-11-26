# MiniAgent Memory Runtime

`metacorsi_miniagent` 是新 Target。`metacorsi` 仍可通过原生 DSH 运行。
通过 `agent.harbor.hosted_miniagent`，基础 MiniAgent 也可以使用相同的执行方式。

- FlashRSI 的 MiniAgent ReAct 循环、选定的 Snapshot、模型 SDK、API 凭据和
  结构化记忆都留在宿主机。共同的 Shell 工具调用交给 Harbor 的任务容器。
- 任务进程保留 DeepSWE 的无网络配置，verifier 的独立容器和补丁 collect
  脚本继续由 Harbor 管理。共同路径要求 Agent 按题目提交 Git，宿主不自动提交补丁。
- 宿主只执行经本机 MiniAgent 资源逐字节校验的 runtime、配置和工具。
  Prompt 和文本技能可以由相应基线修改。修改可执行工具或配置的方法应使用原有容器
  路径，不能在本宿主执行方式下直接比较。
- 起始证据沿用原生只读文件列表，原生 V23 检索只执行一次。载入版本、原生
  Working Memory、Grounding 和失败码触发的 Deep RCA 都记录在轨迹中。
- 24+6 映射到 MiniAgent 模型轮次。通过可识别的测试命令才允许内层预算
  延长，配置中的共同上限仍生效。一轮 MiniAgent 与一步 DSH 的工具语义不同。
- MiniAgent 供应商返回的 usage、工具调用次数、终止原因可审计。知识注入
  单独记录字节数，未精确拆分其输入 token 或网关费用。
- 每次运行持有本次上下文和冻结 Snapshot。宿主运行串行化以隔离 MiniAgent
  的 SDK 全局配置。运行时诊断与推理不会直接写回记忆。
