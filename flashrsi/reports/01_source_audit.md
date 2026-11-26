# MetaCoRSI 源码审计与版本基线

审计日期为 2026-11-26。开发目录为本地 FlashRSI，起点为干净的
`metacorsi-local` 工作树，提交 `8ce6a02`，其上游基线为 `4ef9946`。
`../metacorsi` 是另一个干净的合并仓库，HEAD 为 `00d636b`，包含
`a42eb3c` 的原生 MetaCoRSI 和从 `8ce6a02` 导入的 FlashRSI。实际既有
适配的 `agent.project_dir` 指向无独立 Git 元数据的 `../MetaCogRSI-main`。
当两个副本存在差异时，原生算法核对以对应该版本的历史代码及实际执行链为准。
FlashRSI 环境为 Python 3.12 / Harbor 0.23.0，原 DSH 运行环境为
Python 3.12 / Harbor 0.24.0。

| 机制 | 代码与当前行为 | 接口决策 |
|---|---|---|
| 原生调用 | `RUN_FULL20.ps1` / `RUN_EVOLVE_QUEUE.ps1` → `metacorsi.harbor_agents.dsh_runner` → Harbor 独立任务环境 → `AgentHarness` → DSH 和 shell → 独立 verifier。顺序脚本另行运行 CLS 巩固。 | 保留历史 Target 和配置，用作原生复现。 |
| 查询、分层和 Grounding | `MetaCoRSIBrain.evolved_knowledge_prompt` 调用 `IndexedLTM.ground` 和 `render_grounded_application`。任务、仓库、能力、失败模式、因果位置由原生索引构建。CLS 依据 manifest 中的 `backbone_atom_ids` / `delta_atom_ids` 分层，同题快记忆角色为成功 R/F/M、失败 R/F/S。成功证据与角色配额决定 retention anchors。 | 通过原生代码读取冻结知识并输出结构化来源，复用原始渲染。 |
| 动态重检索 | `AgentHarness.run` 将 `evidence_gated` 限于 `evidence_gated_v20` / `failure_locked_v21`。`modular_reverse_v23` 开始时用仓库证据检索，后续这两个分支不执行。 | V23 只做初始检索。 |
| RCA 与步数 | DSH 依据真实非零返回码构建 `FailureEvidence`，`rca_only` 调用原生 RCA。一次 DSH 步会计划多条 shell 命令，成功的定向测试可解锁 24 步基础预算外最多 6 步。DSH 完成/耗尽预算时有自动提交。 | 复用 RCA，报告 MiniAgent turn 与 DSH step 的差异。统一对照保留任务提交要求。 |
| 蒸馏和晋升 | `knowledge/evolution.py` 定义 VerifiedEpisode、角色蒸馏、图和索引。标准模式可蒸馏失败的负证据，`grounding_feedback` 仅把严格通过任务纳入知识原子，却仍记录 grounding 和隔离候选。候选隔离由 `update_candidate_registry` 管理。 | 保留策略选择，区分任务评分、原子入库和 Snapshot 选择。 |
| 知识来源 | K20 215 原子，K28 251，K29 263。K29 冻结 manifest 的慢主干是 K28 的 251 原子，另有 12 个近失误原子。版本构建含自动蒸馏和历史任务的人工选择，不能当作无历史训练起点。 | K29 用于历史任务复现。独立测试采用 cold-start 或已验证来源隔离的知识。 |
| FlashRSI 调度 | `runner.py` 建立 Agent/Algorithm/Worker/Benchmark。Algorithm 用 `_evaluate` 调用 Target 的 rollout，再由 Benchmark.grade 评分。Snapshot 保存组件文件、父版本和哈希。测试集在选优后运行。 | 装配新 Target，共用原有调度、评分和资源权限。 |
| MiniAgent | `agents/miniagent` 启动各 Snapshot 的原生 ReAct runtime。 `runtime/agent.py` 与 `runtime/workspace.py` 已提供模型调用和工具调用钩子。轨迹记录 messages、model_calls 和 turns。 | 插入原生 Memory Runtime，继续使用共同 MiniAgent 工具循环。 |
| Harbor/DeepSWE | `integrations/deepswe.py` 绑定 Harbor；任务及独立 verifier 默认无网络，Agent phase 只允许模型主机。补丁由任务声明的 collect 从 Git HEAD 获取。容器 setup 默认下载 uv 和 Python 依赖。 | 检查实际执行环境，保持原始任务校验与 verifier。 |
| 历史适配 | `8ce6a02` 把 DSH 作为宿主子进程，以路径和 reward 构造最终事件。DSH 细节存原生文件，FlashRSI Trajectory 没有 DSH 对话/工具或 usage。进化硬编码 `grounding_feedback`，禁重放，只提交当前索引和图。 | 保留旧适配。新路径需要从可信 verifier 事件统一构建 Episode。 |
| 提示词/技能基线 | ACE、GEPA 操作 prompt，SkillOpt 操作 skill。RRSI、ModularRSI 使用 Worker 和各自搜索空间。 | MiniAgent 默认配置不加载 MetaCoRSI，也不改变基线编辑权限。 |

原始方案为工作区的 `适配提示词.docx`。源码迁移属于 adapter 或等价重构，
独立记录运行依赖修复。历史 DeepSWE-20、首次 DSH 单题、合成任务冒烟均保留
原有实验口径。原始评测任务的标准补丁和隐藏答案不参与本次接入。

复核发现 `MetaCogRSI-main` 与 `a42eb3c` 的 `indexed_ltm.py`、
`evolution.py` 和 DSH 主入口哈希不同。前者进一步实现了
`event_driven_v24`，后者没有。MiniAgent 配置指向 Git 仓库中
`a42eb3c` 的原生派生目录。该目录只增加了一个抽取原有进化逻辑的公共入口。
V23 的动态重检索范围结论同时适用于两个副本。
