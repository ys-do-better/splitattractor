# MetaCoRSI

这是当前精简后的 MetaCoRSI 主工程。目录只保留 K20/K28/K29 知识版本、RSI 核心算法与依赖、必要启动脚本和说明。

## 当前状态

| 知识版本 | 原子数 | 作用 |
|---|---:|---|
| K20 | 215 | 稳定 slow-store 起点 |
| K28 | 251 | K20 + 36 个 fast episodic atoms |
| K29 | 263 | K28 + 三个 near-miss episode 的 12 个 P/R/F/S atoms |

当前冻结包：`knowledge/releases/k29_cls_nearmiss3`。

K29 在 DeepSWE-20 上的严格 **Pass@5 为 18/20（90%）**。这不是 one-shot 18/20，也不是边运行边学习的累计成绩；每题最多五次独立轨迹，K29 全程只读，通过即停止。

两个 Fail@5：

- `arktype-json-schema-refs-dependencies`：递归 `$ref` 缺少稳定节点 identity、占位与 SCC 闭合；
- `bandit-structured-nosec-directives`：suppression 仍按物理行传播，没有 grounding 到完整 AST statement。

逐题结论见 `knowledge/releases/k29_cls_nearmiss3/PASS_AT_5_RESULTS.md`。

| DeepSWE task | Pass@5 |
|---|---:|
| ABS module cache | PASS |
| ABS stepped slices | PASS |
| Actionlint pinning | PASS |
| Adaptix aliases | PASS |
| aiomonitor snapshots | PASS |
| Anko default arguments | PASS |
| Anko typed variables | PASS |
| Arcane drift baselines | PASS |
| Arktype JSON Schema refs | **FAIL** |
| Awilix async container | PASS |
| Bandit incremental cache | PASS |
| Bandit interprocedural taint | PASS |
| Bandit structured nosec | **FAIL** |
| Boa cancellation | PASS |
| Cattrs partial structuring | PASS |
| Clack async autocomplete | PASS |
| Claude Code delegation | PASS |
| Cliffy config parsing | PASS |
| CSS Tree shorthand | PASS |
| Dasel HTML | PASS |

## 目录

| 路径 | 内容 |
|---|---|
| `knowledge_sources/k20_215/` | K20 不可变来源 |
| `knowledge_sources/k28_251/` | K28 不可变来源 |
| `knowledge_sources/k29_nearmiss3/` | K29 新增 episode 来源 |
| `knowledge/active/` | K28 active snapshot |
| `knowledge/releases/k29_cls_nearmiss3/` | K29 冻结包、索引与最终结果 |
| `metacorsi/` | RSI、LTM、CLS、abstraction/grounding、RCA、runner |
| `metacorsi_dsh_agent/` | DSH Agent harness |
| `vendor/rca-repair/` | RCA repair 与 five-layer walker |
| `configs/` | DSH/模型执行配置 |
| `tests/` | 算法与集成测试 |
| `scripts/` | K20→K28→K29 构建、审计和整合脚本 |

## 外部依赖

- Windows Python / 本地 DSH；
- Docker Desktop Linux 容器；
- DeepSWE verifier；
- 模型配置默认使用 `D:\.ds`，或通过 `METACORSI_DOTENV` 指定。

严禁读取 gold/reference patch 或隐藏答案。允许公开题目、仓库代码、公开测试、自身执行日志与 verifier 的通过/失败反馈。

## 常用入口

```powershell
# 环境
.\SETUP_LOCAL.ps1
.\CHECK_ENV.ps1
.\VALIDATE_PACKAGE.ps1

# 构建知识
.\BUILD_K28_CLS.ps1
.\BUILD_K29.ps1

# 冻结全量重放；显式指定 K29
.\RUN_FULL20.ps1 -Knowledge .\knowledge\releases\k29_cls_nearmiss3

# task-wise Pass@5
.\RUN_K29_PASSAT5.ps1 -TasksFile .\data\deepswe20_ids.txt -MaxAttempts 5

# 顺序进化
.\RUN_EVOLVE_QUEUE.ps1

# 验证后显式晋升
.\PROMOTE_KNOWLEDGE.ps1 -Knowledge <working-brain-path>
```

## K29 完整性

- version：29
- atoms：263
- current SHA-256：`EF5FE5DED43F12B4A59CBE0FC65778D52D7C1133EE002C708DA20088396C5B67`
- manifest SHA-256：`CF4058BD3DDDB6E68BA9348BDC35C40B6DCED736940ED7931F8EE6AE68CA56B3`
- retrieval audit：20 tasks，0 fast-delta leakage violations
- answer/reference patch access：false

算法说明见 [ALGORITHM.md](ALGORITHM.md)、[INDEXED_LTM_DESIGN.md](INDEXED_LTM_DESIGN.md) 和 [抽象与Grounding算法设计_V9.md](抽象与Grounding算法设计_V9.md)。
