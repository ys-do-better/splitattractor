# MetaCoRSI V9 抽象与 Grounding 算法设计

日期：2026-11-14
适用工程：当前 MetaCogRSI 仓库根目录
当前脑结构版本：`V9`（hierarchical grounding）
当前知识版本：`K12`（155 个知识原子）

## 1. 命名与目标

本文统一使用两条相互独立的版本线：

- `K11、K12……`：知识状态版本。只有知识原子发生提交时才递增。
- `V7、V8、V9……`：脑结构与算法版本，包括执行控制、检索、Working Memory 组织和 Grounding 策略。

因此，当前系统应写作：

```text
V9 脑结构 + K12 知识
```

本算法要解决的问题不是保存历史补丁，而是把一次具体任务中的经验整理成两类可复用信息：

1. 正向路径：在什么条件下，哪类观察、推理、修复和验证程序已经被证明有效。
2. 反向应用：面对一个新任务，如何把抽象规律重新绑定到当前仓库、当前反例和当前代码位置。

核心原则：

> 抽象是整理知识的方式；Grounding 是从抽象返回具体问题的应用方式。

## 2. 总体数据流

```text
任务描述 + 当前仓库证据
          │
          ▼
Perception：识别 task / repository / capability / causal locus
          │
          ▼
多轴 LTM 检索
retention anchor → task → repository → capability → failure mode → causal locus
          │
          ▼
V9 分层激活
同任务已验证程序 > 同任务失败证据 > 能力族抽象
          │
          ▼
最多选择 2 个 AbstractKnowledge
          │
          ▼
Grounding：绑定当前任务、仓库、能力、因果位置和失败类型
          │
          ▼
Working Memory 胶囊
P/M/R/F/S + 最小动作 + 范围约束 + 验证计划
          │
          ▼
Agent 执行 → 独立 verifier（F2P/P2P）
          │
          ├── 成功：grounding=verified，生成成功原子，建立 retention anchor
          └── 失败：grounding=rejected，生成 RCA 负原子，partial patch 隔离
          │
          ▼
Consolidation：重建索引、抽象、正负 grounding 证据
          │
          ▼
提交新的 K 版本，或在无新原子时生成同 K 的新索引 generation
```

## 3. 知识原子层

### 3.1 P/M/R/F/S 五类原子

每个 episode 经独立 verifier 后被压缩为以下角色：

| 类型 | 含义 | 主要内容 |
|---|---|---|
| P | Perception | 应观察的失败、测试、补丁落盘和集成证据 |
| M | Memory | 已验证 episode、F2P/P2P、补丁摘要和修改文件 |
| R | Reasoning | 根因判断或已验证推理模式 |
| F | Fix | 下一修复程序或已验证修复模式 |
| S | Safety | 晋升条件、回归边界和禁止条件 |

成功 episode 产生 `P/M/R/F/S`；失败 episode 产生 `P/R/F/S`。失败知识可以参与同任务恢复，但不能建立 retention anchor。

### 3.2 原子的五维坐标

每个原子进入 Indexed LTM 后都被映射到：

```text
task
repository
capability
failure_mode
causal_locus
```

当前 capability 由任务、原子文本和标签中的关键词规则识别，包括：

- `parsing-schema`
- `cache-state`
- `async-lifecycle`
- `binding-type`
- `transformation-roundtrip`
- `dependency-policy`
- `security-analysis`
- 默认 `repository-contract`

当前 causal locus 包括：

- `normalization`
- `dispatch-registration`
- `state-propagation`
- `boundary-decision`
- `validation-preservation`
- `materialization`
- 默认 `unspecified`

这些坐标是检索索引，不是对真实代码因果链的完整证明。

## 4. 两层抽象设计

当前系统存在两种相互关联但用途不同的抽象。

### 4.1 K 版本中的抽象原子

`evolve()` 按下面的键聚合同类任务证据：

```text
(capability, root_kind)
```

只要该组至少存在一个成功任务，系统就会建立一个 `R` 类型抽象原子：

```text
abstract.<capability>.<root_kind>.<digest>
```

其内容是一条通用假设：

```text
保留已经通过的行为；
先复现一个具名反例；
定位最近的共享因果决策点；
只有当前仓库证据支持时才执行最小修改。
```

这类抽象原子会进入 K 版本，参与版本哈希和知识原子计数。例如 `K11 → K12` 新增的 6 个原子中，有 1 个是 `parsing-schema × verified_success` 抽象原子，另外 5 个是 Cliffy 的 P/M/R/F/S 成功原子。

### 4.2 V9 运行时 AbstractKnowledge

Indexed LTM 在每次 build/consolidation 时，按照更细的键重新组织全部原子：

```text
(capability, failure_mode, causal_locus)
```

每个 `AbstractKnowledge` 包含：

```text
abstract_id
capability
failure_mode
causal_locus
trigger_tags
invariant
action_schema
safety_boundary
supporting_atom_ids
positive_tasks
negative_tasks
status
```

其中：

- `supporting_atom_ids` 保留抽象的证据来源；
- `positive_tasks` 记录验证成功的应用任务；
- `negative_tasks` 记录尚未被成功覆盖的失败任务；
- `action_schema` 是带 `{task}`、`{repository}`、`{causal_locus}` 等槽位的动作模板，而不是历史补丁。

当前 K12 构建出 62 个运行时 abstraction。它们是从 155 个原子重建的索引结构，不等于又增加了 62 个知识原子。

## 5. 抽象生成算法

### 5.1 分组

对所有带任务坐标的原子执行：

```python
key = (capability, failure_mode, first_causal_locus)
groups[key].append(atom_record)
```

### 5.2 正负证据

```python
positive_tasks = tasks containing pass atoms
negative_tasks = tasks containing fail atoms but no pass atoms
```

如果同一任务后来出现成功证据，它将从该抽象的负集合中移除。

### 5.3 抽象状态

| 条件 | 状态 | 含义 |
|---|---|---|
| 0 个 positive task | `unverified` | 只有失败证据，不能作为已证实技能 |
| 1 个 positive task | `provisional` | 单任务有效，属于待验证假设 |
| ≥2 个 positive tasks | `validated` | 已有跨任务正证据 |
| ≥3 个 positive tasks 且无 negative task | `transferable` | 满足当前可迁移门槛 |

这是证据等级，不代表形式化证明。

### 5.4 不变量和动作模板

当前实现对两类主要失败生成专门模板。

#### patch_materialization_failure

不变量：

```text
在最小、可编译的 vertical slice 被真正落盘并执行前，修复不能被视为已建立能力。
```

动作模板：

```text
1. 将一个失败反例绑定到当前仓库最近的 causal decision。
2. 只落盘最小可编译修改，并立即 build/typecheck。
3. 依次运行 focused counterexample 和 retained tests，再决定是否扩展。
```

#### contract_boundary_gap

不变量：

```text
部分修复已经保持旧行为时，应冻结通过核心，只修改剩余反例共享的最近决策边界。
```

动作模板：

```text
1. 将一个剩余反例绑定到最近边界决策。
2. 不重写已通过核心，只做一个最小边界修改。
3. 按 build/typecheck → focused test → retained tests 的顺序验证。
```

#### 其他 failure mode

采用通用程序：复现一个反例、绑定最近因果位置、做一个最小修改、执行 focused 与 retained verification。

### 5.5 安全边界

所有抽象共享三个基本边界：

1. 不复制来源任务的源码或补丁，只应用抽象不变量。
2. 编译或 focused counterexample 仍为红色时，不扩大实现范围。
3. 只有独立 F2P 与 P2P verifier 才能决定晋升。

## 6. V9 多轴检索

### 6.1 检索顺序和配额

```text
retention_anchor：最多 5
task：最多 5
repository：最多 2
capability：最多 3
failure_mode：最多 2
causal_locus：最多 2
总量默认最多 14
```

`retention_anchor` 只由 pass 原子建立，每个任务按 P/R/F/S/M 各选一个最高置信原子。

### 6.2 V8 已修复的两个检索问题

V9 继承了 V8 的修复：

1. Working Memory 使用 `retention_anchor ∪ task`，避免原子被 anchor bucket 去重后从任务胶囊消失。
2. Perception 即使把 capability 判断错误，精确同任务的 positive/negative abstraction 仍保持可达。

### 6.3 Working Memory 角色平衡

同任务原子不会按文本排名简单截断，而是按角色组织：

```text
M → F → R → S → P
```

每类最多选择一个代表原子，优先使用 pass、较高置信度和信息较完整的记录。这样可以在有限上下文中保留一个完整 episode 程序，而不是被大量同类失败描述淹没。

## 7. V9 分层 Grounding

### 7.1 Grounding 的定义

Grounding 不是检索历史答案，也不是补丁重放。它是把一个抽象模板实例化为当前任务的应用收据：

```text
抽象不变量
  + 当前 task
  + 当前 repository
  + 当前 capability
  + 当前 causal locus
  + 当前 failure mode
  + 当前仓库证据
  = GroundingRecord
```

### 7.2 激活层级

V9 的优先关系是：

```text
第一层：精确同任务、已验证的成功程序
第二层：精确同任务、尚未解决的失败/边界证据
第三层：经过验证的能力族迁移抽象
第四层：当前仓库和当前反例证据（最终裁决）
```

关键门控：

- `exact_positive`：抽象的 positive_tasks 中包含当前任务；优先级最高。
- `exact_negative`：negative_tasks 中包含当前任务；用于恢复，但不能冒充成功程序。
- `transferable`：状态为 validated/transferable，且 trigger tag 或 causal locus 与当前任务匹配。
- 若当前任务已经有 pass 原子，则仅含 stale negative、没有 exact positive 的抽象被抑制。
- V9 一次最多激活 2 个 abstraction，防止 Working Memory 过载。

这不是完全隔离：同任务成功程序占主导，但允许少量能力族抽象补充缺失步骤。

### 7.3 GroundingRecord

每条记录包含：

```text
grounding_id
abstract_id
task / repository / capability
context_digest
activation_evidence
bindings
instantiated_action
scope_guard
verification_plan
materialization_protocol
status
reward
outcome_evidence
```

动作模板实例化示意：

```text
模板：Bind one failing counterexample to the nearest {causal_locus} decision in {repository}.

绑定：
task = actionlint-action-pinning-lint
repository = actionlint
causal_locus = dispatch-registration

实例：
将一个失败反例绑定到 actionlint 中最近的 dispatch-registration 决策点。
```

### 7.4 Grounding 输出给 Agent 的内容

每条应用只输出紧凑胶囊：

- 最多 3 个 activation trigger；
- 最多 3 个 minimal actions；
- 最多 2 个 scope guards；
- 固定验证顺序：build/typecheck、focused test、独立 F2P/P2P；
- 可执行 materialization protocol；
- V9 整体上下文上限约 6800 字符。

## 8. Verifier 回写与 Consolidation

### 8.1 Grounding 状态转移

运行时新生成的 grounding 初始状态为：

```text
pending
```

独立 verifier 完成后：

```text
reward == 1 → verified
reward != 1 → rejected
```

同时记录 F2P、P2P 和最多三个失败反例。

### 8.2 Consolidation

系统重新构建全部 abstraction，然后把 grounding 历史合并进去：

```python
if grounding.status == "verified":
    positive_tasks.add(task)
    negative_tasks.discard(task)

elif grounding.status == "rejected" and task not in positive_tasks:
    negative_tasks.add(task)
```

之后重新计算 `unverified / provisional / validated / transferable`。

### 8.3 成功与失败的不同写回

成功：

- 生成 P/M/R/F/S；
- 建立 retention anchor；
- grounding 记为 verified；
- 清理该任务的隔离 candidate；
- 若产生新原子，则提交下一个 K 版本。

失败：

- 生成 P/R/F/S 负原子；
- grounding 记为 rejected；
- 有有效 partial patch 时只进入同任务 candidate registry；
- candidate 不进入跨任务抽象，也不建立 retention anchor。

## 9. K11 → K12 实例

```text
起点：K11
149 atoms
9 个 retention tasks
```

K11 上先验证 Cattrs：

```text
F2P 69/69，P2P 7/7
```

随后执行 Cliffy：

```text
F2P 37/37，P2P 451/451
```

成功 episode 经 distillation 与 consolidation 后新增：

- Cliffy P/M/R/F/S：5 个原子；
- `parsing-schema × verified_success`：1 个抽象原子。

最终得到：

```text
K12
155 atoms
10 个 retention tasks
62 个运行时 abstractions
2 条 Cliffy verified grounding records
```

冻结后 Cattrs 与 Cliffy 均通过，证明这一次进化完成了“新增 Cliffy，同时保持 Cattrs”的最小闭环。

## 10. 当前算法已经实现与尚未实现的边界

### 已实现

- 版本化不可变知识原子；
- task/repository/capability/failure/locus 多轴索引；
- 成功任务 retention anchor；
- P/M/R/F/S 角色平衡 Working Memory；
- 正负任务证据驱动的 abstraction；
- abstraction 到当前任务的反向 Grounding；
- verifier 驱动的 verified/rejected 回写；
- 同任务 partial candidate 隔离；
- 禁止历史补丁直接作为跨任务技能重放。

### 尚未实现或仍较弱

1. capability 和 causal locus 主要依赖关键词规则，可能出现 Perception 误路由。
2. 当前 abstraction 的 invariant/action schema 主要是规则模板，不是从多条因果链自动归纳出的新程序。
3. 已接入根因构造层归因，但尚未完成代码级因果链搜索和系统化反事实干预验证。
4. `transferable` 仅由任务计数和负证据门槛决定，尚未做严格 held-out transfer 统计。
5. 当前成功晋升以 reward=1 为硬边界；部分改进只进入 candidate/负知识，不会成为成功技能。
6. V9 + K12 的六题 4/6 是冻结重放结果，不代表全部 20 题的即时能力。

## 11. 关键不变量

后续修改算法时应保持以下约束：

1. K 版本与 V 版本分开编号。
2. 原子是证据，不是答案；当前仓库证据优先于 LTM。
3. 抽象必须保留 supporting atom ids、positive tasks 和 negative tasks。
4. Grounding 必须有 scope guard 和 verification plan。
5. 成功任务的 verified program 不得被旧失败分支覆盖。
6. 允许软迁移，不做完全记忆隔离；但每轮最多激活少量抽象。
7. 只有独立 verifier 可以晋升知识。
8. partial patch 只能同任务续跑，不能成为跨任务技能。
9. 禁止直接读取或存储标准答案；允许保存由自身执行和 verifier 反馈形成的做题经验。

## 12. 代码对应关系

| 功能 | 文件 |
|---|---|
| 原子蒸馏、verifier 回写、K 版本提交 | `metacorsi/knowledge/evolution.py` |
| 能力识别、sidecar graph、基础 grounded retrieval | `metacorsi/knowledge/grounding.py` |
| 多轴索引、运行时 abstraction、V9 retrieval/grounding | `metacorsi/knowledge/indexed_ltm.py` |
| DSH 执行与 Working Memory 注入 | `metacorsi_dsh_agent/__init__.py` |
| abstraction/grounding 单元测试 | `tests/test_abstraction_grounding.py` |
| Indexed LTM 与策略测试 | `tests/test_indexed_ltm.py` |
| V9 实验配置 | `experiments/attempt09/spec.json` |
| V9 六题证据 | `evidence/indexed_attempt09_best/` |

## 13. 一句话总结

MetaCoRSI V9 不把一次成功保存成可直接重放的答案，而是把 episode 压缩为带任务边界和 verifier 证据的 P/M/R/F/S 原子；consolidation 将这些原子整理成能力—失败类型—因果位置三维抽象，grounding 再把最多两个抽象绑定到当前仓库的具体反例和验证程序，从而尝试在持续进化与旧能力保持之间取得平衡。
