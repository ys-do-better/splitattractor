# MetaCoRSI 架构综述与引入“分裂吸引子（Split Attractor）”记忆机制技术报告

**文档用途**：架构评审与多智能体（Multi-Agent）方案研讨材料  
**核心议题**：基于 `MetaCoRSI` 主体自进化架构，引入 *Nature* (2026)《*A split attractor design for rapidly writing a navigational goal*》的分裂吸引子与去抑制门控记忆机制，优化工作记忆（Working Memory）的动态写入、稳态维持与快速擦除能力。

---

## 目录

1. **执行摘要（Executive Summary）**
2. **第一部分：MetaCoRSI 方法主体架构与现有记忆机制深度解析**
   - 1.1 系统定位与双版本线（V-Line / K-Line）设计原则
   - 1.2 自进化主闭环（Self-Evolution & Execution Pipeline）
   - 1.3 知识原子层：P/M/R/F/S 五元角色与五维检索坐标
   - 1.4 现有记忆组织机制：CLS 双存储、三维抽象与逆向 Grounding
   - 1.5 已实现的 10 种认知记忆策略与运行时门控演进（V9 → V23）
   - 1.6 当前基准表现（DeepSWE-20）与现有记忆机制的四大瓶颈
3. **第二部分：《A split attractor design for rapidly writing a navigational goal》记忆机制深度解析**
   - 2.1 核心科学悖论：工作记忆的“稳定性 vs. 灵活性”
   - 2.2 环路拓扑创新：PFG 与 hΔK 的“分裂吸引子（Split Attractor）”结构
   - 2.3 三大核心动力学机制（慢兴奋/快抑制、去抑制瞬间写入、重抑制快速擦除）
   - 2.4 分裂吸引子与传统单一吸引子的计算特性对比
4. **第三部分：融合方案设计——基于 Split Attractor 重构 MetaCoRSI 记忆机制**
   - 3.1 神经计算机制与 MetaCoRSI 组件的精确映射（Isomorphism Mapping）
   - 3.2 分裂吸引子工作记忆（SA-WM）的四层运作机制设计
   - 3.3 核心代码模块改造落点规划
   - 3.4 预期收益分析（以 Clack 负迁移与 Arktype/Bandit 难题为例）
5. **第四部分：面向其他 Agent / 评审专家的开放研讨问题清单**

---

## 1. 执行摘要（Executive Summary）

`MetaCoRSI`（Meta-Cognitive Recursive Self-Improvement）是一个面向复杂软件工程任务（如 DeepSWE）的受控自进化智能体框架。与传统的“提示词优化”或“历史补丁检索重放（Patch Replay）”不同，`MetaCoRSI` 严格禁止直接保存或复制历史标准答案，而是通过独立验证器（Independent Verifier: F2P + P2P）与五层深度根因分析（Deep RCA），将每一次做题经验蒸馏为结构化的 **`P/M/R/F/S` 知识原子**，再经由互补学习系统（CLS）整合为跨任务可迁移的**三维机制抽象（AbstractKnowledge）**，并在运行时通过**逆向绑定（Reverse Grounding）**注入智能体的工作记忆（Working Memory）。

目前，`MetaCoRSI` 的 `K29` 冻结知识包在 DeepSWE-20 基准上已达到 **Pass@5 = 18/20 (90%)**（单轮进化 K20→K28-CLS 从 5/20 提升至 11/20）。然而，在跨任务迁移与多步执行过程中，现有的工作记忆与 Grounding 机制仍面临一个根本性矛盾：
- **若在 Step 0 过早锁定跨任务 Schema（如 V22/V23）**：在尚未观察到真实代码反例前，仅凭题面词汇激活抽象先验，容易对无精确历史记忆的任务（如 `Clack`、`Claude delegation`、`Adaptix`）产生**跨任务负迁移（Premature Lock / Negative Transfer）**；
- **若在每步随终端输出动态刷新 Schema（如 V20）**：由于缺乏吸引子锁存机制，工作记忆中的指导假设会随单步噪声发生**闪烁与漂移（Bump Drift）**；
- **若在首次测试失败后单向死锁 Schema（如 V21）**：一旦首次锁定的因果假设不完整或走入死胡同（如 `Arktype` 递归 `$ref` SCC 闭合、`Bandit` 物理行与 AST 语句绑定问题），系统缺乏**快速擦除旧目标并重锁新假设**的动态门控能力。

2026 年发表于 *Nature* 的论文《*A split attractor design for rapidly writing a navigational goal*》（Lanz et al.）揭示了果蝇导航中枢如何通过**“分裂吸引子（Split Attractor：将实时感知追踪层 PFG 与循环记忆维持层 hΔK 在结构上解耦）+ 去抑制门控（Disinhibitory Gating）+ 慢速循环兴奋与快速侧向抑制平衡”**完美解决上述矛盾。本报告系统梳理了 `MetaCoRSI` 的现有架构与该论文的神经记忆机制，并提出了一套完整的融合替换方案。

---

## 2. 第一部分：MetaCoRSI 方法主体架构与现有记忆机制深度解析

### 1.1 系统定位与双版本线（V-Line / K-Line）设计原则

`MetaCoRSI` 严格区分两条相互正交的演进轴线：
1. **知识状态版本线（`K` 版本，如 `K7 → K12 → K20 → K28 → K29`）**：指不可变原子库（Immutable Atom Vault）与多轴索引图的快照版本，仅在经独立 Verifier 验证产生新原子并提交事务时递增。
2. **脑结构与认知控制版本线（`V` 版本，如 `V9 → V15 → V20 → V21 → V23`）**：指感知编译（Perception）、检索路由、工作记忆胶囊组装、逆向 Grounding 策略以及运行时阶段控制的算法版本。

**核心安全与实验完整性边界**：
- **禁止答案/补丁泄漏**：严禁读取 gold/reference patch、隐藏测试或跨容器搜索；原子仅保存机制不变量与语义级推理轨迹（`_execution_trace` / `_patch_blueprint`），绝不保存可执行源码 diff。
- **独立验证器裁决**：只有独立容器中的 `F2P`（Fail-to-Pass 新能力验证）与 `P2P`（Pass-to-Pass 旧行为保持）全部通过（`reward == 1`），经验才允许晋升为正面知识锚点（Retention Anchor）。

### 1.2 自进化主闭环（Self-Evolution & Execution Pipeline）

`MetaCoRSI` 的完整运行与进化数据流如下：

```text
Task 描述 + 目标代码仓库 (/app)
  │
  ├─► 1. Perception (感知编译)
  │     提取 task / repository / capability (多标签亲和度) / causal_locus 查询向量
  │
  ├─► 2. Indexed LTM 多轴检索 & CLS 分层胶囊构建 (_layered_capsule)
  │     Slow Store (5 个稳定骨干原子 M/F/R/S/P) + Fast Store (最多 3 个同任务增量原子)
  │
  ├─► 3. Runtime Abstraction & Reverse Grounding (逆向绑定)
  │     匹配 Top-1 三维抽象 (capability × failure_mode × causal_locus)
  │     → 逆向检索 Top-1 目标不相交源任务 (target-disjoint source episode) 的 R/F/S 角色
  │     → 组装 GroundingRecord (含触发证据、绑定槽位、最小动作、范围守卫、验证计划)
  │
  ├─► 4. DSH Agent 四阶段受控执行 (24 基础步 + 最多 6 步有条件扩展)
  │     ├─ Step 1~4   [DIAGNOSE]    定位残余反例与最近共享因果决策点
  │     ├─ Step 5~12  [MATERIALIZE] 落盘最小可编译垂直切片并打通导出/分发路径
  │     ├─ Step 13~18 [VERIFY]      运行端到端聚焦探针，冻结已通过核心，闭合边界缺口
  │     └─ Step 19~24 [CLOSE]       停止广泛探索，完成最小决定性测试并持久化 Git 检查点
  │
  ├─► 5. 在线 Deep RCA 与回归门控 (Online RCA & Focused Regression Gate)
  │     每当命令/测试返回非零状态，启动五层因果归因 (Logic/Boundary/Schema/Domain/FDE)
  │     执行 walk 与 walk-back 校验；Agent 声明 done=true 时必须通过历史失败探针回归门控
  │
  ├─► 6. 独立 Verifier 裁决 (Independent F2P + P2P Verification)
  │
  └─► 7. 进化写回与 CLS 整合 (Evolution Write-back & Consolidation)
        ├─ Pass (reward=1): 蒸馏 P/M/R/F/S 正原子，建立 Retention Anchor，Grounding 记为 verified
        └─ Fail (reward<1): 蒸馏 P/R/F/S 负原子，Grounding 记为 rejected；
                            若存在有效局部修复 (f2p_passed > 0)，将其隔离至同任务 candidates.json (Pareto 前沿管理)
```

### 1.3 知识原子层：P/M/R/F/S 五元角色与五维检索坐标

每次 Verifier Episode 结束后，`distill_episode()` 将执行轨迹与验证反馈压缩为五类互补的角色原子：

| 原子类型 | 角色名称 | 核心承载内容 | 生成条件与作用域 |
| :--- | :--- | :--- | :--- |
| **P** | **Perception** (感知观察) | 应当观察的失败表现、补丁落盘状态、F2P/P2P 计数、反例摘要与反射不变量 | Pass / Fail 均生成；指导优先观察点 |
| **M** | **Memory** (验证记忆) | 已验证通过的 Episode 摘要、补丁哈希（无源码）、修改文件集合（situation signature） | **仅 Pass 生成**；作为同任务能力保持锚点 |
| **R** | **Reasoning** (因果推理) | Pass 时为结构化实现蓝图与验证轨迹；Fail 时为 RCA 根因类别、阶段与反例不变量 | Pass / Fail 均生成；提供因果假设模型 |
| **F** | **Fix** (修复程序) | Pass 时为修改站点与语义级执行配方；Fail 时为 RCA 恢复策略与备选假设列表 | Pass / Fail 均生成；指导最小修改动作 |
| **S** | **Safety** (安全守卫) | 晋升所需的 F2P/P2P 回归边界、禁止越界扩展的转换守卫（Transition Guard） | Pass / Fail 均生成；约束修改范围 |

每个原子进入 `IndexedLTM`（`ltm/versions/v*/`）时，会自动建立 **5 维坐标索引**：
1. **`task`**：规范化任务标识（如 `actionlint-action-pinning-lint`）。
2. **`repository`**：仓库族前缀（如 `actionlint`、`bandit`、`arktype`）。
3. **`capability`**（7 类能力族 + 默认项）：`parsing-schema`、`cache-state`、`async-lifecycle`、`binding-type`、`transformation-roundtrip`、`dependency-policy`、`security-analysis`（默认 `repository-contract`）。
4. **`failure_mode`**（7 类失败模式）：`patch_materialization_failure`、`execution_protocol_failure`、`preservation_regression`、`contract_boundary_gap`、`mixed_contract_and_preservation_failure`、`contract_integration_failure`、`verified_success`。
5. **`causal_locus`**（6 类因果位置 + 默认项）：`normalization`、`dispatch-registration`、`state-propagation`、`boundary-decision`、`validation-preservation`、`materialization`（默认 `unspecified`）。

### 1.4 现有记忆组织机制：CLS 双存储、三维抽象与逆向 Grounding

为了解决“持续学习新任务时不遗忘旧任务，且不让未验证的局部经验污染全局检索”，`MetaCoRSI`（特别是在 K28/K29 与 V15~V23 中）构建了三层记忆架构：

#### （1）互补学习系统（CLS Dual-Store）与 `_layered_capsule`
- **Slow Store（慢速稳定骨干，如 K20 的 215 个原子 / K28 的 251 个原子）**：代表已经固化稳定的核心能力。在构建同任务工作记忆胶囊时，系统为 Slow Store **硬性预留 5 个槽位（按 `M → F → R → S → P` 各选最优代表）**，任何新增分支都无法将旧骨干挤出工作记忆。
- **Fast Store（快速情景增量，如 K28 新增的 36 个原子、K29 新增的 12 个 near-miss 原子）**：保存在 `memory_layers.delta_atom_ids` 中。Fast 原子**严禁直接跨任务检索**（跨任务泄漏率为 0），仅能作为同任务（exact-task）的适应分支，最多占据 **3 个增量槽位**（若为已验证成功增量则取 `R/F/M`，若为 near-miss 失败增量则取 `R/F/S`）。

#### （2）三维运行时抽象（`AbstractKnowledge`）
在每次 Consolidation 时，`IndexedLTM._build_abstractions()` 将所有带任务坐标的原子按三维键 `(capability, failure_mode, first_causal_locus)` 聚类，并融合历史 `GroundingRecord` 的验证反馈（`verified` vs. `rejected`），构建抽象节点：
- **证据晋升阶梯**：
  - `0` 个独立成功任务 → `unverified`
  - `1` 个独立成功任务 → `provisional`
  - `≥2` 个独立成功任务 → `validated`
  - `≥3` 个独立成功任务且无未解决负证据 → `transferable`
- **结构化模板**：每个 `AbstractKnowledge` 包含跨任务共现的 `trigger_tags`、机制不变量 `invariant`、带 `{task}`/`{repository}`/`{causal_locus}` 槽位的 `action_schema` 以及 3 条 `safety_boundary`。

#### （3）模块化逆向绑定（Modular Reverse Grounding, `modular_reverse_v23`）
当面对一个没有同任务成功记忆的新任务（或需要跨任务机制补充）时，`ground()` 执行**“自顶向下抽象匹配 → 自底向上源任务逆向投射”**：
1. 计算当前任务的多标签能力亲和度分布 `_situation_capability_affinities(task_id, task_text)` 与因果位置匹配度；
2. 选出 Top-1 的 `AbstractKnowledge` 模块；
3. 在该抽象节点的 `supporting_atom_ids` 中，检索出一个**与当前目标任务完全不相交（target-disjoint）且同义重叠度最高的历史成功源任务（`source_episode`）**；
4. 提取该源任务的 `R/F/S` 原子，通过 `_semantic_episode_projection()` 剥离所有具体文件名与路径（替换为 `<source-file>`），生成语义级动作角色序列；
5. 按照混合权重（有同任务记忆时 `local_episode_weight=0.75~0.94`，无同任务记忆时 `abstract_residual_weight=0.16~0.65`）渲染进 `GROUNDED ABSTRACT APPLICATIONS` 提示块。

### 1.5 已实现的 10 种认知记忆策略与运行时门控演进（V9 → V23）

在 `metacorsi/knowledge/indexed_ltm.py` 中，系统已经实现了一个支持切换 10 种认知/神经启发式检索策略的对比框架（通过 `METACORSI_MEMORY_STRATEGY` 控制）：
- `v23_modular`（默认多因子模块化加权）
- `cls_dual_store`（互补学习系统慢速 Schema 残差）
- `anthropic_search_notes`（高触发词权重的笔记检索）
- `anthropic_dreaming`（高源任务多样性与反馈权重的梦境整合）
- `memtree_collapsed` / `memtree_traversal`（层次记忆树坍缩与根叶遍历）
- `schema_assimilation`（Schema 同化乘性一致性评分）
- `art_resonance`（自适应共振理论匹配与重置指导）
- `hippocampal_index`（海马体稀疏线索补全）
- `event_segmentation`（事件边界预测误差切分，仅提取 `R/F`）

同时，在运行时门控（`METACORSI_LTM_POLICY`）上经历了三次关键迭代：
- **`evidence_gated_v20`**：Step 0 不加载跨任务抽象，Step > 1 时每步根据累积的 32KB `repository_evidence_window` 重新调用 `ground()`，仅当代码证据同时支持 `trigger_tags` 和 `causal_locus` 时才注入诊断假设。
- **`failure_locked_v21`**：只有在容器内真正执行了测试命令且返回非零失败（`_confirmed_focused_failure`）后，才解锁跨任务记忆；且通过 `_lock_grounding_receipt()` **一旦锁定第一个 `abstract_id`，在剩余所有步数中永久锁定该抽象，抑制后续关系切换**。
- **`soft_grounding_v22` / `modular_reverse_v23`**：取消硬性激活门控，在 Step 0 基于连续 Situation-Graph 评分选出 Top-K（V22 为 3 个，V23 为 1 个）抽象并逆向绑定源任务 Episode。

### 1.6 当前基准表现（DeepSWE-20）与现有记忆机制的四大瓶颈

- **当前实验表现**：
  - `K20`（215 原子基线）：单轮严格通过 5/20；
  - `K28-CLS`（215 慢速骨干 + 36 快速情景原子，启用 `_layered_capsule`）：单轮严格通过提升至 **11/20**（完整保留 K20 的 5 题，并新增 6 题通过）；
  - `K29-CLS-NearMiss3`（263 原子，引入 3 个 near-miss 任务的 12 个 P/R/F/S 原子）：在只读冻结评估下，**DeepSWE-20 严格 Pass@5 达到 18/20 (90%)**。
  - **2 个未通过任务（Fail@5）**：
    1. `arktype-json-schema-refs-dependencies`：递归 `$ref` 缺少稳定节点 identity、预先占位（placeholder）与强连通分量（SCC）闭合；
    2. `bandit-structured-nosec-directives`：安全抑制指令（`# nosec`）仍按物理行传播，未能在 AST 层面 grounding 到完整的多行 Statement 节点。

- **现有记忆机制的四大核心瓶颈（亟待突破）**：
  1. **瓶颈一：Step 0 过早绑定引发跨任务负迁移（Premature Lock & Negative Transfer）**
     在 `V22/V23` 中，Step 0 尚未运行任何容器内探针，仅靠题目描述和 `_situation_capability_affinities`（其中 `task_id` 静态权重占 70%）就将跨任务 Schema 和源任务轨迹注入工作记忆。当题面词汇与真实代码因果结构存在偏差时（如 `Clack`、`Claude delegation`、`Adaptix`），Agent 会在前几步被错误的跨任务先验带偏。
  2. **瓶颈二：实时状态追踪与记忆锁存混杂（Coupled Tracking & Maintenance）**
     系统没有将“实时追踪当前代码探索状态的指针”与“维持长期修复目标的吸引子”分离开。在 `V20` 中，每步用 32KB 原始终端输出重写记忆，导致工作记忆随局部命令输出剧烈抖动（Bump Drift）。
  3. **瓶颈三：单向死锁缺乏快速擦除与逃逸机制（Irreversible Lock in V21）**
     `V21` 的 `_lock_grounding_receipt()` 虽然防止了记忆抖动，但采用了“一锁定终身”的硬编码策略。如果第一次测试报错触发锁定的 `AbstractKnowledge` 只是表象错误而非深层根因（例如 Bandit 任务中先修物理行解析，后发现需要改 AST 访问器），或者在 `VERIFY/CLOSE` 阶段被 `RCA FOCUSED GATE` 拦截，系统**无法擦除已锁定的错误吸引子并切换到备选假设**。
  4. **瓶颈四：抽象槽位的静态粗粒度绑定（Coarse Slot Binding）**
     目前 `GroundingRecord.bindings` 在实例化 `action_schema` 时，填入的仍是 `{causal_locus} = "state-propagation"` 这类静态宏标签，而未能将在 `DIAGNOSE` 阶段实时探测到的具体函数符号、AST 节点或边界变量原地锁存进工作记忆指令中。

---

## 3. 第二部分：《A split attractor design for rapidly writing a navigational goal》记忆机制深度解析

### 2.1 核心科学悖论：工作记忆的“稳定性 vs. 灵活性”

2026 年发表于 *Nature* 的论文《*A split attractor design for rapidly writing a navigational goal*》（作者：Aaron J. Lanz, Nicholas D. Kathman, Emily Hao, Bard Ermentrout & Katherine I. Nagel）研究了神经科学中关于**工作记忆（Working Memory）**的一个经典计算难题：
- **稳定性要求（Stability）**：工作记忆通常由**循环吸引子网络（Recurrent Attractor Network，如环形吸引子 Ring Attractor）**实现。为了在外部刺激消失后仍能长时间维持稳定的神经活动峰（Persistent "Bump" Activity）且抵抗噪声漂移，网络内部必须具备很强的循环兴奋连接。
- **灵活性要求（Flexibility / Rapid Writing & Erasing）**：强循环连接会产生巨大的“动力学惯性”。当动物在环境中高速移动并突然决定将某个瞬时朝向锁定为新的导航目标（Goal）时，传统单一吸引子网络需要用极强的外部输入把活动峰从旧位置“硬推”到新位置，不仅写入延迟大，而且极易在切换过程中失控或卡死。

**核心问题**：*神经环路如何在保持吸引子长期稳定性的同时，实现毫秒级的“零延迟目标写入”与“瞬间记忆擦除”？*

### 2.2 环路拓扑创新：PFG 与 hΔK 的“分裂吸引子（Split Attractor）”结构

研究者结合果蝇（*Drosophila*）全细胞电生理记录、连接组学与计算建模，发现在果蝇导航中枢——中央复合体的扇形体（Fan-shaped Body, FB）中，负责维持导航目标工作记忆的环形吸引子**并不是一个单一均质的神经元群体，而是被“分裂（Split）”为两个功能解耦但拓扑互连的环形群体**：

1. **PFG 神经元群体（实时输入追踪层 / Sensory Compass Pointer）**：
   - 直接接收来自大脑内部指南针（E-PG 神经元）的实时朝向输入。
   - **自身不具备维持持续性活动的强自循环惯性**：当没有来自 hΔK 的反馈锁存时，PFG 就像一个零延迟的指针，实时、敏捷地跟随当前感知状态旋转。
2. **hΔK 神经元群体（循环记忆维持层 / Persistent Memory Latch）**：
   - 与 PFG 神经元形成按拓扑位置对齐的双向循环连接（Recurrent Loop），并且 hΔK 内部具有循环兴奋通路。
   - **专门负责提供“记忆锁存”所需的持续性正反馈动力**，并接收外部抑制性门控神经元的靶向调控。

### 2.3 三大核心动力学机制

#### 机制一：慢速循环兴奋 + 快速突触抑制（Slow Recurrent Excitation + Fast Inhibition）
传统的连续吸引子模型对突触权重的不均匀性极其敏感（微小噪声就会导致活动峰随机漂移或全网爆发）。论文发现 hΔK 群体通过**时间尺度分离（Timescale Separation）**解决了这一问题：
- **慢速循环兴奋（Slow Recurrent Excitation）**：hΔK 的循环自激并不依赖毫秒级的快速神经递质，而是由**慢速神经调质与神经肽（如酪胺 Tyramine、MIP、Dh31）**介导。这种慢动力学相当于在反馈环路中内置了一个**时间积分低通滤波器**，使活动峰天然抵抗高频感知噪声，不会因短暂干扰而发生位置漂移。
- **快速突触抑制（Fast Inhibition）**：hΔK 同时接收来自突触伙伴的快速抑制输入。快速抑制负责实时约束活动峰的宽度与幅值（实现侧向竞争 Winner-Take-All），防止慢速兴奋累积导致全网络失控激活。

#### 机制二：去抑制门控实现零延迟写入（Disinhibitory Gating for Rapid Writing）
这是“分裂吸引子”最核心的创新点——**利用靶向 hΔK 的抑制信号作为控制两层网络耦合与解耦的高速逻辑门（Gate）**：
- **当果蝇处于转弯或探索期（Tracking Mode）**：外部抑制信号强烈抑制 hΔK 神经元 → **hΔK 与 PFG 之间的循环正反馈回路被切断（解耦）** → PFG 神经元完全不受旧记忆吸引子的拉扯，自由、实时地跟随指南针输入反映当前瞬时朝向。
- **当果蝇闻到气味、决定锁定当前方向进行目标导向直行时（Writing / Latching Mode）**：针对 hΔK 的外部抑制信号突然撤除（**去抑制 Disinhibition**） → **hΔK 与 PFG 瞬间闭合形成完整的循环吸引子** → 由于 PFG 此时此刻正停留在当前瞬时朝向的位置上，闭环产生的慢速循环兴奋**直接将 PFG 上的瞬时活动峰“原地冻结（Latch）”进 PFG–hΔK 回路中**，瞬间成为稳定的导航目标工作记忆！

#### 机制三：重抑制实现快速擦除与重置（Re-inhibition for Rapid Erasing）
- 当一段目标导向航程结束、或者果蝇需要重新转弯探索新方向时，针对 hΔK 的抑制信号再次升高（**重抑制 Re-inhibition**）。
- 这相当于瞬间切断了吸引子的维持电源，hΔK 的持续性活动立即坍缩归零（**旧目标记忆被瞬间擦除**），同时 PFG 立即恢复自由追踪模式，为下一次去抑制写入做好准备。

### 2.4 分裂吸引子与传统单一吸引子的计算特性对比

| 维度 | 传统单一循环吸引子 (Monolithic Attractor) | 分裂吸引子设计 (Split Attractor Design, Nature 2026) | 工程/数字电路等价类比 |
| :--- | :--- | :--- | :--- |
| **网络拓扑** | 单一群体的神经元同时负责接收外部输入和维持循环记忆 | **结构分裂为两层**：**PFG**（无惯性实时追踪层）+ **hΔK**（受门控的循环记忆锁存层） | 采样保持电路 (Sample-and-Hold) / D 触发器 (D Flip-Flop) |
| **新记忆写入方式** | 必须用强外部输入克服旧吸引子势阱，把活动峰从旧状态“推”到新状态 | PFG 平时已自由跟随至最新状态；只需对 hΔK **去抑制（松开刹车）**，即可**原地零延迟锁定** | 时钟沿触发锁存（Clock-edge Latching） |
| **旧记忆擦除方式** | 需精确施加反向抵消刺激或等待缓慢衰减，易留残余偏置 | 对 hΔK **施加全局重抑制**，切断循环回路，旧记忆瞬间坍缩，且完全不干扰 PFG 继续感知 | 异步清零端（Asynchronous Clear / Reset） |
| **抗噪与防漂移能力** | 易受瞬时高频噪声驱动而发生随机漂移 | **慢速调质兴奋（低通滤波积分）+ 快速侧向抑制（单峰锐化）**，锁存后极度稳固 | 低通滤波锁相环（PLL） |

---

## 4. 第三部分：融合方案设计——基于 Split Attractor 重构 MetaCoRSI 记忆机制

### 3.1 神经计算机制与 MetaCoRSI 组件的精确映射（Isomorphism Mapping）

将 Lanz et al. (2026) 的分裂吸引子机制引入 `MetaCoRSI`，不需要推翻现有的 `IndexedLTM` 原子库与 `Deep RCA` 验证框架，而是对 **`Perception → Grounding → Working Memory → DSH Runtime Loop`** 进行精准的动力学升级：

| 果蝇分裂吸引子环路组件 | MetaCoRSI 架构对应模块 | 改造后的功能定义 |
| :--- | :--- | :--- |
| **内部指南针与感官输入**<br>*(E-PG Compass & Odor Cues)* | **容器实时观测与 Deep RCA**<br>*(`observation`, `DeepSWEAgentRCA`, `_confirmed_focused_failure`)* | 每一轮 Shell 执行输出、报错堆栈、AST 符号探测结果，以及 RCA 五层归因（Logic/Boundary/Schema/Domain/FDE）的 `walk_back` 验证信号。 |
| **PFG 神经元群体**<br>*(实时追踪层 / Pointer)* | **PFG 实时因果追踪器**<br>*(`PFGSituationTracker`)* | **无历史惯性的实时状态指针**：剥离静态 `task_id` 词汇偏置，每步仅根据最新容器证据与 RCA 输出，实时计算当前 `(capability, causal_locus, failure_mode, live_symbols)` 的瞬时激活分布。 |
| **hΔK 神经元群体**<br>*(循环记忆维持层 / Latch)* | **hΔK 目标记忆锁存胶囊**<br>*(`hDKGoalAttractor` / `GroundingRecord`)* | **受门控的持久工作记忆**：一旦与 PFG 耦合锁存，持续向 DSH Agent 输出稳定的单一修复目标（选定的 `AbstractKnowledge` + 逆向绑定的 `source_episode` R/F/S + PFG 捕获的当前代码符号）。 |
| **慢速循环兴奋**<br>*(Slow Excitation: Tyramine/MIP/Dh31)* | **跨步证据慢积分 + Verifier 信用**<br>*(Slow Evidence Integrator + `grounding_credit`)* | 候选抽象 Schema 的兴奋度由跨步指数移动平均（EMA）与历史独立 Verifier 验证信用（`verified` vs `rejected`）共同维持，过滤单步 Shell 输出的偶然词汇噪声。 |
| **快速突触抑制**<br>*(Fast Lateral Inhibition)* | **候选 Schema 侧向竞争抑制**<br>*(Winner-Take-All Lateral Penalty)* | 在候选 `AbstractKnowledge` 之间施加快速侧向抑制，压制次优及因果冲突的 Schema，确保锁存时只形成**单一、高纯度的因果活动峰**。 |
| **靶向 hΔK 的抑制/去抑制门控**<br>*(Inhibition / Disinhibition Gate)* | **运行时三态门控控制器**<br>*(`DisinhibitoryGateController`)* | **DIAGNOSE 探索期强抑制 hΔK**（防负迁移）→ **确认反例/因果锚点瞬间去抑制锁存**（零延迟精确写入）→ **遭遇 Stagnation/RCA 门控失败时重抑制擦除**（快速逃逸重路由）。 |

### 3.2 分裂吸引子工作记忆（SA-WM）的四层运作机制设计

#### 第一层：PFG 实时因果追踪器（Zero-Inertia Live Situation Pointer）
在现有的 `indexed_ltm.py` 中，`_situation_capability_affinities(task_id, task_text)` 将 `70%` 的权重固定给了 `task_id` 字符串。在分裂吸引子设计中，我们将其重构为实时追踪向量 $\mathbf{p}_t$：
- 在第 $t$ 步，PFG 仅读取当前步的最新有效观测 $\mathcal{O}_t$（含测试报错、定位到的文件路径、AST 节点/函数符号、RCA 诊断记录 $D_t$）：
  $$\mathbf{p}_t = \Phi_{	ext{PFG}}(\mathcal{O}_t, D_t) = ig(C_t,\; L_t,\; F_t,\; \mathcal{S}_t^{	ext{live}}ig)$$
  其中 $C_t \in \Delta^{	ext{cap}}$ 为瞬时能力族分布，$L_t \in \Delta^{	ext{locus}}$ 为瞬时因果位置分布，$F_t$ 为当前失败模式，$\mathcal{S}_t^{	ext{live}}$ 为从容器输出中实时提取的高置信代码符号集合（真实文件路径、类名、函数名、报错行关联 AST 节点）。
- **关键特性**：当 hΔK 处于被抑制状态时，$\mathbf{p}_t$ 不受任何跨任务历史 Schema 的拉扯，能够随着 Agent 在 `DIAGNOSE` 阶段的 `grep`/`pytest` 探测零延迟地跳转到真实根因位置。

#### 第二层：hΔK 慢兴奋-快抑制目标吸引子动力学（Slow-Excitation / Fast-Inhibition Dynamics）
对于 `IndexedLTM` 中的每一个候选三维抽象 $a_i \in \mathcal{A}$：
1. **瞬时共振输入（PFG → hΔK 输入）**：
   $$u_i(t) = w_c \cdot C_t(a_i.	ext{cap}) + w_l \cdot L_t(a_i.	ext{locus}) + w_g \cdot 	ext{TriggerOverlap}(\mathcal{O}_t, a_i.	ext{tags})$$
2. **慢速循环兴奋积分（Slow Recurrent Excitation）**：
   $$E_i(t) = (1 - lpha_{	ext{slow}}) E_i(t-1) + lpha_{	ext{slow}} \cdot u_i(t) + eta_{	ext{loop}} \cdot A_i(t-1) + \eta \cdot 	ext{VerifierReliability}(a_i) - R_i(t)$$
   - 其中 $lpha_{	ext{slow}} \in [0.25, 0.40]$ 为慢时间尺度积分系数（模拟神经肽慢动力学，只有连续多步观测一致支持同一因果位置时，$E_i(t)$ 才会稳步爬升）；
   - $eta_{	ext{loop}} \cdot A_i(t-1)$ 为 hΔK–PFG 闭环后的自维持正反馈项；
   - $R_i(t)$ 为重抑制擦除后对该抽象施加的**不应期惩罚项（Refractory Inhibition）**。
3. **快速侧向抑制与门控输出（Fast Lateral Inhibition & Gate）**：
   $$A_i(t) = \sigma\Bigg(E_i(t) - \gamma_{	ext{fast}} \sum_{j 
eq i} 	ext{Sim}(a_i, a_j) E_j(t) - I_{	ext{gate}}(t)\Bigg)$$
   - 当门控抑制 $I_{	ext{gate}}(t) = I_{	ext{high}}$ 时，所有跨任务抽象的激活输出 $A_i(t) = 0$（hΔK 完全静默）；
   - 当触发去抑制 $I_{	ext{gate}}(t) = 0$ 时，快速侧向抑制 $\gamma_{	ext{fast}}$ 迅速压灭次优竞争者，仅保留唯一胜出的峰值抽象 $a^* = rg\max_i A_i(t)$。

#### 第三层：去抑制门控（Disinhibitory Gating）与符号级延迟绑定（Late Symbol-Level Grounding）
系统在单个 Trial 内部维护一个三态门控状态机：

1. **状态 1：`INHIBITED_TRACKING`（强抑制追踪态，默认初始状态，对应 `DIAGNOSE` 阶段）**
   - **工作记忆内容**：仅加载同任务已验证的 Slow Store 骨干锚点（`RETENTION ANCHOR`）或同任务 near-miss 增量（若有）；**跨任务 `AbstractKnowledge` 与 `SOURCE EPISODE` 被强抑制，不注入执行指令**。
   - **行为导向**：Agent 专注于在 `/app` 中复现具名反例、追踪调用链，PFG 指针在后台自由更新 $\mathbf{p}_t$ 与实时代码符号表 $\mathcal{S}_t^{	ext{live}}$。

2. **状态 2：`DISINHIBITED_LATCHED`（去抑制锁存态，对应“气味触发”后的 `MATERIALIZE / VERIFY` 阶段）**
   - **双通道去抑制触发条件（Disinhibition Trigger）**：满足以下任一条件时，$I_{	ext{gate}}(t)$ 瞬间降为 $0$：
     - **通道 A（反例执行确认）**：`_confirmed_focused_failure()` 捕获到真实测试命令失败，且 RCA 归因置信度 $\ge 	heta_{	ext{rca}}$（或 `walk_back_ok == True`）；
     - **通道 B（源码因果结构确认）**：对于前期以静态结构检查为主的任务，PFG 指针 $\mathbf{p}_t$ 在连续 2 步内稳定指向同一非空 `causal_locus`（非 `unspecified`），且捕获到明确的仓库源码符号 $\mathcal{S}_t^{	ext{live}}$ 与 `trigger_tags` 共振。
   - **原地锁存与符号级 Grounding（Rapid Writing）**：
     去抑制瞬间，hΔK 将 PFG 当前时刻的状态 $\mathbf{p}_t$ 原地锁存为当前导航目标 `GroundingRecord`：
     - 不仅绑定抽象的 `{capability}` 和 `{causal_locus}`，还将 PFG 刚刚捕获到的**当前仓库真实文件与函数/AST 符号 $\mathcal{S}_t^{	ext{live}}$** 直接写入 `GroundingRecord.bindings["live_causal_symbols"]`；
     - 在接下来的 `MATERIALIZE` 与 `VERIFY` 步中，$eta_{	ext{loop}}$ 自维持正反馈保持该目标稳定不变，不再因中间编译或日志输出的无关词汇而漂移。

3. **状态 3：`REINHIBITED_RESET`（重抑制擦除与重路由态，对应死胡同逃逸）**
   - **重抑制触发条件（Re-inhibition Trigger）**：在 `DISINHIBITED_LATCHED` 状态下，若出现以下任一负反馈：
     - 连续 $K=2$ 步修改代码后，聚焦测试或编译结果签名（`result_signature`）毫无进展（`stagnation` 递增）；
     - 触发了新的 `preservation_regression`（破坏了原本通过的测试）；
     - Agent 尝试提交 `done=true` 时被 `RCA FOCUSED GATE BLOCKED COMPLETION` 拦截。
   - **快速擦除与不应期（Rapid Erase & Refractory Period）**：
     - 立即将 $I_{	ext{gate}}(t)$ 拉回高抑制状态 $I_{	ext{high}}$，**瞬间清空当前锁存的 `hDKGoalAttractor`**；
     - 对刚刚失败的抽象 $a^*$ 施加不应期惩罚 $R_{a^*}(t) \leftarrow +\infty$（本轮 Trial 内不再重复激活该失败 Schema）；
     - 向 DSH 会话显式发送一条高优先级指令 **`[SPLIT-ATTRACTOR DISINHIBITORY RESET]`**，强制清除 LLM 会话历史中的“幽灵先验”，让 PFG 基于最新失败反例立即重定位并在下一步重新去抑制锁定备选假设（Alternative Hypothesis）。

#### 第四层：跨 Episode 的 CLS 慢/快存储协同进化
在 Episode 结束进入 `evolve_from_episodes()` 时，分裂吸引子的运行轨迹（`grounding_history` 中的去抑制锁存记录与重抑制擦除记录）直接赋能 LTM 的 Consolidation：
- **历经去抑制锁存且最终通过 Verifier（`reward == 1`）的 `GroundingRecord`**：记为 `verified`，增强该 `AbstractKnowledge` 在 hΔK 中的慢速基础兴奋权重；
- **在运行时触发了重抑制擦除（Re-inhibited）或最终 Verifier 失败的 `GroundingRecord`**：记为 `rejected` 并记录触发擦除的具体反例签名，精准收缩该 Schema 的适用边界（Scope Guard），从根源上迭代消除跨任务负迁移。

---

### 3.3 核心代码模块改造落点规划

基于现有 `MetaCoRSI` 工程目录，无需破坏任何外部接口或 Verifier 契约，只需在以下 3 个核心文件中完成模块化接入：

| 目标文件 | 现有核心类/函数 | 拟改造/新增内容 |
| :--- | :--- | :--- |
| `metacorsi/metacorsi/knowledge/indexed_ltm.py` | `_MEMORY_STRATEGIES`、`_strategy_score()`、`IndexedLTM.ground()` | 1. 在 `_MEMORY_STRATEGIES` 中新增 `"split_attractor"` 策略与 `"split_attractor_v24"` 策略模式。<br>2. 新增 `track_pfg(observation, rca_record)`：计算无静态 `task_id` 偏置的瞬时因果坐标与 `live_symbols`。<br>3. 重构 `ground()` 支持传入 `pfg_state`、`hdk_prior_state`、`inhibition_gate` 与 `refractory_ids`，实现慢兴奋积分 + 快侧向抑制评分。 |
| `metacorsi/metacorsi/cognition.py` & `brain.py` | `WorkingMemory`、`MetaCoRSIBrain` | 1. 在 `WorkingMemory` 中新增分裂吸引子状态字段：`pfg_pointer`、`hdk_latched_grounding`、`gate_state` (`INHIBITED` / `DISINHIBITED` / `REINHIBITED`)、`refractory_abstract_ids`。<br>2. 在 `MetaCoRSIBrain` 中提供 `step_split_attractor(observation, rca_diagnosis, confirmed_failure, gate_blocked)` 接口。 |
| `metacorsi/metacorsi_dsh_agent/__init__.py` | `AgentHarness.run()`、`_lock_grounding_receipt()` | 1. 将原有的 `failure_locked_v21` 单向锁升级为 `split_attractor_v24` 三态门控循环。<br>2. Step 0 默认处于 `INHIBITED_TRACKING`（仅加载同任务 Slow Backbone，抑制跨任务 Schema）。<br>3. 双通道检测去抑制信号后锁存 hΔK，并在触发 `RCA FOCUSED GATE BLOCKED` 或连续失败时触发 Re-inhibition Reset 及会话重置提示。 |

---

### 3.4 预期收益分析（以具体难题为例）

1. **彻底消除无同任务记忆时的 Step 0 负迁移（如 `Clack`、`Claude delegation`、`Adaptix`）**
   - **现状**：Step 0 题面关键词误触发不匹配的跨任务 Schema，Agent 从第 1 步就开始按错误模板寻找并不存在的边界。
   - **引入 Split Attractor 后**：`DIAGNOSE` 阶段（Step 1~4）hΔK 处于强抑制状态，Agent 完全依据 `/app` 真实源码进行探测；直到 PFG 在真实代码中确认了因果边界才去抑制写入，若无匹配的因果证据则保持抑制，**实现“无确定因果证据则零负迁移”**。
2. **攻克需要深层结构探测的 Fail@5 任务（如 `Arktype` 与 `Bandit`）**
   - **`bandit-structured-nosec-directives`**：Agent 往往在初期只看到物理行号（physical line），若过早锁定修复模式，就会写出按行传播的补丁。在分裂吸引子机制下，前期 PFG 追踪 AST 结构，即使初次尝试按物理行修复被 `RCA FOCUSED GATE` 或多行语句反例拦截，**重抑制门控（Re-inhibition）会立即擦除“物理行传播”假设并将其打入不应期**，迫使 PFG 切换到备选假设（“绑定完整 AST Statement 节点”）并重新锁存。
   - **`arktype-json-schema-refs-dependencies`**：递归 `$ref` 需要在探测到循环引用堆栈后，将具体递归节点 Identity 与 SCC 闭合机制精确绑定。去抑制瞬间锁存能把 PFG 在报错堆栈中捕获的具体引用节点符号直接注入 `GroundingRecord` 的执行槽位中。

---

## 5. 第四部分：面向其他 Agent / 评审专家的开放研讨问题清单

> **说明**：当您将本报告发给其他 AI Agent 或领域专家征求意见时，可以请他们重点围绕以下 5 个具体问题给出批评或改进建议：

1. **去抑制门控（Disinhibitory Gate）的最优开启时机**：
   在代码修复任务（24~30 步预算）中，除了“聚焦测试首次失败（`_confirmed_focused_failure`）”和“Deep RCA `walk_back_ok` 确认”这两个显式信号外，如果某道题在 `DIAGNOSE` 阶段主要通过静态代码阅读（`grep`/`sed`/`python -c`）定位问题而未触发测试报错，**如何设计最稳健的“静态因果确认去抑制条件”**，既防止开门过早（负迁移），又防止开门过晚（浪费步数预算）？
2. **LLM 会话上下文中的“幽灵吸引子（Ghost Attractor）”擦除问题**：
   生物神经网络对 hΔK 施加抑制即可瞬间清空电活动；但在基于 LLM 的 Agent（如使用 `--session-id` 保持多轮上下文的 DSH）中，前几步注入过的错误跨任务 Schema 仍留在对话历史（KV Cache）中。当触发**重抑制擦除（Re-inhibition Reset）**时，**仅靠在最新 Prompt 中注入高优先级的 `[DISINHIBITORY RESET DIRECTIVE]` 文本是否足够？还是应当在重抑制触发时主动截断/重建 Solver Session，仅保留客观的容器命令输出与最新 PFG 状态？**
3. **hΔK 慢速循环兴奋（Slow Excitation）在离散步长下的数学形式**：
   在只有 24 步的离散交互中，hΔK 的慢时间尺度兴奋积分 $E_i(t)$ 与快速侧向抑制 $A_i(t)$ 采用何种参数化形式最有效（例如：指数滑动平均 EMA + 硬性 Top-1 Winner-Take-All vs. 带温度退火的 Softmax 竞争 vs. 贝叶斯后验对数似然累积）？
4. **分裂吸引子与 CLS 双存储（Slow Backbone vs. Fast Delta）的耦合边界**：
   目前我们的设计是：**同任务的 Slow Backbone 锚点（Retention Anchor）不受 hΔK 抑制门控限制（Step 0 即驻留），只有跨任务的 `AbstractKnowledge` / `SOURCE EPISODE` 以及未验证的候选假设受 PFG–hΔK 去抑制门控调控**。这种“同任务锚点常驻 + 跨任务/增量假设门控锁存”的划分是否最优？同任务的 near-miss 失败增量（K29 Fast Store）是否也应该纳入分裂吸引子的重抑制擦除范围？
5. **符号级 Grounding（Symbol-Level Latching）的粒度设计**：
   当去抑制门控打开、将 PFG 追踪到的瞬时状态写入 hΔK 时，除了绑定现有的 5 维宏坐标 `(task, repo, capability, failure_mode, causal_locus)` 外，**最应该从 PFG 瞬时观测中提取哪些具体代码结构特征（如 AST 节点类型、调用栈顶端函数签名、修改文件集合 signature）锁存进 `GroundingRecord` 的动作模板中？**
