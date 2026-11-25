# MetaCoRSI 1116 算法说明

## 1. 自进化算法

```text
Task / repository
  → Perception：构造 situation、capability、causal-locus 查询
  → Working Memory：5 个 stable atoms + 最多 3 个 exact-task delta atoms
  → Top-1 consolidated schema + target-disjoint source episode
  → DSH Agent：DIAGNOSE → MATERIALIZE → VERIFY → CLOSE
  → Deep RCA：Logic / Boundary / Schema / Domain / FDE walk
  → independent verifier：F2P + P2P
  → evolution write-back
      pass：蒸馏并晋升任务 episode
      fail：保留 candidate、反例和 grounding feedback，不作为通用技能晋升
  → CLS consolidation
  → replay / promotion
```

基础预算为 24 步。只有 materialized candidate 获得 focused-test 成功证据时才允许增加最多 6 步。最终成功只由独立 verifier 判定。

## 2. 知识组织算法

### Slow store

K20/215 是稳定主干，保存已经形成的任务契约、因果模式、动作与保持边界。新增经验不能通过普通排序把它从 WM 中挤掉。

### Fast store

所有 K20 之后新增的原子作为 episode delta 保存，并带有 task、repository、capability、causal locus、outcome 和来源。fast atom 只能直接进入自身任务的 capsule。

### Consolidation

每次 episode 写回后，系统从 `stable backbone + verified deltas` 重建 abstraction graph。跨任务使用的不是原始 episode，而是 consolidation 后的机制 schema。

### Grounding

```text
同任务：stable M/F/R/S/P + exact-task delta R/F/M
跨任务：Top-1 schema → Top-1 target-disjoint source episode → 最多 R/F/S
```

抽象只提供机制假设，必须绑定当前仓库的 state、boundary、consumer roles；源任务文件名、符号和补丁不得直接复制，除非当前仓库独立存在同一证据。

## 3. 为什么 11/20 优于原始 K28

原始 K28 的 251 个原子没有 `memory_layers`，运行时仍会扁平竞争。K28-CLS manifest 明确记录 215 个 backbone IDs 和 36 个 delta IDs，因此 `_layered_capsule()` 真正激活：

- K20 的 5 个严格通过全部保持；
- Actionlint、Cattrs、Cliffy 的新增 episode 被正确召回；
- Anko typed、Bandit taint、Dasel 出现新的严格通过；
- 最终由 5/20 提升到 11/20。

当前缺点是 slow schema 仍可能对没有 exact-task fast episode 的任务产生负迁移，例如 Clack、Claude delegation、Adaptix。后续改进应基于 grounding receipt 调整 schema 的适用域和反证，而不是继续无差别增加原子。
