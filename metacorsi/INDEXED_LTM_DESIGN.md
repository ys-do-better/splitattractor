# MetaCoRSI Indexed LTM

## 目标

知识进化时不再把新原子直接追加到一个平面列表后参与全局 Top-K。每次 verifier episode
产生的 P/M/R/F/S 原子会进入不可变原子库，并同时建立稳定的多维索引。Working Memory
加载的是由 Perception 路由得到的知识子图。

## 存储布局

```text
ltm/
├── CURRENT.json
└── versions/
    └── v11/
        ├── atoms.jsonl
        ├── nodes.json
        ├── edges.jsonl
        ├── retention_anchors.json
        ├── indexes/
        │   ├── task.json
        │   ├── repository.json
        │   ├── capability.json
        │   ├── failure_mode.json
        │   └── causal_locus.json
        └── manifest.json
```

`manifest.json` 保存所有文件的 SHA256。只有整个版本写完后才原子更新 `CURRENT.json`。

## 进化事务

```text
Verifier episode
  → distill P/M/R/F/S atoms
  → append to VersionedKnowledgeStore
  → assign task/repository/capability/failure/causal coordinates
  → rebuild relation nodes and edges
  → preserve verified-task retention anchors
  → write immutable indexed LTM version
  → atomically advance CURRENT.json
```

## 检索顺序

```text
1. exact-task retention anchors
2. exact task branch
3. repository branch
4. capability branch
5. matched failure mode
6. matched causal locus
```

每个分支有独立配额。新增其他任务的原子不能占用 exact-task anchor 配额。

## 抽象节点晋升

| 独立成功 grounding | Capability 状态 |
|---:|---|
| 0 | unverified |
| 1 | provisional |
| 2 | validated |
| ≥3 且无未解决负 grounding | transferable |

原始 task atoms 永远保留；能力节点只建立 `ABSTRACTS_FROM` 关系，不覆盖具体经验。

## 当前离线验收

- 全套测试：40 passed。
- v11/149：56 nodes，917 edges，9 组 retention anchors。
- 索引键：20 task、17 repository、7 capability、5 failure mode、7 causal locus。
- 增加 500 个无关失败原子后，9/9 retention anchor 路径与顺序完全稳定。

注意：当前 v11 是从冻结 v15 与 evolution receipts 按追加顺序重建的审计快照，原始中间
store 当时没有保存。它包含 9 个带 `pass` 证据的任务分支，因此不能把“9 anchors”直接
解释为一次正式 9/20 benchmark 分数；它只表示 LTM 中存在 9 组已验证知识记录。

离线报告：`experiments/indexed_ltm_v1/offline_audit/OFFLINE_AUDIT.md`。

## 结论边界

当前已经证明结构保持：知识增长不会使旧能力原子从 Working Memory 的稳定入口消失。
还没有证明行为保持：模型在冻结 Indexed LTM 下是否仍能完成旧任务。行为保持必须通过
禁止 candidate patch replay、禁止知识写回的 DeepSWE retention evaluation 验证。
