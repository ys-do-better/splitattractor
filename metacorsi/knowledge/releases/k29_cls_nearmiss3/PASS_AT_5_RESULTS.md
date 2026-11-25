# K29 DeepSWE-20 Pass@5

冻结知识：K29-CLS-NearMiss3，263 atoms。

最终结果：**18/20（90%）**。

评测约束：相同 K29、相同 DSH/DeepSeek 配置、每题最多五次独立轨迹、通过即停止、知识只读、禁止 candidate patch replay、禁止访问 gold/reference patch。

## 通过任务

- abs-module-cache-flags
- abs-stepped-slices
- actionlint-action-pinning-lint
- adaptix-name-mapping-aliases
- aiomonitor-task-snapshots-diff
- anko-default-function-arguments
- anko-typed-variable-bindings
- arcane-drift-detection-baselines
- awilix-async-container-initialization
- bandit-incremental-cache-control
- bandit-interprocedural-taint-checks
- boa-hierarchical-evaluation-cancellation
- cattrs-partial-structuring-recovery
- clack-async-autocomplete-options
- claude-code-by-agents-recursive-delegation
- cliffy-config-file-parsing
- csstree-shorthand-expansion-compression
- dasel-html-document-format

## Fail@5

### Arktype JSON Schema refs/dependencies

- 最好：F2P 24/25，P2P 1679/1679。
- 后四次稳定：F2P 23/25，P2P 1679/1679。
- 根因：递归 `$ref` 仍使用 `Map + lazy closure`，没有递归节点占位、稳定 identity 和 SCC 闭合。

### Bandit structured nosec

- F2P 69/69。
- 后四次稳定：P2P 281/282。
- 根因：suppression 仍按物理行单向传播，没有对完整 AST statement 做 selector 聚合与反向 grounding。

## 口径

这是 task-wise Pass@5，不是 one-shot 18/20，也不是 sequential cumulative evolution score。各 attempt 之间 K29 保持冻结，因此 Pass@5 测量的是当前策略在五次采样内产生成功轨迹的能力。
