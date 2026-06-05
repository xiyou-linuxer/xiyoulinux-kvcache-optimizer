# Radix Tree 设计说明

MiniFlex 的 radix tree 是自研实现，FlexKV 只作为结构和语义参考。当前目标不是和 FlexKV 完全对接，而是先提供一个适合 MiniFlex 单机单卡 CPU/SSD 缓存路径的前缀索引。

## 核心语义

- key 是 `SequenceMeta.block_hashes` 中的 block hash。
- child map 使用 Python `int` 作为 key，和 `SequenceMeta.get_hash()` 返回值一致。
- 每个非 root 节点保存一段压缩后的连续 block：
  - `block_hashes: np.ndarray`
  - `physical_block_ids: np.ndarray`
- `block_hashes` 和 `physical_block_ids` 长度必须一致。
- root 是空节点，始终 ready，永远不进入 `leaf_nodes`。
- `is_leaf()` 明确排除 root：只有非 root 且无 children 的节点才是 leaf。

## Root 和 leaf_nodes

当前设计中 `leaf_nodes` 只跟踪可作为淘汰候选入口的非 root leaf。

这意味着：

- 空树时 root 没有 children，但 root 不是 leaf。
- 空树时 `leaf_nodes` 为空，`is_empty()` 返回 true。
- 一个普通缓存节点插入后，如果没有 children，会进入 `leaf_nodes`。
- split 后 shared-prefix parent 不是 leaf，不进入 `leaf_nodes`。
- eviction 删除最后一个 child 后，如果 parent 变成非 root leaf，parent 会进入 `leaf_nodes`。

这个设计和 FlexKV 中 root 不作为普通 leaf 管理的行为一致，但 MiniFlex 明确把它写进 `is_leaf()` 语义。

## MatchResult

`match_prefix(sequence)` 返回 `MatchResult`：

- `num_matched_blocks`: 命中的 block 数，包括 ready 和 pending 节点。
- `num_ready_matched_blocks`: 已 ready 的命中 block 数。
- `last_ready_node`: 最后一个 ready 节点。
- `last_node`: 匹配停止时所在节点。
- `last_node_matched_length`: 在 `last_node` 内匹配到的 block 数。
- `physical_block_ids`: 已匹配 block 对应的物理 block id，dtype 为 `np.int64`。

pending 节点会贡献 `num_matched_blocks` 和 `physical_block_ids`，但不会贡献 `num_ready_matched_blocks`。

## 插入语义

`insert(sequence, physical_block_ids, match_result=None, is_ready=True)` 插入的是完整未命中后缀。

约束：

- `physical_block_ids` 必须是一维 `np.int64`。
- `len(physical_block_ids) == sequence.num_blocks - match_result.num_matched_blocks`。
- 不支持 `num_insert_blocks`；MiniFlex 当前默认插入全部未命中后缀。
- 如果序列已经完整命中，返回 `None`，不修改树。
- 传入的 `physical_block_ids` 是未命中后缀的物理 block id，不是整条序列的物理 block id。

部分匹配到现有节点中间时会 split：

```text
已有节点: [A, B, C, D]
新序列:   [A, B, X, Y]

插入后:
        [A, B]
        /    \
    [C, D]  [X, Y]
```

split 时 MiniFlex 采用 FlexKV 风格：

- 新建 shared-prefix parent。
- 原 node 缩成 suffix child。
- 原 node 对象身份保留，避免外部持有 node 时语义突然变成 prefix parent。

当前不会在 insert 后自动合并 single-child parent。这样可以保留 node identity 和 prefix 边界，后续如果需要压缩树形，可以单独设计 merge 策略。

## Ready 和 pin

节点有两个和淘汰相关的状态：

- `_is_ready`: 数据是否已经可用。
- `_pin_count`: 是否被外部逻辑保护，不能淘汰。

`is_in_use()` 当前定义为：

```text
_pin_count > 0 or not _is_ready
```

因此 pending 节点和 pinned 节点都不可淘汰。

`set_ready(node, True)` 用于把 pending 节点标记为 ready。当前实现不支持 ready 回滚；对已经 ready 的节点调用 `set_ready(node, False)` 会报错。

## 驱逐语义

`evict(num_evict_blocks)` 返回被释放的 physical block id 数组，dtype 为 `np.int64`。

规则：

- `num_evict_blocks < 0` 报错。
- `num_evict_blocks == 0` 返回空 `np.int64` 数组。
- 只从 `leaf_nodes` 中选择 `is_evictable()` 的节点。
- pending 节点不可淘汰。
- pinned 节点不可淘汰。
- 如果 leaf 大小大于剩余待淘汰 block 数，则从节点尾部 shrink。
- 如果 leaf 大小小于或等于剩余待淘汰 block 数，则删除整个 leaf。
- 删除 leaf 后，如果 parent 变成非 root leaf，则把 parent 加入 `leaf_nodes`。
- 当前不在驱逐后自动合并 single-child parent。

当前 `evict()` 只返回 physical block ids。FlexKV 还会返回 evicted block hashes，主要用于事件发布和跨层元数据同步；MiniFlex 后续接入这些需求时再扩展接口。

## 统计接口

- `total_cached_blocks()` 返回树中所有非 root 节点累计保存的 block 数。
- `total_node_size()` 返回非 root 节点数量，不是 block 数。

示例：

```text
空树: 0 blocks, 0 nodes
单节点 [A, B, C]: 3 blocks, 1 node
split 为 [A, B] + 两个 suffix leaf: 4 blocks, 3 nodes
```

`total_node_size()` 名字里的 `size` 容易和 block 数混淆；如果后续要清理接口，可以考虑改名为 `total_node_num()`。

## 当前测试覆盖

测试文件：

```text
test/radix_tree_test.py
```

覆盖方向：

- node 构造校验。
- root 不算 leaf。
- 空树匹配。
- 单序列插入、完整匹配、前缀匹配、未命中匹配。
- 重复插入已缓存序列。
- shared-prefix split 后两个分支都能匹配。
- 插入参数错误时不污染树。
- pending/ready 状态影响 ready match 和 eviction。
- pin/unpin 控制 eviction。
- `evict(0)`、负数参数、部分 shrink、超量 evict。
- split 树删除 leaf 后 parent 变 leaf。
- `reset()`。
- lru/lfu/slru/fifo priority。

运行：

```bash
PYTHONPATH=pysrc python test/radix_tree_test.py
```

直接运行时会打印中文逐项日志，包含每个测试用例的开始、通过和失败状态。
