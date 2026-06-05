# Mempool Lazy Recycle

## 问题

`Mempool` 管理的是物理 block id。核心问题是：

> block 被回收后，要不要立刻重建空闲 block 列表？

如果每次 recycle 都重建，行为最直观，但开销偏大。FlexKV 选择把这部分成本延迟到 allocate 路径上。

## FlexKV 的设计

FlexKV 的 `Mempool` 维护了一份 `_free_ids` 快照，并用 `_free_ids_offset` 记录当前快照中下一个可分配位置。

分配 block 时，它不会每次都重新扫描所有空闲块，而是直接从当前快照里切出一段：

```python
_free_ids[_free_ids_offset: _free_ids_offset + num]
```

回收 block 时，FlexKV 不会立刻修改 `_free_ids_offset`，也不会立刻重建 `_free_ids`。它只更新 free mask 和 free block 计数。

直到后续某一次 allocate 发现当前快照剩余数量不足：

```python
len(_free_ids) - _free_ids_offset
```

才重新扫描 free mask，重建 `_free_ids`。

### 影响

- 优点：recycle 路径很轻，不会频繁执行全量扫描。
- 代价：刚释放的 block 不一定马上进入下一次分配结果。
- 触发重建的时机：旧 `_free_ids` 快照不够满足本次 allocate。

## MiniFlex 的折中

我最初考虑过在每次 recycle 时都立即重建 `_free_ids`。这样行为最直观：释放后的 block 下一次就可以被分配到。但这个方案会把重建空闲列表的成本放到每一次 recycle 上，开销偏大。

MiniFlex 当前采用折中方案：增加 `_is_dirty` 标记。

回收时仍然不立刻重建 `_free_ids`，只做三件事：

1. 更新 free mask。
2. 更新 free block 计数。
3. 将 `_is_dirty` 置为 `True`。

下一次 allocate 时，如果发现 `_is_dirty == True`，再重建 `_free_ids`。

## 对比

| 方案 | recycle 时重建 `_free_ids` | allocate 时重建 `_free_ids` | 特点 |
| --- | --- | --- | --- |
| 每次回收立即重建 | 是 | 否 | 行为直观，但 recycle 开销大 |
| FlexKV | 否 | 仅当旧快照不够用 | 最 lazy，减少回收路径开销 |
| MiniFlex 当前方案 | 否 | 只要 `_is_dirty` 为真 | 更保守，调试更直接 |
