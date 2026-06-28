# 本测试文件由 Claude (Anthropic) 编写。
# 测试内容：SequenceMeta 的 block 切分与 namespace/salt 哈希隔离。

import miniflex.common.block as block
import numpy as np


def test_block():
  prefix = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
  full = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)

  prefix_meta = block.SequenceMeta(prefix, tokens_per_block=4, namespace=["ns"])
  full_meta = block.SequenceMeta(full, tokens_per_block=4, namespace=["ns"])
  other_meta = block.SequenceMeta(prefix, tokens_per_block=4, namespace=["other"])

  assert prefix_meta.length == 8
  assert full_meta.num_blocks == 2
  assert prefix_meta.has_hashed()
  assert np.array_equal(prefix_meta.block_hashes, full_meta.block_hashes)
  assert not np.array_equal(prefix_meta.block_hashes, other_meta.block_hashes)
  assert prefix_meta.get_hash(0) == int(prefix_meta.block_hashes[0])
  assert isinstance(prefix_meta.get_hash(0), int)
  assert prefix_meta.get_hash(2) is None


if __name__ == "__main__":
  test_block()
