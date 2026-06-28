# 本测试文件由 Claude (Anthropic) 编写。
# 测试内容：Mempool 物理 block-id 池的分配、回收与非法输入校验。

from miniflex.cache.mempool import Mempool

def test_mempool():
  mempool = Mempool(10)
  allocated_first = mempool.allocate(3)
  print(allocated_first)
  mempool.recycle(allocated_first)
  print(mempool.num_free_blocks)
  print(mempool.num_used_blocks)
  for invalid_blocks in ([-1], [10]):
    try:
      mempool.recycle(__import__("numpy").array(invalid_blocks, dtype=__import__("numpy").int64))
    except ValueError as exc:
      print(type(exc).__name__)
    else:
      raise AssertionError(f"expected ValueError for {invalid_blocks}")

if __name__ == "__main__":
  test_mempool()
