import numpy as np

from miniflex.common.hash import gen_block_hashes


def test_hash():
  prefix = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
  full = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)

  prefix_hashes = gen_block_hashes(prefix, tokens_per_block=4, namespace=b"ns")
  full_hashes = gen_block_hashes(full, tokens_per_block=4, namespace=b"ns")
  other_hashes = gen_block_hashes(prefix, tokens_per_block=4, namespace=b"other")

  assert prefix_hashes.dtype == np.uint64
  assert len(prefix_hashes) == 2
  assert np.array_equal(prefix_hashes, full_hashes)
  assert not np.array_equal(prefix_hashes, other_hashes)


if __name__ == "__main__":
  test_hash()
