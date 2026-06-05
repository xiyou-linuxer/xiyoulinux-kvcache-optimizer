import hashlib
from typing import Optional
import numpy as np

class Hasher:
  """Incremental hasher producing 64-bit unsigned hash values."""
  
  def __init__(self):
    self._hasher = hashlib.blake2b(digest_size=8)
    
  def reset(self) -> "Hasher":
    "Reset the hasher to initial state."
    self._hasher = hashlib.blake2b(digest_size=8)
    return self
    
  def update(self, data: bytes) -> "Hasher": 
    "Feed bytes into the hasher."
    self._hasher.update(data)
    return self
  
  def update_numpy(self, array: np.ndarray) -> "Hasher":
    "Feed numpy array's raw bytes into the hasher."
    if not array.flags.c_contiguous:
      array = np.ascontiguousarray(array)
    self._hasher.update(memoryview(array))
    return self
  
  def digest(self) -> int:
    "Return the current hash value as int."
    return int.from_bytes(self._hasher.digest(), byteorder="little")
  
def _gen_block_hashes_impl(token_ids: np.ndarray, tokens_per_block: int, hasher: Hasher) -> np.ndarray:
  if token_ids.ndim != 1:
    raise ValueError(f"token_ids must be 1D, got {token_ids.ndim}D")
  if token_ids.dtype != np.int64:
    raise ValueError(f"token_ids must be int64, got {token_ids.dtype}")
  if tokens_per_block <= 0:
    raise ValueError(f"tokens_per_block must be positive, got {tokens_per_block}")
  
  num_blocks = len(token_ids) // tokens_per_block
  result = np.zeros(num_blocks, dtype=np.uint64)
  for i in range(num_blocks):
    block = token_ids[i * tokens_per_block:(i + 1) * tokens_per_block]
    hasher.update_numpy(block)
    result[i] = hasher.digest()
  
  return result

def gen_block_hashes(token_ids: np.ndarray, tokens_per_block: int,namespace: Optional[bytes] = None, hasher: Optional[Hasher] = None) -> np.ndarray:
  """
  Compute prefix hash for each block of tokens.
  
  Args:
    token_ids: 1D array of int64 token IDs.
    tokens_per_block: Number of tokens per block.
    namespace: Optional namespace bytes to hash before token blocks.
    hasher: Optional Hasher to reuse. A fresh one is created if None.
  Returns:
    1D uint64 array of length len(token_ids) // tokens_per_block.
  """
  if hasher is None:
    hasher = Hasher()
  else:
    hasher.reset()
  if namespace is not None:
    hasher.update(namespace)
  return _gen_block_hashes_impl(token_ids, tokens_per_block, hasher)