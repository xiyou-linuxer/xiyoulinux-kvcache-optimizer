import numpy as np
from typing import Optional
from miniflex.common.hash import Hasher, gen_block_hashes
from typing import List
from dataclasses import dataclass, field



def _get_namespace_hash_key(namespace: Optional[List[str]]) -> Optional[bytes]:
  """
  Encode namespace components into stable bytes for cache isolation.
  """
  if namespace is None:
    return None
  if len(namespace) == 0:
    return None
  namespace_str = "\x00".join(namespace)
  return namespace_str.encode("utf-8")


def hash_token(token_ids : np.ndarray, namespace: Optional[List[str]] = None) -> int:
  """
  Hash the whole token sequence with an optional namespace prefix.
  """
  namespace_hash_key = _get_namespace_hash_key(namespace)
  hasher = Hasher()
  if namespace_hash_key is not None:
    hasher.update(namespace_hash_key)
  hasher.update_numpy(token_ids)
  return hasher.digest()


@dataclass
class SequenceMeta:
  """
  Metadata for a token sequence and its complete-block prefix hashes.
  """
  token_ids: np.ndarray
  tokens_per_block: int
  namespace: Optional[List[str]] = None
  _namespace_hash_key: Optional[bytes] = field(init=False, default=None)
  block_hashes: np.ndarray = field(init=False)
  _has_hashed: bool = field(init=False, default=False)
  
  
  def __post_init__(self):
    """
    Validate input tokens and eagerly generate block hashes.
    """
    if self.token_ids.ndim != 1:
      raise ValueError(f"token_ids must be a 1D array, got {self.token_ids.ndim}D")
    if self.token_ids.dtype != np.int64:
      raise ValueError(f"token_ids must be a 1D array of int64, got {self.token_ids.dtype}")
    if self.tokens_per_block <= 0:
      raise ValueError(f"tokens_per_block must be greater than 0, got {self.tokens_per_block}")
    self.gen_hashes()
    
  def gen_hashes(self):
    """
    Generate block hashes once and cache them on the sequence.
    """
    if self._has_hashed:
      return
    if self.namespace is not None:
      if self._namespace_hash_key is None:
        self._namespace_hash_key = _get_namespace_hash_key(self.namespace)
    self.block_hashes = gen_block_hashes(self.token_ids, self.tokens_per_block, self._namespace_hash_key)
    
    assert self.block_hashes.ndim == 1
    assert len(self.block_hashes) == self.num_blocks
    self._has_hashed = True
  
  def has_hashed(self) -> bool:
    """
    Return whether block hashes have already been generated.
    """
    return self._has_hashed
  
  def get_hash(self, block_idx: int) -> Optional[int]:
    """
    Return the block hash as a Python int, or None if out of range.
    """
    if block_idx < 0 or block_idx >= self.num_blocks:
      return None
    return int(self.block_hashes[block_idx])
  
  @property 
  def length(self) -> int:
    """
    Return the number of tokens in the sequence.
    """
    return len(self.token_ids)
  
  @property
  def num_blocks(self) -> int:
    """
    Return the number of complete token blocks.
    """
    return len(self.token_ids) // self.tokens_per_block
