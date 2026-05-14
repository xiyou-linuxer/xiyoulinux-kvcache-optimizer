import numpy as np

class Mempool:
  """
  A NumPy-based memory pool (block) manager.
    
  Uses a boolean array to track the occupancy status of each block and an internal 
  cache (_free_ids and _free_offset) to accelerate allocation. It employs a 
  "lazy update" mechanism to amortize the performance overhead of recycling.
  """
  def __init__(self, num_blocks: int): 
    if num_blocks <= 0:
      raise ValueError(f"num_blocks must be greater than 0, got {num_blocks}")
    self._num_total_blocks = num_blocks
    self._free_blocks = np.ones(num_blocks, dtype=np.bool_)
    self._free_ids = self._free_blocks.nonzero()[0]
    self._num_free_blocks = num_blocks
    self._free_offset = 0
    self._is_dirty = False
    
  def reset(self):
    """
    Reset the mempool to initial state.
    """
    self._free_blocks = np.ones(self._num_total_blocks, dtype=np.bool_)
    self._free_ids = self._free_blocks.nonzero()[0]
    self._num_free_blocks = self._num_total_blocks
    self._free_offset = 0
        
  def allocate(self, num_blocks: int) -> np.ndarray:
    """
    Allocate a contiguous block of memory.
    Args:
      num_blocks: Number of blocks to allocate.
    Returns:
      A 1D array of block IDs.
    """
    if num_blocks <= 0:
      raise ValueError(f"num_blocks must be greater than 0, got {num_blocks}")
    if num_blocks > self._num_free_blocks:
      raise ValueError(f"num_blocks must be less than or equal to {self._num_free_blocks}, got {num_blocks}")
    if self._is_dirty or num_blocks > len(self._free_ids) - self._free_offset:
      self._update_free_ids()
    free_ids = self._free_ids[self._free_offset:self._free_offset + num_blocks]
    self._free_offset += num_blocks
    self._num_free_blocks -= num_blocks
    self._free_blocks[free_ids] = False
    return free_ids.copy()
    
  def _update_free_ids(self):
    self._free_ids = self._free_blocks.nonzero()[0]
    self._free_offset = 0
  
  def recycle(self, blocks: np.ndarray):
    """
    Recycle a block of memory.
    Args:
      blocks: A 1D array of block IDs.
    """
    if blocks.ndim != 1:
      raise ValueError(f"blocks must be a 1D array, got {blocks.shape}")
    if blocks.dtype != np.int64:
      raise ValueError(f"blocks must be a 1D array of int64, got {blocks.dtype}")
    if len(blocks) == 0:
      return
    if np.any(blocks < 0) or np.any(blocks >= self._num_total_blocks):
      raise ValueError(f"blocks must be in range [0, {self._num_total_blocks}), got {blocks}")
    blocks_id = np.unique(blocks)
    already_free = self._free_blocks[blocks_id]
    if already_free.any():
      blocks_id = blocks_id[~already_free]
      if len(blocks_id) == 0:
        return
    self._free_blocks[blocks_id] = True
    self._num_free_blocks += len(blocks_id)
    self._is_dirty = True
    
  @property
  def num_free_blocks(self) -> int:
    return self._num_free_blocks
  
  @property
  def num_used_blocks(self) -> int:
    return self._num_total_blocks - self._num_free_blocks
  
  @property
  def num_total_blocks(self) -> int:
    return self._num_total_blocks
