import numpy as np
from typing import Optional, List
from miniflex.cache.radix_tree import RadixTree, RadixTreeNode, MatchResult
from miniflex.cache.mempool import Mempool
from miniflex.common.transfer import DeviceType
from miniflex.common.block import SequenceMeta

class CacheEngine:
  def __init__(self, 
               device_type: DeviceType,
               num_total_blocks: int,
               tokens_per_block: int,
               eviction_policy: str = "lru",
               hit_add_counts: float = 0,
               evict_ratio: float = 0.1,
               evict_start_threshold: float = 1.0,
               protected_threshold: int = 2,
              ):
    if device_type not in [DeviceType.CPU, DeviceType.SSD]:
      raise ValueError(f"invalid device type, got {device_type}")
    if num_total_blocks <= 0:
      raise ValueError(f"num_total_blocks must be greater than 0, got {num_total_blocks}")
    if tokens_per_block <= 0 or (tokens_per_block & (tokens_per_block - 1)) != 0:   
      raise ValueError(f"tokens_per_block must be greater than 0 and a power of 2, got {tokens_per_block}")
    if eviction_policy not in ["lru", "lfu", "slru", "fifo"]:
      raise ValueError(f"invalid eviction policy, got {eviction_policy}")
    if evict_ratio < 0 or evict_ratio >= 1:
      raise ValueError(f"evict_ratio must be between 0 and 1, got {evict_ratio}")
    if evict_start_threshold <= 0 or evict_start_threshold > 1:
      raise ValueError(f"evict_start_threshold must be between 0 and 1, got {evict_start_threshold}")
    if protected_threshold <= 0:
      raise ValueError(f"protected_threshold must be greater than 0, got {protected_threshold}")
    self.device_type = device_type
    self.num_total_blocks = num_total_blocks
    self.tokens_per_block = tokens_per_block
    self.eviction_policy = eviction_policy
    self.evict_ratio = evict_ratio
    self.evict_start_threshold = evict_start_threshold
    self.protected_threshold = protected_threshold
    self.radix_tree = RadixTree(tokens_per_block, num_total_blocks, eviction_policy, hit_add_counts,protected_threshold)
    self.mempool = Mempool(num_total_blocks)
    
  def reset(self):
    self.radix_tree.reset()
    self.mempool.reset()
  
  def match(self, sequence: SequenceMeta) -> MatchResult:
    return self.radix_tree.match_prefix(sequence)
  
  def insert(self,sequence: SequenceMeta, physical_block_ids: np.ndarray, is_ready: bool = True, match_result: Optional[MatchResult] = None) -> Optional[RadixTreeNode]:
    return self.radix_tree.insert(sequence, physical_block_ids, match_result, is_ready)
  
  def take(self,num_required_blocks: int, protected_node: Optional[RadixTreeNode] = None, strict: bool = True) -> np.ndarray:
    if num_required_blocks < 0:
      raise ValueError(f"num_required_blocks must be greater than 0, got {num_required_blocks}")
    utilization = self.mempool.num_used_blocks / self.num_total_blocks
    should_evict = utilization >= self.evict_start_threshold or self.mempool.num_free_blocks < num_required_blocks
    if should_evict:
      if protected_node is not None:
        self.pin(protected_node)
      target_blocks = int(self.mempool.num_total_blocks * (1.0 - self.evict_start_threshold))
      num_evict_blocks = max(0,target_blocks - self.mempool.num_free_blocks)
      num_needed_evict_blocks = max(
        num_evict_blocks,
        num_required_blocks - self.mempool.num_free_blocks, 
        int(self.evict_ratio * self.num_total_blocks),
      )
      if num_needed_evict_blocks > 0:
        evicted_blocks = self.radix_tree.evict(num_needed_evict_blocks)
        self.mempool.recycle(evicted_blocks)
      if protected_node is not None:
        self.unpin(protected_node)
    if strict and num_required_blocks > self.mempool.num_free_blocks:
      raise RuntimeError(f"failed to take {num_required_blocks} blocks")
    num_required_blocks = min(num_required_blocks, self.mempool.num_free_blocks)
    physical_block_ids = self.mempool.allocate(num_required_blocks)

    return physical_block_ids
    
  
  def pin(self, node: RadixTreeNode) -> None:
    self.radix_tree.pin(node)
    
  def unpin(self, node: RadixTreeNode) -> None:
    self.radix_tree.unpin(node)
    
  def set_ready(self, node: RadixTreeNode, ready: bool = True, ready_length: int = -1) -> None:
    self.radix_tree.set_ready(node, ready, ready_length)
    
  def recycle(self, physical_block_ids: np.ndarray) -> None:
    self.mempool.recycle(physical_block_ids)
