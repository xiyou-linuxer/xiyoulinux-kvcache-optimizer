from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Dict, Tuple
import time
import heapq
from miniflex.common.block import SequenceMeta

@dataclass
class RadixTreeNode:
  block_hashes : np.ndarray
  physical_block_ids : np.ndarray
  
  parent : Optional["RadixTreeNode"] = None
  children : Dict[int, "RadixTreeNode"] = field(default_factory=dict)
  
  _pin_count : int = 0
  _is_ready : bool = False
  hit_count : int = 0
  create_time : float = 0.0
  last_access_time : float = 0.0
  grace_time : float = 0.0
  
  def __post_init__(self):
    if self.block_hashes.ndim != 1 or self.physical_block_ids.ndim != 1:
      raise ValueError("block_hashes and physical_block_ids must be 1D arrays")
    if self.block_hashes.size != self.physical_block_ids.size:
      raise ValueError("block_hashes and physical_block_ids must have the same length")

  def __lt__(self, other: "RadixTreeNode") -> bool:
    return self.grace_time < other.grace_time

  def size(self) -> int:
    return self.block_hashes.size
  
  def head_hash(self) -> int:
    return int(self.block_hashes[0]) if self.size() > 0 else int(0)
  
  def is_root(self) -> bool:
    return self.parent is None
  
  def is_leaf(self) -> bool:
    return len(self.children) == 0 and not self.is_root() 
   
  def is_ready(self) -> bool:
    return self._is_ready
  
  def is_in_use(self) -> bool:
    return self._pin_count > 0 or not self._is_ready
  
  def is_evictable(self) -> bool:
    return (not self.is_in_use() and self.is_leaf() and not self.is_root())
  
  def num_children(self) -> int:
    return len(self.children)
  
  def split(self, prefix_length: int) -> "RadixTreeNode":
    if prefix_length <= 0 or prefix_length >= self.size():
      raise ValueError("prefix_length must be between 0 and the size of the node")
    if self.is_root():
      raise ValueError("root node cannot be split and root node must be empty")
    
    newparentnode = RadixTreeNode(
      block_hashes=self.block_hashes[:prefix_length],
      physical_block_ids=self.physical_block_ids[:prefix_length],
      parent=self.parent,
      _pin_count=0,
      _is_ready=self._is_ready,
      hit_count=self.hit_count,
      create_time=self.create_time,
      last_access_time=self.last_access_time,
      grace_time=self.grace_time,
    )
    self.block_hashes = self.block_hashes[prefix_length:]
    self.physical_block_ids = self.physical_block_ids[prefix_length:]
    self.parent.children[newparentnode.head_hash()] = newparentnode
    newparentnode.children[self.head_hash()] = self
    newparentnode.parent = self.parent
    self.parent = newparentnode
    return newparentnode
    
  def shrink(self, shrink_length: int) -> np.ndarray:
    if shrink_length < 0 or shrink_length >= self.size():
      raise ValueError("prefix_length must be between 0 and the size of the node")
    if self.is_in_use():
      raise ValueError("in-use node cannot be shrunk")
    if not self.is_leaf():
      raise ValueError("non-leaf node cannot be shrunk")
    if shrink_length == 0:
      return np.array([], dtype=np.int64)
    
    shrinked_physical_block_ids = self.physical_block_ids[-shrink_length:]
    self.block_hashes = self.block_hashes[:-shrink_length]
    self.physical_block_ids = self.physical_block_ids[:-shrink_length]
    return shrinked_physical_block_ids
    
  def merge_single_child(self):
    if self.num_children() != 1:
      raise ValueError("node must have exactly one child to merge")
    child = list(self.children.values())[0]
    self.block_hashes = np.concatenate([self.block_hashes, child.block_hashes])
    self.physical_block_ids = np.concatenate([self.physical_block_ids, child.physical_block_ids])
    self.children.pop(child.head_hash())
    self.grace_time = max(self.grace_time, child.grace_time)
    self.last_access_time = max(self.last_access_time, child.last_access_time)
    self.hit_count = max(self.hit_count, child.hit_count)
    for child_child in child.children.values():
      self.set_child(child_child.head_hash(), child_child)
      child_child.parent = self
    return
  
  def set_child(self,hash: int, child: "RadixTreeNode") -> None:
    if hash in self.children:
      raise ValueError("child with hash already exists")
    self.children[hash] = child
    child.parent = self
    
  def get_child(self,hash: int) -> Optional["RadixTreeNode"]:
    return self.children.get(hash)
  
@dataclass
class MatchResult:
  num_matched_blocks: int = 0
  num_ready_matched_blocks: int = 0
  last_ready_node: Optional["RadixTreeNode"] = None
  last_node: Optional["RadixTreeNode"] = None
  last_node_matched_length: int = 0
  physical_block_ids: np.ndarray = field(default_factory = lambda: np.array([], dtype=np.int64))
  
  def __post_init__(self):
    if self.physical_block_ids.ndim != 1:
      raise ValueError("physical_block_ids must be a 1D array")
    if self.physical_block_ids.dtype != np.int64:
      raise ValueError("physical_block_ids must be a 1D array of int64")
    
  def is_empty(self) -> bool:
    return self.num_matched_blocks == 0
  
  
class RadixTree:
  def __init__(self,tokens_per_block: int,max_num_blocks: int,eviction_policy: str = "lru",hit_add_counts: float = 0,protected_threshold: int = 2):
    self.tokens_per_block = tokens_per_block
    self.max_num_blocks = max_num_blocks
    self.eviction_policy = eviction_policy
    self.hit_add_counts = hit_add_counts
    self.protected_threshold = protected_threshold
    self.root = RadixTreeNode(
      block_hashes=np.array([], dtype=np.uint64),
      physical_block_ids=np.array([], dtype=np.int64),
      parent=None,
      _is_ready=True,
      _pin_count=0,
      grace_time = time.time()
    )
    self.leaf_nodes: Dict[int, RadixTreeNode] = {}
  
  def reset(self):
    self.root = RadixTreeNode(
      block_hashes=np.array([], dtype=np.uint64),
      physical_block_ids=np.array([], dtype=np.int64),
      parent=None,
      _is_ready=True,
      _pin_count=0,
      grace_time = time.time()
    )
    self.leaf_nodes.clear()
    
  def match_prefix(self, sequence: SequenceMeta) -> MatchResult:
    if not sequence.has_hashed():
      sequence.gen_hashes()
    current_node = self.root
    
    num_matched_blocks = 0
    num_ready_matched_blocks = 0
    ready_prefix_blocked = False
    last_ready_node = self.root
    last_node_matched_length = 0
    physical_block_ids = np.array([], dtype=np.int64)
    
    while num_matched_blocks < sequence.num_blocks:
      current_node.hit_count += 1
      current_node.last_access_time = time.time()
      if current_node.grace_time >= time.time():
        current_node.grace_time += self.hit_add_counts
      else:
        current_node.grace_time = time.time() + self.hit_add_counts
      next_node = current_node.children.get(sequence.get_hash(num_matched_blocks + current_node.size()))
      if next_node is not None:
        if current_node.is_ready() and not ready_prefix_blocked:
          last_ready_node = current_node
          num_ready_matched_blocks += current_node.size()
        elif not current_node.is_ready():
          ready_prefix_blocked = True
        num_matched_blocks += current_node.size()
        physical_block_ids = np.concatenate([physical_block_ids, current_node.physical_block_ids])
        current_node = next_node
      else:
        if not current_node.is_root():
          cmp_num = min(current_node.size(), sequence.num_blocks - num_matched_blocks);
          left = 0
          right = cmp_num
          while left < right:
            mid = (left + right) // 2
            if current_node.block_hashes[mid] == sequence.get_hash(num_matched_blocks + mid):
              left = mid + 1
            else:
              right = mid
          match_length = left
          physical_block_ids = np.concatenate([physical_block_ids, current_node.physical_block_ids[:match_length]])
        else:
          match_length = 0 
        if current_node.is_ready() and not ready_prefix_blocked:
          last_ready_node = current_node
          num_ready_matched_blocks += match_length
        elif not current_node.is_ready():
          ready_prefix_blocked = True
        num_matched_blocks += match_length
        last_node_matched_length = match_length
        break
    return MatchResult(
      num_matched_blocks=num_matched_blocks,
      num_ready_matched_blocks=num_ready_matched_blocks,
      last_ready_node=last_ready_node,
      last_node=current_node,
      last_node_matched_length=last_node_matched_length,
      physical_block_ids=physical_block_ids,
    )
    
  def num_matched(self, sequence: SequenceMeta) -> int:
    return self.match_prefix(sequence).num_matched_blocks
  
  def insert(self,sequence: SequenceMeta, physical_block_ids: np.ndarray,  match_result: Optional[MatchResult] = None, is_ready: bool = True) -> Optional["RadixTreeNode"]:
    if physical_block_ids.ndim != 1:
      raise ValueError("physical_block_ids must be a 1D array")
    if physical_block_ids.dtype != np.int64:
      raise ValueError("physical_block_ids must be a 1D array of int64")
    if not sequence.has_hashed():
      sequence.gen_hashes()
    if match_result is None:
      match_result = self.match_prefix(sequence)
    num_matched_blocks = match_result.num_matched_blocks
    if(sequence.num_blocks - num_matched_blocks != len(physical_block_ids)):
      raise ValueError("physical_block_ids must have the same length as the number of blocks in the sequence")
    last_node = match_result.last_node
    last_node_matched_length = match_result.last_node_matched_length
    if num_matched_blocks >= sequence.num_blocks:
      """In fact, > should never happen"""
      return None
    is_leaf = last_node.is_leaf() and not last_node.is_root()
    if is_leaf:
      self.leaf_nodes.pop(last_node.head_hash(),None)
    if last_node_matched_length < last_node.size():
      last_node.split(last_node_matched_length)
      if is_leaf:
        self.leaf_nodes[last_node.head_hash()] = last_node
      last_node = last_node.parent
    
      
    now = time.time()
    new_node = RadixTreeNode(
      block_hashes=sequence.block_hashes[num_matched_blocks:],
      physical_block_ids=physical_block_ids,
      parent=last_node,
      _is_ready=is_ready,
      create_time=now,
      last_access_time=now,
      grace_time=now + self.hit_add_counts,
    )
    last_node.set_child(new_node.head_hash(), new_node)
    self.leaf_nodes[new_node.head_hash()] = new_node
    return new_node
  
  
  def evict(self,num_evict_blocks: int) -> np.ndarray:
    if num_evict_blocks < 0:
      raise ValueError("num_evict_blocks must be greater than 0")
    if num_evict_blocks == 0:
      return np.array([], dtype=np.int64)
    evictable_nodes = []
    for node in self.leaf_nodes.values():
      if node.is_evictable():
        priority = self._get_evictable_priority(node)
        evictable_nodes.append((priority, node))
    heapq.heapify(evictable_nodes)
    evicted_physical_block_ids = np.array([], dtype=np.int64)
    while num_evict_blocks > 0:
      if len(evictable_nodes) == 0:
        break
      _, node = heapq.heappop(evictable_nodes)
      if node.size() <= num_evict_blocks:
        evicted_physical_block_ids = np.concatenate([evicted_physical_block_ids, node.physical_block_ids])
        num_evict_blocks -= node.size()
        self.leaf_nodes.pop(node.head_hash())
        node.parent.children.pop(node.head_hash())
        if node.parent.is_leaf():
          self.leaf_nodes[node.parent.head_hash()] = node.parent
        if node.parent.is_evictable():
          priority = self._get_evictable_priority(node.parent)
          heapq.heappush(evictable_nodes, (priority, node.parent))
        node.parent = None
        del node
      else:
        evicted_physical_block_ids = np.concatenate([evicted_physical_block_ids, node.shrink(num_evict_blocks)])
        num_evict_blocks = 0
    return evicted_physical_block_ids
    
        
  def _get_evictable_priority(self,node: RadixTreeNode):
    match self.eviction_policy:
      case "lru":
        return node.grace_time
      case "lfu":
        return (node.hit_count, node.last_access_time)
      case "slru":
        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.grace_time)
      case "fifo":
        return node.create_time
      case _:
        raise ValueError("invalid eviction policy")
      
  def is_empty(self) -> bool:
    return len(self.leaf_nodes) == 0

  def pin(self,node: RadixTreeNode) -> None:
    if node._pin_count < 0:
      raise ValueError("pin count cannot be negative")
    node._pin_count += 1
    
  def unpin(self,node: RadixTreeNode) -> None:
    if node._pin_count < 1:
      raise ValueError("pin count must be greater than 0")
    node._pin_count -= 1
    
    
  def set_ready(self,node: RadixTreeNode, ready: bool, ready_length: int = -1) -> None:
    if not ready:
      if node._is_ready:
        raise ValueError("node is already ready")
      node._is_ready = False
      return
    if node._is_ready and ready_length <= 0:
      raise ValueError("node is already ready")
    node._is_ready = ready
    if ready_length > 0:
      remaining_ready_length = ready_length - node.size()
      while remaining_ready_length > 0:
        if node.parent is None:
          raise ValueError("ready_length exceeds node ancestry")
        node = node.parent
        remaining_ready_length -= node.size()
        node._is_ready = True
      if remaining_ready_length != 0:
        raise ValueError("ready_length must align to node boundaries")
    
  def _node_num_blocks(self,node: RadixTreeNode) -> int:
    if node.is_leaf():
      return node.size()
    return sum(self._node_num_blocks(child) for child in node.children.values()) + node.size()
  
  def total_cached_blocks(self) -> int:
    return self._node_num_blocks(self.root)
  
  def _child_size(self,node: RadixTreeNode) -> int:
    if node.is_leaf():
      return 1
    return sum(self._child_size(child) for child in node.children.values()) + 1
  
  def total_node_size(self) -> int:
    return self._child_size(self.root) - 1
