from collections import OrderedDict
from typing import List, Tuple

from miniflex.common.transfer import TransferOp, TransferOpGraph, TransferType


class TransferScheduler:
  def __init__(self):
    self._transfer_graphs: OrderedDict[int, TransferOpGraph] = OrderedDict()

  def add_transfer_graph(self, transfer_graph: TransferOpGraph) -> None:
    if transfer_graph.graph_id in self._transfer_graphs:
      raise ValueError(f"transfer graph with graph_id {transfer_graph.graph_id} already exists")
    self._transfer_graphs[transfer_graph.graph_id] = transfer_graph

  def schedule(self, finished_ops: List[TransferOp]) -> Tuple[List[int], List[TransferOp]]:
    for finished_op in finished_ops:
      transfer_graph = self._transfer_graphs.get(finished_op.graph_id)
      if transfer_graph is None:
        continue
      transfer_graph.mark_completed(finished_op.op_id)

    next_ops = []
    for transfer_graph in self._transfer_graphs.values():
      for op in transfer_graph.take_ready_ops():
        if op.transfer_type == TransferType.VIRTUAL:
          transfer_graph.mark_completed(op.op_id)
        next_ops.append(op)

    completed_graphs = [
      graph_id
      for graph_id, transfer_graph in self._transfer_graphs.items()
      if transfer_graph.all_transfer_ops_completed()
    ]
    for graph_id in completed_graphs:
      self._transfer_graphs.pop(graph_id)
    return completed_graphs, next_ops

  def has_pending_graphs(self) -> bool:
    return len(self._transfer_graphs) > 0
