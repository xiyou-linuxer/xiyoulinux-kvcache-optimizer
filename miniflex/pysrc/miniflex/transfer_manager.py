from abc import ABC, abstractmethod
import os
import queue
import selectors
import tempfile
from typing import List, Optional

import zmq
import multiprocessing as mp

from miniflex.common.config import CacheConfig, ModelConfig
from miniflex.common.storage import StorageHandle
from miniflex.common.transfer import CompletedOp, DeviceType, TransferOpGraph
from miniflex.server.request import RegisterGPUBlocksRequest
from miniflex.server.utils import get_zmq_socket, normalize_zmq_endpoint
from miniflex.storage.storage_engine import StorageEngine
from miniflex.transfer.transfer_engine import TransferEngine


class TransferManager:
  def __init__(
      self,
      model_config: ModelConfig,
      cache_config: CacheConfig,
      gpu_register_port: str,
  ):
    self.model_config = model_config
    self.cache_config = cache_config
    self.gpu_register_port = gpu_register_port
    self.gpu_register_endpoint = normalize_zmq_endpoint(gpu_register_port)

    self.zmq_context = zmq.Context(2)
    self.recv_client_socket = get_zmq_socket(
      self.zmq_context,
      zmq.SocketType.PULL,
      self.gpu_register_endpoint,
      bind=True,
    )

    self._storage_engine = StorageEngine(
      model_config=self.model_config,
      cache_config=self.cache_config,
    )
    self._transfer_engine: Optional[TransferEngine] = None
    self._gpu_handle: Optional[StorageHandle] = None
    self._cpu_handle: Optional[StorageHandle] = self._storage_engine.get_storage_handle(
      DeviceType.CPU,
      0,
    )
    if self._cpu_handle is None:
      raise RuntimeError("CPU storage handle is required")

    self._ssd_handle: Optional[StorageHandle] = (
      self._storage_engine.get_storage_handle(DeviceType.SSD, 0)
      if self.cache_config.enable_ssd else None
    )
    if self.cache_config.enable_ssd and self._ssd_handle is None:
      raise RuntimeError("SSD storage handle is required when enable_ssd=True")

  def _register_gpu_storage_handle_via_zmq(self) -> None:
    req = self.recv_client_socket.recv_pyobj()
    if not isinstance(req, RegisterGPUBlocksRequest):
      raise ValueError(f"unexpected request type: {type(req).__name__}")
    self._handle_register_gpu_storage_request(req)

  def _handle_register_gpu_storage_request(self, req: RegisterGPUBlocksRequest) -> None:
    req.validate_single_gpu()
    if self._gpu_handle is not None:
      raise ValueError("GPU storage handle already registered")

    self._storage_engine.register_gpu_blocks(
      req.handles,
      req.gpu_layout,
      device_id=req.device_id,
      dtype=self.model_config.dtype,
    )
    self._gpu_handle = self._storage_engine.get_storage_handle(
      DeviceType.GPU,
      req.device_id,
    )
    if self._gpu_handle is None:
      raise RuntimeError("GPU storage registration did not create a storage handle")

  def initialize_transfer_engine(self) -> None:
    if self._transfer_engine is not None:
      return
    if self._gpu_handle is None:
      self._register_gpu_storage_handle_via_zmq()
    if self._gpu_handle is None:
      raise RuntimeError("GPU storage handle is not registered")

    self._transfer_engine = TransferEngine(
      model_config=self.model_config,
      cache_config=self.cache_config,
      gpu_handle=self._gpu_handle,
      cpu_handle=self._cpu_handle,
      ssd_handle=self._ssd_handle,
    )

  def _require_transfer_engine(self) -> TransferEngine:
    if self._transfer_engine is None:
      raise RuntimeError("TransferEngine is not initialized; call start() first")
    return self._transfer_engine

  def start(self) -> None:
    self._require_transfer_engine().start()

  def submit(self, transfer_graph: TransferOpGraph) -> None:
    self._require_transfer_engine().submit_transfer_graph(transfer_graph)

  def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
    self._require_transfer_engine().submit_transfer_graph(transfer_graphs)

  def wait(self, timeout: Optional[float] = None) -> List[CompletedOp]:
    return self._require_transfer_engine().get_completed_graphs_and_ops(timeout)

  def shutdown(self) -> None:
    if self._transfer_engine is not None:
      self._transfer_engine.shutdown()
      self._transfer_engine = None
    try:
      self.recv_client_socket.close()
    finally:
      self.zmq_context.term()


class TransferManagerHandleBase(ABC):
  @abstractmethod
  def start(self) -> None:
    pass

  @abstractmethod
  def is_ready(self) -> bool:
    pass

  @abstractmethod
  def submit(self, transfer_graph: TransferOpGraph) -> None:
    pass

  @abstractmethod
  def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
    pass

  @abstractmethod
  def wait(self, timeout: Optional[float] = None) -> List[CompletedOp]:
    pass

  @abstractmethod
  def shutdown(self) -> None:
    pass
  
class TransferManagerHandleIntraProcessHandle(TransferManagerHandleBase):
  def __init__(self,
               model_config: ModelConfig,
               cache_config: CacheConfig,
               gpu_register_port: str):
    self._manager = TransferManager(model_config, cache_config, gpu_register_port)
    self._is_ready = False
    
  def start(self) -> None:
    self._manager.initialize_transfer_engine()
    self._manager.start()
    self._is_ready = True

  def is_ready(self) -> bool:
    return self._is_ready

  def submit(self, transfer_graph: TransferOpGraph) -> None:
    self._manager.submit(transfer_graph)

  def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
    self._manager.submit_batch(transfer_graphs)
  
  def wait(self, timeout: Optional[float] = None) -> List[CompletedOp]:
    return self._manager.wait(timeout)

  def shutdown(self) -> None:
    self._manager.shutdown()
    self._is_ready = False
  
class TransferManagerHandleInterProcessHandle(TransferManagerHandleBase):
  def __init__(self,
               model_config: ModelConfig,
               cache_config: CacheConfig,
               gpu_register_port: str):
    self._mp_ctx = mp.get_context("spawn")
    self.model_config = model_config
    self.cache_config = cache_config
    self.gpu_register_port = gpu_register_port
    self.command_parent_conn, self.command_child_conn = self._mp_ctx.Pipe()
    self.result_parent_conn, self.result_child_conn = self._mp_ctx.Pipe()

    self.process: Optional[mp.Process] = None
    self.start_event = self._mp_ctx.Event()
    self.ready_event = self._mp_ctx.Event()

  @staticmethod
  def _worker_process(model_config: ModelConfig,
                      cache_config: CacheConfig,
                      gpu_register_port: str,
                      command_conn,
                      result_conn,
                      start_event,
                      ready_event) -> None:
    os.environ["MPI4PY_RC_INITIALIZE"] = "false"
    transfer_manager: Optional[TransferManager] = None
    sel: Optional[selectors.BaseSelector] = None
    try:
      transfer_manager = TransferManager(model_config, cache_config, gpu_register_port)
      start_event.set()
      transfer_manager.initialize_transfer_engine()
      transfer_manager.start()
      ready_event.set()

      transfer_engine = transfer_manager._require_transfer_engine()
      sel = selectors.DefaultSelector()
      sel.register(
        command_conn.fileno(),
        selectors.EVENT_READ,
        data=lambda: TransferManagerHandleInterProcessHandle._handle_command_queue(
          command_conn,
          transfer_manager,
        ),
      )
      sel.register(
        transfer_engine._completed_queue._reader,
        selectors.EVENT_READ,
        data=lambda: TransferManagerHandleInterProcessHandle._handle_completed_queue(
          result_conn,
          transfer_manager,
        ),
      )

      while True:
        events = sel.select(timeout=None)
        should_shutdown = False
        for key, _mask in events:
          callback = key.data
          should_shutdown = bool(callback()) or should_shutdown
        if should_shutdown:
          break
    except Exception as e:
      print(f"Error in worker process: {type(e).__name__}: {e}")
    finally:
      if sel is not None:
        try:
          sel.close()
        except Exception as e:
          print(f"Error closing selector: {type(e).__name__}: {e}")
      if transfer_manager is not None:
        try:
          transfer_manager.shutdown()
        except Exception as e:
          print(f"Error shutting down transfer manager: {type(e).__name__}: {e}")
      command_conn.close()
      result_conn.close()

  @staticmethod
  def _handle_command_queue(command_conn, transfer_manager: TransferManager) -> bool:
    request = command_conn.recv()
    if not isinstance(request, dict):
      raise ValueError(f"invalid command request type: {type(request).__name__}")
    request_type = request.get("type")
    match request_type:
      case "submit":
        transfer_graph = request.get("transfer_graph")
        if not isinstance(transfer_graph, TransferOpGraph):
          raise ValueError(f"invalid transfer graph type: {type(transfer_graph).__name__}")
        transfer_manager.submit(transfer_graph)
        return False
      case "submit_batch":
        transfer_graphs = request.get("transfer_graphs")
        if not isinstance(transfer_graphs, list) or not all(isinstance(graph, TransferOpGraph) for graph in transfer_graphs):
          raise ValueError(f"invalid transfer graphs type: {type(transfer_graphs).__name__}")
        for g in transfer_graphs:
          print(f"[DBG-TFMGR] recv graph={g.graph_id} ops={g.num_ops}", flush=True)
        transfer_manager.submit_batch(transfer_graphs)
        return False
      case "shutdown":
        return True
      case _:
        raise ValueError(f"unknown request type: {request_type}")

  @staticmethod
  def _handle_completed_queue(result_conn, transfer_manager: TransferManager) -> bool:
    completed_ops = transfer_manager.wait(timeout=0)
    if completed_ops:
      result_conn.send(completed_ops)
    return False

  def _start_process(self) -> None:
    if self.process is not None and self.process.is_alive():
      return

    self.process = self._mp_ctx.Process(
      target=TransferManagerHandleInterProcessHandle._worker_process,
      args=(
        self.model_config,
        self.cache_config,
        self.gpu_register_port,
        self.command_child_conn,
        self.result_child_conn,
        self.start_event,
        self.ready_event,
      ),
      daemon=False,
    )
    self.process.start()

  def start(self) -> None:
    old_mpi_initialize = os.environ.get("MPI4PY_RC_INITIALIZE")
    os.environ["MPI4PY_RC_INITIALIZE"] = "false"
    try:
      self._start_process()
      while not self.start_event.wait(timeout=0.1):
        if self.process is None:
          raise RuntimeError("TransferManager process is not started")
        if not self.process.is_alive():
          raise RuntimeError(
            f"TransferManager process exited before startup completed, exitcode={self.process.exitcode}"
          )
    finally:
      if old_mpi_initialize is None:
        os.environ.pop("MPI4PY_RC_INITIALIZE", None)
      else:
        os.environ["MPI4PY_RC_INITIALIZE"] = old_mpi_initialize

  def is_ready(self) -> bool:
    return self.ready_event.is_set()

  def _require_ready_process(self) -> mp.Process:
    if self.process is None:
      raise RuntimeError("TransferManager process is not started")
    if not self.process.is_alive():
      raise RuntimeError(f"TransferManager process is not alive, exitcode={self.process.exitcode}")
    if not self.ready_event.is_set():
      raise RuntimeError("TransferManager process is not ready")
    return self.process

  def submit(self, transfer_graph: TransferOpGraph) -> None:
    self._require_ready_process()
    if not isinstance(transfer_graph, TransferOpGraph):
      raise ValueError(f"invalid transfer graph type: {type(transfer_graph).__name__}")
    self.command_parent_conn.send({
      "type": "submit",
      "transfer_graph": transfer_graph,
    })

  def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
    self._require_ready_process()
    if not isinstance(transfer_graphs, list) or not all(isinstance(graph, TransferOpGraph) for graph in transfer_graphs):
      raise ValueError(f"invalid transfer graphs type: {type(transfer_graphs).__name__}")
    self.command_parent_conn.send({
      "type": "submit_batch",
      "transfer_graphs": transfer_graphs,
    })

  def wait(self, timeout: Optional[float] = None) -> List[CompletedOp]:
    finished_ops: List[CompletedOp] = []
    try:
      if self.result_parent_conn.poll(timeout=timeout):
        finished_ops.extend(self.result_parent_conn.recv())
        while self.result_parent_conn.poll():
          finished_ops.extend(self.result_parent_conn.recv())
    except EOFError:
      pass
    return finished_ops

  def shutdown(self) -> None:
    if self.process is not None and self.process.is_alive():
      try:
        self.command_parent_conn.send({"type": "shutdown"})
      except (BrokenPipeError, EOFError, OSError):
        pass
      self.process.join(timeout=5)
      if self.process.is_alive():
        self.process.terminate()
        self.process.join(timeout=5)
      if self.process.is_alive():
        self.process.kill()
        self.process.join()
    self.process = None
    try:
      self.command_parent_conn.close()
      self.result_parent_conn.close()
    except OSError:
      pass

  def __del__(self):
    try:
      self.shutdown()
    except Exception:
      pass

class TransferManagerHandle:
  def __init__(
      self,
      model_config: ModelConfig,
      cache_config: CacheConfig,
      gpu_register_port: Optional[str] = None,
      mode: str = "process",
  ):
    if gpu_register_port is None or gpu_register_port == "":
      gpu_register_port = self._make_default_gpu_register_port()
    if not isinstance(gpu_register_port, str) or gpu_register_port == "":
      raise ValueError("gpu_register_port must be a non-empty string")
    if not isinstance(mode, str):
      raise ValueError(f"mode must be a string, got {type(mode).__name__}")

    self.gpu_register_port = gpu_register_port
    self.gpu_register_endpoint = normalize_zmq_endpoint(gpu_register_port)

    mode_name = mode.lower()
    self.mode = mode_name
    if mode_name in ("process", "inter", "inter_process"):
      self._handle: TransferManagerHandleBase = TransferManagerHandleInterProcessHandle(
        model_config,
        cache_config,
        gpu_register_port,
      )
    elif mode_name in ("thread", "intra", "intra_process"):
      self._handle = TransferManagerHandleIntraProcessHandle(
        model_config,
        cache_config,
        gpu_register_port,
      )
    else:
      raise ValueError(
        f"invalid TransferManager mode: {mode}; expected process or thread"
      )

  @staticmethod
  def _make_default_gpu_register_port() -> str:
    fd, path = tempfile.mkstemp(prefix="miniflex_gpu_register_", suffix=".sock")
    os.close(fd)
    return path

  def start(self) -> None:
    self._handle.start()

  def is_ready(self) -> bool:
    return self._handle.is_ready()

  def submit(self, transfer_graph: TransferOpGraph) -> None:
    self._handle.submit(transfer_graph)

  def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
    self._handle.submit_batch(transfer_graphs)

  def wait(self, timeout: Optional[float] = None) -> List[CompletedOp]:
    return self._handle.wait(timeout)

  def shutdown(self) -> None:
    self._handle.shutdown()

