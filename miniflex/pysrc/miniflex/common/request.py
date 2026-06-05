from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy as np


class KVRequestType(Enum):
  GET = "get"
  PUT = "put"


def _check_1d_array(name: str, value: np.ndarray, dtype: type) -> None:
  if not isinstance(value, np.ndarray):
    raise ValueError(f"{name} must be np.ndarray, got {type(value).__name__}")
  if value.ndim != 1:
    raise ValueError(f"{name} must be a 1D array, got {value.ndim}D")
  if value.dtype != dtype:
    raise ValueError(f"{name} must have dtype {np.dtype(dtype)}, got {value.dtype}")


@dataclass(slots=True)
class KVRequest:
  request_type: KVRequestType
  token_ids: np.ndarray
  slot_mapping: Optional[np.ndarray] = None
  token_mask: Optional[np.ndarray] = None
  request_id: int = -1
  namespace: Optional[List[str]] = None

  def __post_init__(self) -> None:
    if not isinstance(self.request_type, KVRequestType):
      raise ValueError(f"request_type must be KVRequestType, got {type(self.request_type).__name__}")
    if not isinstance(self.request_id, int):
      raise ValueError(f"request_id must be int, got {type(self.request_id).__name__}")
    if self.request_id < -1:
      raise ValueError(f"request_id must be >= -1, got {self.request_id}")

    _check_1d_array("token_ids", self.token_ids, np.int64)

    if self.token_mask is not None:
      _check_1d_array("token_mask", self.token_mask, np.bool_)
      if self.token_mask.size != self.token_ids.size:
        raise ValueError(
          "token_mask size must equal token_ids size, "
          f"got {self.token_mask.size} and {self.token_ids.size}"
        )

    if self.slot_mapping is not None:
      _check_1d_array("slot_mapping", self.slot_mapping, np.int64)
      expected_size = int(self.token_mask.sum()) if self.token_mask is not None else self.token_ids.size
      if self.slot_mapping.size != expected_size:
        raise ValueError(
          "slot_mapping size must equal selected token count, "
          f"got {self.slot_mapping.size} and {expected_size}"
        )

    if self.namespace is not None:
      if not isinstance(self.namespace, list) or not all(isinstance(item, str) for item in self.namespace):
        raise ValueError("namespace must be None or list[str]")

  @property
  def task_id(self) -> int:
    return self.request_id


class KVResponseStatus(Enum):
  SUCCESS = "success"
  NOTFOUND = "not_found"
  UNREADY = "unready"
  TIMEOUT = "timeout"
  CANCELED = "canceled"
  FAILED = "failed"


@dataclass(slots=True)
class KVResponse:
  status: KVResponseStatus
  task_id: int
  return_mask: Optional[Union[np.ndarray, List[np.ndarray]]]

  def __post_init__(self) -> None:
    if not isinstance(self.status, KVResponseStatus):
      raise ValueError(f"status must be KVResponseStatus, got {type(self.status).__name__}")
    if not isinstance(self.task_id, int):
      raise ValueError(f"task_id must be int, got {type(self.task_id).__name__}")
    if self.return_mask is None:
      return
    if isinstance(self.return_mask, np.ndarray):
      if self.return_mask.ndim != 1 or self.return_mask.dtype != np.bool_:
        raise ValueError("return_mask must be a 1D bool ndarray")
      return
    if isinstance(self.return_mask, list):
      return
    raise ValueError(f"return_mask must be None, np.ndarray, or list[np.ndarray], got {type(self.return_mask).__name__}")

  def get_mask(self, idx: int = 0) -> np.ndarray:
    if self.return_mask is None:
      raise ValueError("return_mask is None")
    if isinstance(self.return_mask, np.ndarray):
      if idx != 0:
        raise IndexError("single return_mask only supports idx=0")
      return self.return_mask
    return self.return_mask[idx]
