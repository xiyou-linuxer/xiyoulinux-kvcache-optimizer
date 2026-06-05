from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from miniflex.common.config import CacheConfig, ModelConfig
from miniflex.common.storage import KVCacheLayoutType

if TYPE_CHECKING:
  from vllm.config import VllmConfig


def _parse_bool(value: Any) -> bool:
  if isinstance(value, bool):
    return value
  if isinstance(value, int):
    return bool(value)
  if isinstance(value, str):
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
      return True
    if normalized in ("0", "false", "no", "off"):
      return False
  raise ValueError(f"cannot parse bool from {value!r}")


def _parse_layout_type(value: Any) -> KVCacheLayoutType:
  if isinstance(value, KVCacheLayoutType):
    return value
  if isinstance(value, str):
    normalized = value.strip().upper()
    if normalized in KVCacheLayoutType.__members__:
      return KVCacheLayoutType[normalized]
    for layout_type in KVCacheLayoutType:
      if normalized == layout_type.value.upper():
        return layout_type
  raise ValueError(f"unsupported KV cache layout type: {value!r}")


def _parse_cache_dir(value: Any) -> str | list[str] | None:
  if value is None:
    return None
  if isinstance(value, str):
    stripped = value.strip()
    if stripped == "":
      return None
    if stripped.startswith("["):
      loaded = json.loads(stripped)
      if not isinstance(loaded, list):
        raise ValueError("cache dir JSON must decode to list[str]")
      return loaded
    if "," in stripped:
      return [item.strip() for item in stripped.split(",") if item.strip()]
    return stripped
  if isinstance(value, list):
    return value
  raise ValueError(f"unsupported cache dir value: {value!r}")


def _load_json_config(path: str) -> dict[str, Any]:
  with open(path, "r", encoding="utf-8") as config_file:
    data = json.load(config_file)
  if not isinstance(data, dict):
    raise ValueError("MiniFlex config file must contain a JSON object")
  return data


def _apply_cache_overrides(cache_config: CacheConfig, values: dict[str, Any]) -> None:
  int_fields = ("tokens_per_block", "num_cpu_blocks", "num_ssd_blocks", "protected_threshold")
  float_fields = ("ssd_max_file_size_gb", "evict_ratio", "evict_start_threshold", "hit_add_counts")
  bool_fields = ("enable_cpu", "enable_ssd", "use_direct_io")
  str_fields = ("ssd_file_prefix", "eviction_policy")

  for key in int_fields:
    if key in values and values[key] is not None:
      setattr(cache_config, key, int(values[key]))
  for key in float_fields:
    if key in values and values[key] is not None:
      setattr(cache_config, key, float(values[key]))
  for key in bool_fields:
    if key in values and values[key] is not None:
      setattr(cache_config, key, _parse_bool(values[key]))
  for key in str_fields:
    if key in values and values[key] is not None:
      setattr(cache_config, key, str(values[key]))

  if "ssd_cache_dir" in values:
    cache_config.ssd_cache_dir = _parse_cache_dir(values["ssd_cache_dir"])
  if "cpu_layout_type" in values and values["cpu_layout_type"] is not None:
    cache_config.cpu_layout_type = _parse_layout_type(values["cpu_layout_type"])
  if "ssd_layout_type" in values and values["ssd_layout_type"] is not None:
    cache_config.ssd_layout_type = _parse_layout_type(values["ssd_layout_type"])


def _env_overrides() -> dict[str, Any]:
  mapping = {
    "MINIFLEX_NUM_CPU_BLOCKS": "num_cpu_blocks",
    "MINIFLEX_NUM_SSD_BLOCKS": "num_ssd_blocks",
    "MINIFLEX_SSD_CACHE_DIR": "ssd_cache_dir",
    "MINIFLEX_SSD_FILE_PREFIX": "ssd_file_prefix",
    "MINIFLEX_SSD_MAX_FILE_SIZE_GB": "ssd_max_file_size_gb",
    "MINIFLEX_ENABLE_CPU": "enable_cpu",
    "MINIFLEX_ENABLE_SSD": "enable_ssd",
    "MINIFLEX_USE_DIRECT_IO": "use_direct_io",
    "MINIFLEX_EVICTION_POLICY": "eviction_policy",
    "MINIFLEX_EVICT_RATIO": "evict_ratio",
    "MINIFLEX_EVICT_START_THRESHOLD": "evict_start_threshold",
    "MINIFLEX_HIT_ADD_COUNTS": "hit_add_counts",
    "MINIFLEX_PROTECTED_THRESHOLD": "protected_threshold",
    "MINIFLEX_CPU_LAYOUT_TYPE": "cpu_layout_type",
    "MINIFLEX_SSD_LAYOUT_TYPE": "ssd_layout_type",
  }
  result: dict[str, Any] = {}
  for env_name, field_name in mapping.items():
    if env_name in os.environ:
      result[field_name] = os.environ[env_name]
  return result


def _get_total_num_kv_heads(vllm_config: "VllmConfig") -> int:
  model_config = vllm_config.model_config
  if hasattr(model_config, "get_total_num_kv_heads"):
    return int(model_config.get_total_num_kv_heads())
  if hasattr(model_config, "get_num_kv_heads"):
    return int(model_config.get_num_kv_heads(vllm_config.parallel_config))
  hf_config = getattr(model_config, "hf_config", None)
  if hf_config is not None and hasattr(hf_config, "num_key_value_heads"):
    return int(hf_config.num_key_value_heads)
  raise ValueError("cannot infer num_kv_heads from vLLM config")


@dataclass
class MiniFlexConfig:
  enable_miniflex: bool = True
  gpu_register_port: str = ""
  enable_batch: bool = False
  sync_get: bool = False
  cache_config: CacheConfig = field(default_factory=lambda: CacheConfig(tokens_per_block=1))
  model_config: ModelConfig = field(default_factory=ModelConfig)

  def __post_init__(self) -> None:
    if self.gpu_register_port == "":
      self.gpu_register_port = os.getenv("MINIFLEX_GPU_REGISTER_PORT", "ipc:///tmp/miniflex_gpu_register.sock")
    if not isinstance(self.gpu_register_port, str) or self.gpu_register_port == "":
      raise ValueError("gpu_register_port must be a non-empty string")
    if not isinstance(self.cache_config, CacheConfig):
      raise ValueError(f"cache_config must be CacheConfig, got {type(self.cache_config).__name__}")
    if not isinstance(self.model_config, ModelConfig):
      raise ValueError(f"model_config must be ModelConfig, got {type(self.model_config).__name__}")

  @classmethod
  def from_env(cls) -> "MiniFlexConfig":
    enable_miniflex = _parse_bool(os.getenv("ENABLE_MINIFLEX", "1"))
    gpu_register_port = os.getenv("MINIFLEX_GPU_REGISTER_PORT", "")
    enable_batch = _parse_bool(os.getenv("MINIFLEX_ENABLE_BATCH", "0"))
    sync_get = _parse_bool(os.getenv("MINIFLEX_SYNC_GET", "0"))

    model_config = ModelConfig()
    cache_config = CacheConfig(tokens_per_block=1)

    config_path = os.getenv("MINIFLEX_CONFIG_PATH", None)
    if config_path:
      file_config = _load_json_config(config_path)
      _apply_cache_overrides(cache_config, file_config.get("cache_config", {}))
      if "enable_miniflex" in file_config:
        enable_miniflex = _parse_bool(file_config["enable_miniflex"])
      if "gpu_register_port" in file_config:
        gpu_register_port = str(file_config["gpu_register_port"])
      if "enable_batch" in file_config:
        enable_batch = _parse_bool(file_config["enable_batch"])
      if "sync_get" in file_config:
        sync_get = _parse_bool(file_config["sync_get"])

    env_config = _env_overrides()
    _apply_cache_overrides(cache_config, env_config)

    return cls(
      enable_miniflex=enable_miniflex,
      gpu_register_port=gpu_register_port,
      enable_batch=enable_batch,
      sync_get=sync_get,
      cache_config=cache_config,
      model_config=model_config,
    )

  def post_init_from_vllm_config(self, vllm_config: "VllmConfig") -> None:
    self.cache_config.tokens_per_block = int(vllm_config.cache_config.block_size)

    self.model_config.num_layers = int(
      vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    )
    self.model_config.head_size = int(vllm_config.model_config.get_head_size())
    self.model_config.dtype = vllm_config.model_config.dtype
    self.model_config.use_mla = bool(getattr(vllm_config.model_config, "is_deepseek_mla", False))
    self.model_config.tp_size = int(vllm_config.parallel_config.tensor_parallel_size)
    self.model_config.dp_size = int(vllm_config.parallel_config.data_parallel_size)
    self.model_config.num_kv_heads = 1 if self.model_config.use_mla else _get_total_num_kv_heads(vllm_config)

    overrides = _env_overrides()
    _apply_cache_overrides(self.cache_config, overrides)
