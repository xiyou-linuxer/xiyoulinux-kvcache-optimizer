"""MiniFlex vLLM V1 connector 集成包。

`MiniFlexConnectorV1` 采用惰性导入：只有真正访问它时才 import connector
（进而 import vllm），这样在没有 vllm 的环境里仍可 import 本包及
`vllm_v1_adapter` 子模块（后者对 vllm 做了 try/except 降级）。
"""

__all__ = ["MiniFlexConnectorV1"]


def __getattr__(name):
  if name == "MiniFlexConnectorV1":
    from miniflex.integration.vllm.connector import MiniFlexConnectorV1
    return MiniFlexConnectorV1
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
