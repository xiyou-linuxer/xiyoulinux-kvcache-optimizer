import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME as TORCH_CUDA_HOME,
)


def _cuda_roots():
  roots = (
      os.environ.get("CUDA_HOME"),
      TORCH_CUDA_HOME,
      "/usr/local/cuda",
  )
  seen = set()
  for root in roots:
    if not root:
      continue
    path = Path(root)
    key = str(path.resolve()) if path.exists() else str(path)
    if key not in seen:
      seen.add(key)
      yield path


def _find_cufile():
  """Return the cuFile include and library directories, if present."""
  for cuda_root in _cuda_roots():
    include_dir = cuda_root / "include"
    if not (include_dir / "cufile.h").is_file():
      continue
    for lib_dir in (
        cuda_root / "lib64",
        cuda_root / "targets" / "x86_64-linux" / "lib",
        cuda_root / "lib",
    ):
      if (lib_dir / "libcufile.so").is_file():
        return include_dir, lib_dir
  return None


def _gds_flags():
  """Return cuFile build flags controlled by MINIFLEX_CUFILE.

  "auto" (default) enables cuFile when the toolkit provides both cufile.h and
  libcufile.so; "0" always builds the unavailable backend stub; "1" requires
  cuFile and fails early with a useful message. The GDS worker never performs
  a CPU two-hop fallback itself.
  """
  mode = os.environ.get("MINIFLEX_CUFILE", "auto").strip().lower()
  if mode not in {"auto", "0", "1"}:
    raise RuntimeError(
        "MINIFLEX_CUFILE must be one of 'auto', '0', or '1', "
        f"got {mode!r}"
    )

  if mode == "0":
    print("[miniflex] cuFile backend disabled (MINIFLEX_CUFILE=0)")
    return [], [], [], []

  cufile = _find_cufile()
  if cufile is not None:
    include_dir, lib_dir = cufile
    print(f"[miniflex] building cuFile backend from {lib_dir}")
    return ["MINIFLEX_WITH_CUFILE"], ["cufile"], [str(lib_dir)], [str(include_dir)]

  if mode == "1":
    raise RuntimeError(
        "MINIFLEX_CUFILE=1 requires cufile.h and libcufile.so under CUDA_HOME "
        "(or the CUDA toolkit detected by PyTorch)."
    )

  print("[miniflex] cuFile was not found; building unavailable GDS backend stub")
  return [], [], [], []


_gds_defines, _gds_libs, _gds_lib_dirs, _gds_include_dirs = _gds_flags()

setup(
    ext_modules=[
        CUDAExtension(
            name="miniflex._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/ssd_io_uring.cpp",
                "csrc/gds_io.cpp",
                "csrc/transfer.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"],
            },
            define_macros=[(define, None) for define in _gds_defines],
            include_dirs=_gds_include_dirs,
            libraries=["uring"] + _gds_libs,
            library_dirs=_gds_lib_dirs,
            runtime_library_dirs=(
                _gds_lib_dirs if sys.platform.startswith("linux") else []
            ),
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
