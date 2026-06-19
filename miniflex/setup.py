from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    ext_modules=[
        CUDAExtension(
            name="miniflex._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/ssd_io_uring.cpp",
                "csrc/transfer.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"],
            },
            libraries=["uring"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
