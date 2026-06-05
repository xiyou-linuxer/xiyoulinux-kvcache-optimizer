from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    ext_modules=[
        CppExtension(
            name="miniflex._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/ssd_io_uring.cpp",
            ],
            extra_compile_args=["-O3", "-std=c++17"],
            libraries=["uring"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
