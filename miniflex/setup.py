from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    ext_modules=[
        CppExtension(
            name="miniflex._C",
            sources=[
                "csrc/cache_utils.cc",
                "csrc/bindings.cpp",
            ],
            extra_compile_args=["-O3", "-std=c++17"],
            libraries=["xxhash"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)