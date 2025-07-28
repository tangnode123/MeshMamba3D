# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fps_cuda',
    ext_modules=[
        CUDAExtension(
            name='fps_cuda',
            sources=[
                'fps_cuda.cpp',
                'fps_cuda_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2']
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)