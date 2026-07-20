#python3 setup.py install
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import sys
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)

# NOTE: default to the active (conda) env, NOT /usr/local/cuda. The system
# /usr/local/cuda is 12.5, whose headers emit __cudaLaunchKernel and fail to
# link against the conda cuda 11.8 runtime (undefined symbol at import time).
cuda_home = os.environ.get('CUDA_HOME') or sys.prefix
include_dirs = [os.path.join(cuda_home, 'include')]

setup(
    name='pointops',
    author='Hengshuang Zhao',
    ext_modules=[
        CUDAExtension('pointops_cuda', [
            'src/pointops_api.cpp',
            'src/knnquery/knnquery_cuda.cpp',
            'src/knnquery/knnquery_cuda_kernel.cu',
            'src/sampling/sampling_cuda.cpp',
            'src/sampling/sampling_cuda_kernel.cu',
            'src/grouping/grouping_cuda.cpp',
            'src/grouping/grouping_cuda_kernel.cu',
            'src/interpolation/interpolation_cuda.cpp',
            'src/interpolation/interpolation_cuda_kernel.cu',
            'src/subtraction/subtraction_cuda.cpp',
            'src/subtraction/subtraction_cuda_kernel.cu',
            'src/aggregation/aggregation_cuda.cpp',
            'src/aggregation/aggregation_cuda_kernel.cu',
            ],
        include_dirs=include_dirs,
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
