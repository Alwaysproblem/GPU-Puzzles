conda create -n numba python=3 cuda-nvcc cuda-nvrtc "cuda-version==12.2" nvidia/label/cuda-12.2.2::cuda-toolkit cuda-python jupyter numba numpy -y -c conda-forge -c nvidia -c numba
