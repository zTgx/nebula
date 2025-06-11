## nebula

### pre-required
* CUDA
* Boost
* CMake
* C++17+

### Starter
```bash
mkdir build && cd build
cmake .. && make

./nebula
```

```bash
curl -X POST http://localhost:12345/compute -H "Content-Type: application/json" -d '{
    "task": "matrix_multiply",
    "data": {
        "matrix_a": [[1.0, 2.0], [3.0, 4.0]],
        "matrix_b": [[5.0, 6.0], [7.0, 8.0]]
    }
}'

curl -X POST http://localhost:8080/compute -H "Content-Type: application/json" -d '{
    "task": "matrix_vector_multiply",
    "data": {
        "matrix_a": [[1.0, 2.0], [3.0, 4.0]],
        "vector_x": [5.0, 6.0]
    }
}'

curl -X POST http://localhost:8080/compute -H "Content-Type: application/json" -d '{
    "task": "vector_dot",
    "data": {
        "vector_x": [1.0, 2.0, 3.0],
        "vector_y": [4.0, 5.0, 6.0]
    }
}'
```

# nvidia-smi
```shell
Mon May 19 09:56:03 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.51.03              Driver Version: 575.51.03      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce MX150           Off |   00000000:01:00.0 Off |                  N/A |
| N/A   43C    P8            N/A  / 5001W |       7MiB /   2048MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1759      G   /usr/lib/xorg/Xorg                        4MiB |
+-----------------------------------------------------------------------------------------+
```

# nvcc --version
```shell
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Apr__9_19:24:57_PDT_2025
Cuda compilation tools, release 12.9, V12.9.41
Build cuda_12.9.r12.9/compiler.35813241_0
```
