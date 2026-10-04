向量长度 H=4096，权重形状 4096×4096，输出长度 N=4096，dtype BF16。

| 阶段 | 算子 | Kernel 时长 (µs) | SM 吞吐 | DRAM 吞吐 | DRAM Bytes | Registers / Thread | Achieved Occupancy |
|---|---|---:|---:|---:|---:|---:|---:|
| Kernel 优化前 | GEMV (1024 columns/block, 4 warps) | 181.376 | 13.97% | 96.67% | 33,592,192 | 40 | 97.04% |
| Kernel 优化后 | GEMV (256 columns/block, 4 warps) | 181.376 | 26.87% | 96.52% | 33,591,936 | 40 | 97.33% |

向量长度 H=8192，权重形状 4096×8192，输出长度 N=4096，dtype BF16。

| 阶段 | 算子 | Kernel 时长 (µs) | SM 吞吐 | DRAM 吞吐 | DRAM Bytes | Registers / Thread | Achieved Occupancy |
|---|---|---:|---:|---:|---:|---:|---:|
| Kernel 优化前 | GEMV (1024 columns/block, 4 warps) | 392.416 | 11.92% | 94.84% | 71,373,696 | 64 | 65.55% |
| Kernel 优化后 | GEMV (256 columns/block, 4 warps) | 360.544 | 23.46% | 97.21% | 67,224,960 | 39 | 97.88% |

向量长度 H=1024，权重形状 512×1024，输出长度 N=512，dtype BF16。

| 阶段 | 算子 | Kernel 时长 (µs) | SM 吞吐 | DRAM 吞吐 | DRAM Bytes | Registers / Thread | Achieved Occupancy |
|---|---|---:|---:|---:|---:|---:|---:|
| Kernel 优化前 | GEMV (256 columns/block, 4 warps) | 8.000 | 34.62% | 68.97% | 1,056,512 | 22 | 78.34% |
| Kernel 优化后 | GEMV (1024 columns/block, 4 warps) | 8.064 | 23.81% | 68.62% | 1,056,768 | 21 | 78.87% |
