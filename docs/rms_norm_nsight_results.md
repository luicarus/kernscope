形状 512×1024，dtype BF16。

| 阶段 | 算子 | Kernel 时长 (µs) | SM 吞吐 | DRAM 吞吐 | DRAM Bytes | Registers / Thread | Achieved Occupancy |
|---|---|---:|---:|---:|---:|---:|---:|
| Kernel 优化前 | 普通 RMSNorm (1 row/program) | 12.544 | 17.80% | 82.82% | 1,872,640 | 26 | 78.73% |
| Kernel 优化后 | 普通 RMSNorm (4 rows/program, 8 warps) | 10.176 | 18.16% | 86.46% | 1,674,112 | 40 | 67.49% |
| Kernel 优化前 | 融合 add RMSNorm (1 row/program, 4 warps) | 20.064 | 12.02% | 88.72% | 3,376,896 | 29 | 84.93% |
| Kernel 优化后 | 融合 add RMSNorm (1 row/program, 1 warp) | 19.136 | 8.37% | 90.20% | 3,229,952 | 84 | 25.95% |
