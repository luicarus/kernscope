形状 512×1024，dtype BF16。普通与融合 RMSNorm 优化前后均为 3 次 kernel launch 的中位数。DRAM Bytes = `dram__sectors.sum` × 32 B。

| 阶段 | 算子 | Kernel 时长 (µs) | SM 吞吐 | DRAM 吞吐 | DRAM Bytes | Registers / Thread | Achieved Occupancy |
|---|---|---:|---:|---:|---:|---:|---:|
| Kernel 优化前 | 普通 RMSNorm (1 row/program) | 12.544 | 17.80% | 82.82% | 1,872,640 | 26 | 78.73% |
| Kernel 优化后 | 普通 RMSNorm (4 rows/program, 8 warps) | 10.176 | 18.16% | 86.46% | 1,674,112 | 40 | 67.49% |
| Kernel 优化前 | 融合 add RMSNorm (1 row/program, 4 warps) | 20.064 | 12.02% | 88.72% | 3,376,896 | 29 | 84.93% |
| Kernel 优化后 | 融合 add RMSNorm (1 row/program, 1 warp) | 19.136 | 8.37% | 90.20% | 3,229,952 | 84 | 25.95% |

4 rows/program、8 warps 使普通 RMSNorm 时长下降 18.9%、DRAM Bytes 下降 10.6%。

融合 RMSNorm 使用 1 warp 后时长下降 4.6%、DRAM Bytes 下降 4.4%；Registers / Thread 增至 84，Achieved Occupancy 降至 25.95%。
