形状 512×1024，dtype BF16。普通 RMSNorm 优化前后均为 3 次 kernel launch 的中位数；融合算子为单次基线。DRAM Bytes = `dram__sectors.sum` × 32 B。

| 阶段 | 算子 | Kernel 时长 (µs) | SM 吞吐 | DRAM 吞吐 | DRAM Bytes | Registers / Thread | Achieved Occupancy |
|---|---|---:|---:|---:|---:|---:|---:|
| Kernel 优化前 | 普通 RMSNorm (1 row/program) | 12.544 | 17.80% | 82.82% | 1,872,640 | 26 | 78.73% |
| Kernel 优化后 | 普通 RMSNorm (4 rows/program, 8 warps) | 10.176 | 18.16% | 86.46% | 1,674,112 | 40 | 67.49% |
| Kernel 优化前 | 融合 add RMSNorm | 20.096 | 12.07% | 88.86% | 3,384,832 | 29 | 84.14% |

4 rows/program、8 warps 使普通 RMSNorm 时长下降 18.9%、DRAM Bytes 下降 10.6%。
