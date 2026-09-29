形状512×1024，dtype BF16。

| 阶段 | 算子 | Kernel 时长 (µs) | SM 吞吐 | DRAM 吞吐 |
|---|---|---:|---:|---:|
| Kernel 优化前 | 普通 RMSNorm | 12.29 | 17.63% | 83.47% |
| Kernel 优化前 | 融合 add RMSNorm | 20.10 | 12.17% | 89.26% |
