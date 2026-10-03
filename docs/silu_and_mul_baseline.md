输入形状 512×8192，输出形状 512×4096，dtype BF16。

| 阶段 | 算子 | Kernel 时长 (µs) | SM 吞吐 | DRAM 吞吐 | DRAM Bytes | Registers / Thread | Achieved Occupancy |
|---|---|---:|---:|---:|---:|---:|---:|
| Kernel 优化前 | SwiGLU (1024 elements/block, 4 warps) | 63.648 | 18.48% | 95.27% | 11,535,616 | 33 | 90.34% |
