输入形状 512×1024，输出形状 512×1024，dtype BF16。初始基线三次 launch；本轮行分组对照各五次，均取中位数。

| 阶段 | 算子 | Kernel 时长 (µs) | SM 吞吐 | DRAM 吞吐 | DRAM Bytes | Registers / Thread | Achieved Occupancy |
|---|---|---:|---:|---:|---:|---:|---:|
| Kernel 优化前 | Softmax (1 row/program, 4 warps) | 10.464 | 33.02% | 86.57% | 1,783,424 | 23 | 77.17% |
| 行分组前（同轮） | Softmax (1 row/program, 4 warps) | 10.912 | 32.26% | 87.03% | 1,823,360 | 23 | 76.40% |
| 行分组后（同轮） | Softmax (2 rows/program, 4 warps) | 10.592 | 28.28% | 86.28% | 1,686,272 | 34 | 64.42% |

短行对照：输入／输出形状 512×128，dtype BF16。warp 对照各三次 launch；本轮行分组对照各五次，均取中位数。

| 阶段 | 算子 | Kernel 时长 (µs) | SM 吞吐 | DRAM 吞吐 | DRAM Bytes | Registers / Thread | Achieved Occupancy |
|---|---|---:|---:|---:|---:|---:|---:|
| Kernel 优化前 | Softmax (1 row/program, 4 warps) | 6.304 | 57.72% | 15.69% | 183,424 | 16 | 82.49% |
| Kernel 优化后 | Softmax (1 row/program, 1 warp) | 4.160 | 11.25% | 23.34% | 179,712 | 16 | 22.11% |
| 行分组前（同轮） | Softmax (1 row/program, 1 warp) | 4.096 | 11.22% | 23.70% | 180,480 | 16 | 22.52% |
| 行分组后（同轮） | Softmax (2 rows/program, 1 warp) | 3.552 | 11.02% | 21.46% | 131,840 | 18 | 22.02% |

短行对照：输入／输出形状 512×256，dtype BF16。warp 对照各三次 launch；本轮行分组对照各五次，均取中位数。

| 阶段 | 算子 | Kernel 时长 (µs) | SM 吞吐 | DRAM 吞吐 | DRAM Bytes | Registers / Thread | Achieved Occupancy |
|---|---|---:|---:|---:|---:|---:|---:|
| Kernel 优化前 | Softmax (1 row/program, 4 warps) | 6.656 | 56.09% | 32.73% | 394,112 | 17 | 83.87% |
| Kernel 优化后 | Softmax (1 row/program, 1 warp) | 4.672 | 14.38% | 45.49% | 394,880 | 20 | 24.38% |
| 行分组前（同轮） | Softmax (1 row/program, 1 warp) | 4.864 | 13.97% | 45.57% | 403,200 | 20 | 22.91% |
| 行分组后（同轮） | Softmax (2 rows/program, 1 warp) | 4.384 | 15.99% | 49.97% | 392,960 | 31 | 21.14% |
