# ResNet-110 乘法深度分析

## CKKS 参数

| 参数 | 值 |
|------|----|
| levels_before | 10 |
| bootstrap_depth | 18 |
| circuit_depth | 28 |
| 输入层级 | circuit_depth - 4 = 24 |
| 自举后层级 | 10 |
| n (block 数) | 18 |

## 各操作深度代价（从算法推导）

| 操作 | 深度代价 | 推导依据 |
|------|---------|----------|
| ConvBN (Algorithm 7) | 2 | ① `ctx.mul(rot, weight)` +1；② `ctx.mul(rotated, sel)` +1 |
| DownSamp (Algorithm 5) | 1 | ① `ctx.mul(in, sel)` +1 |
| AvgPool (Algorithm 6) | 1 | ① `ctx.mul(rot_a, sel)` +1 |
| FC (对角线方法) | 1 | ① `ctx.mul(rot_in, diag)` +1 |
| AppReLU (degree 27) | 5 | Chebyshev PS: k=4, m=3, GetDepthByDegree(27)=6, a=-1,b=1 省 1 |
| Bootstrap | → 10 | 恢复至 levels_before |
| Add (skip) | 0 | min(主路径, skip 路径) |
| Rot / SumSlots | 0 | 仅槽位旋转+加法 |

## 逐层深度追踪

| # | Stage | 操作 | 说明 | 输入层级 | 消耗 | 输出层级 |
|---|-------|------|------|---------|------|----------|
| 1 | Stage1 | ConvBN1 | conv(PP_CONV1) = 2 levels | 24 | -2 | 22 |
| 2 | Stage1 | AppReLU | degree 27 = 5 levels | 22 | -5 | 17 |
| 3 | Stage2 blk1 | ConvBN2a | conv(PP_CONV2) | 17 | -2 | 15 |
| 4 | Stage2 blk1 | Bootstrap | → level 10 | 15 | boot | 10 |
| 5 | Stage2 blk1 | AppReLU | degree 27 | 10 | -5 | 5 |
| 6 | Stage2 blk1 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 7 | Stage2 blk1 | Add(skip) | min(3,17) | 3 | 0 | 3 |
| 8 | Stage2 blk1 | Bootstrap | → level 10 | 3 | boot | 10 |
| 9 | Stage2 blk1 | AppReLU | degree 27 | 10 | -5 | 5 |
| 10 | Stage2 blk2 | ConvBN2a | conv(PP_CONV2) | 5 | -2 | 3 |
| 11 | Stage2 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 12 | Stage2 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 13 | Stage2 blk2 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 14 | Stage2 blk2 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 15 | Stage2 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 16 | Stage2 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 17 | Stage2 blk3 | ConvBN2a | conv(PP_CONV2) | 5 | -2 | 3 |
| 18 | Stage2 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 19 | Stage2 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 20 | Stage2 blk3 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 21 | Stage2 blk3 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 22 | Stage2 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 23 | Stage2 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 24 | Stage2 blk4 | ConvBN2a | conv(PP_CONV2) | 5 | -2 | 3 |
| 25 | Stage2 blk4 | Bootstrap | → level 10 | 3 | boot | 10 |
| 26 | Stage2 blk4 | AppReLU | degree 27 | 10 | -5 | 5 |
| 27 | Stage2 blk4 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 28 | Stage2 blk4 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 29 | Stage2 blk4 | Bootstrap | → level 10 | 3 | boot | 10 |
| 30 | Stage2 blk4 | AppReLU | degree 27 | 10 | -5 | 5 |
| 31 | Stage2 blk5 | ConvBN2a | conv(PP_CONV2) | 5 | -2 | 3 |
| 32 | Stage2 blk5 | Bootstrap | → level 10 | 3 | boot | 10 |
| 33 | Stage2 blk5 | AppReLU | degree 27 | 10 | -5 | 5 |
| 34 | Stage2 blk5 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 35 | Stage2 blk5 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 36 | Stage2 blk5 | Bootstrap | → level 10 | 3 | boot | 10 |
| 37 | Stage2 blk5 | AppReLU | degree 27 | 10 | -5 | 5 |
| 38 | Stage2 blk6 | ConvBN2a | conv(PP_CONV2) | 5 | -2 | 3 |
| 39 | Stage2 blk6 | Bootstrap | → level 10 | 3 | boot | 10 |
| 40 | Stage2 blk6 | AppReLU | degree 27 | 10 | -5 | 5 |
| 41 | Stage2 blk6 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 42 | Stage2 blk6 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 43 | Stage2 blk6 | Bootstrap | → level 10 | 3 | boot | 10 |
| 44 | Stage2 blk6 | AppReLU | degree 27 | 10 | -5 | 5 |
| 45 | Stage2 blk7 | ConvBN2a | conv(PP_CONV2) | 5 | -2 | 3 |
| 46 | Stage2 blk7 | Bootstrap | → level 10 | 3 | boot | 10 |
| 47 | Stage2 blk7 | AppReLU | degree 27 | 10 | -5 | 5 |
| 48 | Stage2 blk7 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 49 | Stage2 blk7 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 50 | Stage2 blk7 | Bootstrap | → level 10 | 3 | boot | 10 |
| 51 | Stage2 blk7 | AppReLU | degree 27 | 10 | -5 | 5 |
| 52 | Stage2 blk8 | ConvBN2a | conv(PP_CONV2) | 5 | -2 | 3 |
| 53 | Stage2 blk8 | Bootstrap | → level 10 | 3 | boot | 10 |
| 54 | Stage2 blk8 | AppReLU | degree 27 | 10 | -5 | 5 |
| 55 | Stage2 blk8 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 56 | Stage2 blk8 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 57 | Stage2 blk8 | Bootstrap | → level 10 | 3 | boot | 10 |
| 58 | Stage2 blk8 | AppReLU | degree 27 | 10 | -5 | 5 |
| 59 | Stage2 blk9 | ConvBN2a | conv(PP_CONV2) | 5 | -2 | 3 |
| 60 | Stage2 blk9 | Bootstrap | → level 10 | 3 | boot | 10 |
| 61 | Stage2 blk9 | AppReLU | degree 27 | 10 | -5 | 5 |
| 62 | Stage2 blk9 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 63 | Stage2 blk9 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 64 | Stage2 blk9 | Bootstrap | → level 10 | 3 | boot | 10 |
| 65 | Stage2 blk9 | AppReLU | degree 27 | 10 | -5 | 5 |
| 66 | Stage2 blk10 | ConvBN2a | conv(PP_CONV2) | 5 | -2 | 3 |
| 67 | Stage2 blk10 | Bootstrap | → level 10 | 3 | boot | 10 |
| 68 | Stage2 blk10 | AppReLU | degree 27 | 10 | -5 | 5 |
| 69 | Stage2 blk10 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 70 | Stage2 blk10 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 71 | Stage2 blk10 | Bootstrap | → level 10 | 3 | boot | 10 |
| 72 | Stage2 blk10 | AppReLU | degree 27 | 10 | -5 | 5 |
| 73 | Stage2 blk11 | ConvBN2a | conv(PP_CONV2) | 5 | -2 | 3 |
| 74 | Stage2 blk11 | Bootstrap | → level 10 | 3 | boot | 10 |
| 75 | Stage2 blk11 | AppReLU | degree 27 | 10 | -5 | 5 |
| 76 | Stage2 blk11 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 77 | Stage2 blk11 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 78 | Stage2 blk11 | Bootstrap | → level 10 | 3 | boot | 10 |
| 79 | Stage2 blk11 | AppReLU | degree 27 | 10 | -5 | 5 |
| 80 | Stage2 blk12 | ConvBN2a | conv(PP_CONV2) | 5 | -2 | 3 |
| 81 | Stage2 blk12 | Bootstrap | → level 10 | 3 | boot | 10 |
| 82 | Stage2 blk12 | AppReLU | degree 27 | 10 | -5 | 5 |
| 83 | Stage2 blk12 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 84 | Stage2 blk12 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 85 | Stage2 blk12 | Bootstrap | → level 10 | 3 | boot | 10 |
| 86 | Stage2 blk12 | AppReLU | degree 27 | 10 | -5 | 5 |
| 87 | Stage2 blk13 | ConvBN2a | conv(PP_CONV2) | 5 | -2 | 3 |
| 88 | Stage2 blk13 | Bootstrap | → level 10 | 3 | boot | 10 |
| 89 | Stage2 blk13 | AppReLU | degree 27 | 10 | -5 | 5 |
| 90 | Stage2 blk13 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 91 | Stage2 blk13 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 92 | Stage2 blk13 | Bootstrap | → level 10 | 3 | boot | 10 |
| 93 | Stage2 blk13 | AppReLU | degree 27 | 10 | -5 | 5 |
| 94 | Stage2 blk14 | ConvBN2a | conv(PP_CONV2) | 5 | -2 | 3 |
| 95 | Stage2 blk14 | Bootstrap | → level 10 | 3 | boot | 10 |
| 96 | Stage2 blk14 | AppReLU | degree 27 | 10 | -5 | 5 |
| 97 | Stage2 blk14 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 98 | Stage2 blk14 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 99 | Stage2 blk14 | Bootstrap | → level 10 | 3 | boot | 10 |
| 100 | Stage2 blk14 | AppReLU | degree 27 | 10 | -5 | 5 |
| 101 | Stage2 blk15 | ConvBN2a | conv(PP_CONV2) | 5 | -2 | 3 |
| 102 | Stage2 blk15 | Bootstrap | → level 10 | 3 | boot | 10 |
| 103 | Stage2 blk15 | AppReLU | degree 27 | 10 | -5 | 5 |
| 104 | Stage2 blk15 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 105 | Stage2 blk15 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 106 | Stage2 blk15 | Bootstrap | → level 10 | 3 | boot | 10 |
| 107 | Stage2 blk15 | AppReLU | degree 27 | 10 | -5 | 5 |
| 108 | Stage2 blk16 | ConvBN2a | conv(PP_CONV2) | 5 | -2 | 3 |
| 109 | Stage2 blk16 | Bootstrap | → level 10 | 3 | boot | 10 |
| 110 | Stage2 blk16 | AppReLU | degree 27 | 10 | -5 | 5 |
| 111 | Stage2 blk16 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 112 | Stage2 blk16 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 113 | Stage2 blk16 | Bootstrap | → level 10 | 3 | boot | 10 |
| 114 | Stage2 blk16 | AppReLU | degree 27 | 10 | -5 | 5 |
| 115 | Stage2 blk17 | ConvBN2a | conv(PP_CONV2) | 5 | -2 | 3 |
| 116 | Stage2 blk17 | Bootstrap | → level 10 | 3 | boot | 10 |
| 117 | Stage2 blk17 | AppReLU | degree 27 | 10 | -5 | 5 |
| 118 | Stage2 blk17 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 119 | Stage2 blk17 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 120 | Stage2 blk17 | Bootstrap | → level 10 | 3 | boot | 10 |
| 121 | Stage2 blk17 | AppReLU | degree 27 | 10 | -5 | 5 |
| 122 | Stage2 blk18 | ConvBN2a | conv(PP_CONV2) | 5 | -2 | 3 |
| 123 | Stage2 blk18 | Bootstrap | → level 10 | 3 | boot | 10 |
| 124 | Stage2 blk18 | AppReLU | degree 27 | 10 | -5 | 5 |
| 125 | Stage2 blk18 | ConvBN2b | conv(PP_CONV2) | 5 | -2 | 3 |
| 126 | Stage2 blk18 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 127 | Stage2 blk18 | Bootstrap | → level 10 | 3 | boot | 10 |
| 128 | Stage2 blk18 | AppReLU | degree 27 | 10 | -5 | 5 |
| 129 | Stage3 blk1 main | ConvBN3_1a | conv(PP_CONV3A) | 5 | -2 | 3 |
| 130 | Stage3 blk1 main | Bootstrap | → level 10 | 3 | boot | 10 |
| 131 | Stage3 blk1 main | AppReLU | degree 27 | 10 | -5 | 5 |
| 132 | Stage3 blk1 main | ConvBN3_1b | conv(PP_CONV3) | 5 | -2 | 3 |
| 133 | Stage3 blk1 skip | ConvBN3_s1 | conv(PP_CONV3A) | 5 | -2 | 3 |
| 134 | Stage3 blk1 skip | DownSamp1 | downsamp(PP_CONV3A) | 3 | -1 | 2 **← min** |
| 135 | Stage3 blk1 | Add(skip) | min(3,2) | 3 | 0 | 2 **← min** |
| 136 | Stage3 blk1 | Bootstrap | → level 10 | 2 | boot | 10 |
| 137 | Stage3 blk1 | AppReLU | degree 27 | 10 | -5 | 5 |
| 138 | Stage3 blk2 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 139 | Stage3 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 140 | Stage3 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 141 | Stage3 blk2 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 142 | Stage3 blk2 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 143 | Stage3 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 144 | Stage3 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 145 | Stage3 blk3 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 146 | Stage3 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 147 | Stage3 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 148 | Stage3 blk3 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 149 | Stage3 blk3 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 150 | Stage3 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 151 | Stage3 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 152 | Stage3 blk4 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 153 | Stage3 blk4 | Bootstrap | → level 10 | 3 | boot | 10 |
| 154 | Stage3 blk4 | AppReLU | degree 27 | 10 | -5 | 5 |
| 155 | Stage3 blk4 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 156 | Stage3 blk4 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 157 | Stage3 blk4 | Bootstrap | → level 10 | 3 | boot | 10 |
| 158 | Stage3 blk4 | AppReLU | degree 27 | 10 | -5 | 5 |
| 159 | Stage3 blk5 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 160 | Stage3 blk5 | Bootstrap | → level 10 | 3 | boot | 10 |
| 161 | Stage3 blk5 | AppReLU | degree 27 | 10 | -5 | 5 |
| 162 | Stage3 blk5 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 163 | Stage3 blk5 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 164 | Stage3 blk5 | Bootstrap | → level 10 | 3 | boot | 10 |
| 165 | Stage3 blk5 | AppReLU | degree 27 | 10 | -5 | 5 |
| 166 | Stage3 blk6 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 167 | Stage3 blk6 | Bootstrap | → level 10 | 3 | boot | 10 |
| 168 | Stage3 blk6 | AppReLU | degree 27 | 10 | -5 | 5 |
| 169 | Stage3 blk6 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 170 | Stage3 blk6 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 171 | Stage3 blk6 | Bootstrap | → level 10 | 3 | boot | 10 |
| 172 | Stage3 blk6 | AppReLU | degree 27 | 10 | -5 | 5 |
| 173 | Stage3 blk7 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 174 | Stage3 blk7 | Bootstrap | → level 10 | 3 | boot | 10 |
| 175 | Stage3 blk7 | AppReLU | degree 27 | 10 | -5 | 5 |
| 176 | Stage3 blk7 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 177 | Stage3 blk7 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 178 | Stage3 blk7 | Bootstrap | → level 10 | 3 | boot | 10 |
| 179 | Stage3 blk7 | AppReLU | degree 27 | 10 | -5 | 5 |
| 180 | Stage3 blk8 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 181 | Stage3 blk8 | Bootstrap | → level 10 | 3 | boot | 10 |
| 182 | Stage3 blk8 | AppReLU | degree 27 | 10 | -5 | 5 |
| 183 | Stage3 blk8 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 184 | Stage3 blk8 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 185 | Stage3 blk8 | Bootstrap | → level 10 | 3 | boot | 10 |
| 186 | Stage3 blk8 | AppReLU | degree 27 | 10 | -5 | 5 |
| 187 | Stage3 blk9 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 188 | Stage3 blk9 | Bootstrap | → level 10 | 3 | boot | 10 |
| 189 | Stage3 blk9 | AppReLU | degree 27 | 10 | -5 | 5 |
| 190 | Stage3 blk9 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 191 | Stage3 blk9 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 192 | Stage3 blk9 | Bootstrap | → level 10 | 3 | boot | 10 |
| 193 | Stage3 blk9 | AppReLU | degree 27 | 10 | -5 | 5 |
| 194 | Stage3 blk10 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 195 | Stage3 blk10 | Bootstrap | → level 10 | 3 | boot | 10 |
| 196 | Stage3 blk10 | AppReLU | degree 27 | 10 | -5 | 5 |
| 197 | Stage3 blk10 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 198 | Stage3 blk10 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 199 | Stage3 blk10 | Bootstrap | → level 10 | 3 | boot | 10 |
| 200 | Stage3 blk10 | AppReLU | degree 27 | 10 | -5 | 5 |
| 201 | Stage3 blk11 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 202 | Stage3 blk11 | Bootstrap | → level 10 | 3 | boot | 10 |
| 203 | Stage3 blk11 | AppReLU | degree 27 | 10 | -5 | 5 |
| 204 | Stage3 blk11 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 205 | Stage3 blk11 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 206 | Stage3 blk11 | Bootstrap | → level 10 | 3 | boot | 10 |
| 207 | Stage3 blk11 | AppReLU | degree 27 | 10 | -5 | 5 |
| 208 | Stage3 blk12 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 209 | Stage3 blk12 | Bootstrap | → level 10 | 3 | boot | 10 |
| 210 | Stage3 blk12 | AppReLU | degree 27 | 10 | -5 | 5 |
| 211 | Stage3 blk12 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 212 | Stage3 blk12 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 213 | Stage3 blk12 | Bootstrap | → level 10 | 3 | boot | 10 |
| 214 | Stage3 blk12 | AppReLU | degree 27 | 10 | -5 | 5 |
| 215 | Stage3 blk13 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 216 | Stage3 blk13 | Bootstrap | → level 10 | 3 | boot | 10 |
| 217 | Stage3 blk13 | AppReLU | degree 27 | 10 | -5 | 5 |
| 218 | Stage3 blk13 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 219 | Stage3 blk13 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 220 | Stage3 blk13 | Bootstrap | → level 10 | 3 | boot | 10 |
| 221 | Stage3 blk13 | AppReLU | degree 27 | 10 | -5 | 5 |
| 222 | Stage3 blk14 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 223 | Stage3 blk14 | Bootstrap | → level 10 | 3 | boot | 10 |
| 224 | Stage3 blk14 | AppReLU | degree 27 | 10 | -5 | 5 |
| 225 | Stage3 blk14 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 226 | Stage3 blk14 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 227 | Stage3 blk14 | Bootstrap | → level 10 | 3 | boot | 10 |
| 228 | Stage3 blk14 | AppReLU | degree 27 | 10 | -5 | 5 |
| 229 | Stage3 blk15 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 230 | Stage3 blk15 | Bootstrap | → level 10 | 3 | boot | 10 |
| 231 | Stage3 blk15 | AppReLU | degree 27 | 10 | -5 | 5 |
| 232 | Stage3 blk15 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 233 | Stage3 blk15 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 234 | Stage3 blk15 | Bootstrap | → level 10 | 3 | boot | 10 |
| 235 | Stage3 blk15 | AppReLU | degree 27 | 10 | -5 | 5 |
| 236 | Stage3 blk16 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 237 | Stage3 blk16 | Bootstrap | → level 10 | 3 | boot | 10 |
| 238 | Stage3 blk16 | AppReLU | degree 27 | 10 | -5 | 5 |
| 239 | Stage3 blk16 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 240 | Stage3 blk16 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 241 | Stage3 blk16 | Bootstrap | → level 10 | 3 | boot | 10 |
| 242 | Stage3 blk16 | AppReLU | degree 27 | 10 | -5 | 5 |
| 243 | Stage3 blk17 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 244 | Stage3 blk17 | Bootstrap | → level 10 | 3 | boot | 10 |
| 245 | Stage3 blk17 | AppReLU | degree 27 | 10 | -5 | 5 |
| 246 | Stage3 blk17 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 247 | Stage3 blk17 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 248 | Stage3 blk17 | Bootstrap | → level 10 | 3 | boot | 10 |
| 249 | Stage3 blk17 | AppReLU | degree 27 | 10 | -5 | 5 |
| 250 | Stage3 blk18 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 251 | Stage3 blk18 | Bootstrap | → level 10 | 3 | boot | 10 |
| 252 | Stage3 blk18 | AppReLU | degree 27 | 10 | -5 | 5 |
| 253 | Stage3 blk18 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 254 | Stage3 blk18 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 255 | Stage3 blk18 | Bootstrap | → level 10 | 3 | boot | 10 |
| 256 | Stage3 blk18 | AppReLU | degree 27 | 10 | -5 | 5 |
| 257 | Stage4 blk1 main | ConvBN4_1a | conv(PP_CONV4A) | 5 | -2 | 3 |
| 258 | Stage4 blk1 main | Bootstrap | → level 10 | 3 | boot | 10 |
| 259 | Stage4 blk1 main | AppReLU | degree 27 | 10 | -5 | 5 |
| 260 | Stage4 blk1 main | ConvBN4_1b | conv(PP_CONV4) | 5 | -2 | 3 |
| 261 | Stage4 blk1 skip | ConvBN4_s2 | conv(PP_CONV4A) | 5 | -2 | 3 |
| 262 | Stage4 blk1 skip | DownSamp2 | downsamp(PP_CONV4A) | 3 | -1 | 2 **← min** |
| 263 | Stage4 blk1 | Add(skip) | min(3,2) | 3 | 0 | 2 **← min** |
| 264 | Stage4 blk1 | Bootstrap | → level 10 | 2 | boot | 10 |
| 265 | Stage4 blk1 | AppReLU | degree 27 | 10 | -5 | 5 |
| 266 | Stage4 blk2 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 267 | Stage4 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 268 | Stage4 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 269 | Stage4 blk2 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 270 | Stage4 blk2 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 271 | Stage4 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 272 | Stage4 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 273 | Stage4 blk3 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 274 | Stage4 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 275 | Stage4 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 276 | Stage4 blk3 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 277 | Stage4 blk3 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 278 | Stage4 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 279 | Stage4 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 280 | Stage4 blk4 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 281 | Stage4 blk4 | Bootstrap | → level 10 | 3 | boot | 10 |
| 282 | Stage4 blk4 | AppReLU | degree 27 | 10 | -5 | 5 |
| 283 | Stage4 blk4 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 284 | Stage4 blk4 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 285 | Stage4 blk4 | Bootstrap | → level 10 | 3 | boot | 10 |
| 286 | Stage4 blk4 | AppReLU | degree 27 | 10 | -5 | 5 |
| 287 | Stage4 blk5 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 288 | Stage4 blk5 | Bootstrap | → level 10 | 3 | boot | 10 |
| 289 | Stage4 blk5 | AppReLU | degree 27 | 10 | -5 | 5 |
| 290 | Stage4 blk5 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 291 | Stage4 blk5 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 292 | Stage4 blk5 | Bootstrap | → level 10 | 3 | boot | 10 |
| 293 | Stage4 blk5 | AppReLU | degree 27 | 10 | -5 | 5 |
| 294 | Stage4 blk6 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 295 | Stage4 blk6 | Bootstrap | → level 10 | 3 | boot | 10 |
| 296 | Stage4 blk6 | AppReLU | degree 27 | 10 | -5 | 5 |
| 297 | Stage4 blk6 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 298 | Stage4 blk6 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 299 | Stage4 blk6 | Bootstrap | → level 10 | 3 | boot | 10 |
| 300 | Stage4 blk6 | AppReLU | degree 27 | 10 | -5 | 5 |
| 301 | Stage4 blk7 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 302 | Stage4 blk7 | Bootstrap | → level 10 | 3 | boot | 10 |
| 303 | Stage4 blk7 | AppReLU | degree 27 | 10 | -5 | 5 |
| 304 | Stage4 blk7 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 305 | Stage4 blk7 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 306 | Stage4 blk7 | Bootstrap | → level 10 | 3 | boot | 10 |
| 307 | Stage4 blk7 | AppReLU | degree 27 | 10 | -5 | 5 |
| 308 | Stage4 blk8 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 309 | Stage4 blk8 | Bootstrap | → level 10 | 3 | boot | 10 |
| 310 | Stage4 blk8 | AppReLU | degree 27 | 10 | -5 | 5 |
| 311 | Stage4 blk8 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 312 | Stage4 blk8 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 313 | Stage4 blk8 | Bootstrap | → level 10 | 3 | boot | 10 |
| 314 | Stage4 blk8 | AppReLU | degree 27 | 10 | -5 | 5 |
| 315 | Stage4 blk9 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 316 | Stage4 blk9 | Bootstrap | → level 10 | 3 | boot | 10 |
| 317 | Stage4 blk9 | AppReLU | degree 27 | 10 | -5 | 5 |
| 318 | Stage4 blk9 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 319 | Stage4 blk9 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 320 | Stage4 blk9 | Bootstrap | → level 10 | 3 | boot | 10 |
| 321 | Stage4 blk9 | AppReLU | degree 27 | 10 | -5 | 5 |
| 322 | Stage4 blk10 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 323 | Stage4 blk10 | Bootstrap | → level 10 | 3 | boot | 10 |
| 324 | Stage4 blk10 | AppReLU | degree 27 | 10 | -5 | 5 |
| 325 | Stage4 blk10 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 326 | Stage4 blk10 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 327 | Stage4 blk10 | Bootstrap | → level 10 | 3 | boot | 10 |
| 328 | Stage4 blk10 | AppReLU | degree 27 | 10 | -5 | 5 |
| 329 | Stage4 blk11 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 330 | Stage4 blk11 | Bootstrap | → level 10 | 3 | boot | 10 |
| 331 | Stage4 blk11 | AppReLU | degree 27 | 10 | -5 | 5 |
| 332 | Stage4 blk11 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 333 | Stage4 blk11 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 334 | Stage4 blk11 | Bootstrap | → level 10 | 3 | boot | 10 |
| 335 | Stage4 blk11 | AppReLU | degree 27 | 10 | -5 | 5 |
| 336 | Stage4 blk12 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 337 | Stage4 blk12 | Bootstrap | → level 10 | 3 | boot | 10 |
| 338 | Stage4 blk12 | AppReLU | degree 27 | 10 | -5 | 5 |
| 339 | Stage4 blk12 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 340 | Stage4 blk12 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 341 | Stage4 blk12 | Bootstrap | → level 10 | 3 | boot | 10 |
| 342 | Stage4 blk12 | AppReLU | degree 27 | 10 | -5 | 5 |
| 343 | Stage4 blk13 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 344 | Stage4 blk13 | Bootstrap | → level 10 | 3 | boot | 10 |
| 345 | Stage4 blk13 | AppReLU | degree 27 | 10 | -5 | 5 |
| 346 | Stage4 blk13 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 347 | Stage4 blk13 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 348 | Stage4 blk13 | Bootstrap | → level 10 | 3 | boot | 10 |
| 349 | Stage4 blk13 | AppReLU | degree 27 | 10 | -5 | 5 |
| 350 | Stage4 blk14 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 351 | Stage4 blk14 | Bootstrap | → level 10 | 3 | boot | 10 |
| 352 | Stage4 blk14 | AppReLU | degree 27 | 10 | -5 | 5 |
| 353 | Stage4 blk14 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 354 | Stage4 blk14 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 355 | Stage4 blk14 | Bootstrap | → level 10 | 3 | boot | 10 |
| 356 | Stage4 blk14 | AppReLU | degree 27 | 10 | -5 | 5 |
| 357 | Stage4 blk15 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 358 | Stage4 blk15 | Bootstrap | → level 10 | 3 | boot | 10 |
| 359 | Stage4 blk15 | AppReLU | degree 27 | 10 | -5 | 5 |
| 360 | Stage4 blk15 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 361 | Stage4 blk15 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 362 | Stage4 blk15 | Bootstrap | → level 10 | 3 | boot | 10 |
| 363 | Stage4 blk15 | AppReLU | degree 27 | 10 | -5 | 5 |
| 364 | Stage4 blk16 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 365 | Stage4 blk16 | Bootstrap | → level 10 | 3 | boot | 10 |
| 366 | Stage4 blk16 | AppReLU | degree 27 | 10 | -5 | 5 |
| 367 | Stage4 blk16 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 368 | Stage4 blk16 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 369 | Stage4 blk16 | Bootstrap | → level 10 | 3 | boot | 10 |
| 370 | Stage4 blk16 | AppReLU | degree 27 | 10 | -5 | 5 |
| 371 | Stage4 blk17 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 372 | Stage4 blk17 | Bootstrap | → level 10 | 3 | boot | 10 |
| 373 | Stage4 blk17 | AppReLU | degree 27 | 10 | -5 | 5 |
| 374 | Stage4 blk17 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 375 | Stage4 blk17 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 376 | Stage4 blk17 | Bootstrap | → level 10 | 3 | boot | 10 |
| 377 | Stage4 blk17 | AppReLU | degree 27 | 10 | -5 | 5 |
| 378 | Stage4 blk18 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 379 | Stage4 blk18 | Bootstrap | → level 10 | 3 | boot | 10 |
| 380 | Stage4 blk18 | AppReLU | degree 27 | 10 | -5 | 5 |
| 381 | Stage4 blk18 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 382 | Stage4 blk18 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 383 | Stage4 blk18 | Bootstrap | → level 10 | 3 | boot | 10 |
| 384 | Stage4 blk18 | AppReLU | degree 27 | 10 | -5 | 5 |
| 385 | Head | AvgPool | avg_pool(PP_POOL) | 5 | -1 | 4 |
| 386 | Head | FC | fc(64→10) | 4 | -1 | 3 |

## 关键路径分析

- **最低层级**: 2（位于 Stage3 blk1 skip / DownSamp1）
- **状态**: 安全（余量 2 级）
- **自举间最大消耗**: AppReLU(5) + ConvBN(2) = 7 级
- **转换块 skip 路径**: ConvBN(2) + DownSamp(1) = 3 级，入口层级 5，出口层级 2

## 自举间层级预算

```
Bootstrap 输出: level 10
  ├─ AppReLU:   -5 → level 5
  ├─ ConvBN:    -2 → level 3  (进入下一次 Bootstrap)
  │
  └─ 转换块 skip 路径（最深）:
     ├─ ConvBN:    -2 → level 3
     └─ DownSamp:  -1 → level 2  ← 全局最低
```
