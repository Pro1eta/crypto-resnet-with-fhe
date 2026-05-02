# ResNet-56 乘法深度分析

## CKKS 参数

| 参数 | 值 |
|------|----|
| levels_before | 10 |
| bootstrap_depth | 18 |
| circuit_depth | 28 |
| 输入层级 | circuit_depth - 4 = 24 |
| 自举后层级 | 10 |
| n (block 数) | 9 |

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
| 66 | Stage3 blk1 main | ConvBN3_1a | conv(PP_CONV3A) | 5 | -2 | 3 |
| 67 | Stage3 blk1 main | Bootstrap | → level 10 | 3 | boot | 10 |
| 68 | Stage3 blk1 main | AppReLU | degree 27 | 10 | -5 | 5 |
| 69 | Stage3 blk1 main | ConvBN3_1b | conv(PP_CONV3) | 5 | -2 | 3 |
| 70 | Stage3 blk1 skip | ConvBN3_s1 | conv(PP_CONV3A) | 5 | -2 | 3 |
| 71 | Stage3 blk1 skip | DownSamp1 | downsamp(PP_CONV3A) | 3 | -1 | 2 **← min** |
| 72 | Stage3 blk1 | Add(skip) | min(3,2) | 3 | 0 | 2 **← min** |
| 73 | Stage3 blk1 | Bootstrap | → level 10 | 2 | boot | 10 |
| 74 | Stage3 blk1 | AppReLU | degree 27 | 10 | -5 | 5 |
| 75 | Stage3 blk2 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 76 | Stage3 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 77 | Stage3 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 78 | Stage3 blk2 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 79 | Stage3 blk2 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 80 | Stage3 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 81 | Stage3 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 82 | Stage3 blk3 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 83 | Stage3 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 84 | Stage3 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 85 | Stage3 blk3 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 86 | Stage3 blk3 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 87 | Stage3 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 88 | Stage3 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 89 | Stage3 blk4 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 90 | Stage3 blk4 | Bootstrap | → level 10 | 3 | boot | 10 |
| 91 | Stage3 blk4 | AppReLU | degree 27 | 10 | -5 | 5 |
| 92 | Stage3 blk4 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 93 | Stage3 blk4 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 94 | Stage3 blk4 | Bootstrap | → level 10 | 3 | boot | 10 |
| 95 | Stage3 blk4 | AppReLU | degree 27 | 10 | -5 | 5 |
| 96 | Stage3 blk5 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 97 | Stage3 blk5 | Bootstrap | → level 10 | 3 | boot | 10 |
| 98 | Stage3 blk5 | AppReLU | degree 27 | 10 | -5 | 5 |
| 99 | Stage3 blk5 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 100 | Stage3 blk5 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 101 | Stage3 blk5 | Bootstrap | → level 10 | 3 | boot | 10 |
| 102 | Stage3 blk5 | AppReLU | degree 27 | 10 | -5 | 5 |
| 103 | Stage3 blk6 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 104 | Stage3 blk6 | Bootstrap | → level 10 | 3 | boot | 10 |
| 105 | Stage3 blk6 | AppReLU | degree 27 | 10 | -5 | 5 |
| 106 | Stage3 blk6 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 107 | Stage3 blk6 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 108 | Stage3 blk6 | Bootstrap | → level 10 | 3 | boot | 10 |
| 109 | Stage3 blk6 | AppReLU | degree 27 | 10 | -5 | 5 |
| 110 | Stage3 blk7 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 111 | Stage3 blk7 | Bootstrap | → level 10 | 3 | boot | 10 |
| 112 | Stage3 blk7 | AppReLU | degree 27 | 10 | -5 | 5 |
| 113 | Stage3 blk7 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 114 | Stage3 blk7 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 115 | Stage3 blk7 | Bootstrap | → level 10 | 3 | boot | 10 |
| 116 | Stage3 blk7 | AppReLU | degree 27 | 10 | -5 | 5 |
| 117 | Stage3 blk8 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 118 | Stage3 blk8 | Bootstrap | → level 10 | 3 | boot | 10 |
| 119 | Stage3 blk8 | AppReLU | degree 27 | 10 | -5 | 5 |
| 120 | Stage3 blk8 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 121 | Stage3 blk8 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 122 | Stage3 blk8 | Bootstrap | → level 10 | 3 | boot | 10 |
| 123 | Stage3 blk8 | AppReLU | degree 27 | 10 | -5 | 5 |
| 124 | Stage3 blk9 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 125 | Stage3 blk9 | Bootstrap | → level 10 | 3 | boot | 10 |
| 126 | Stage3 blk9 | AppReLU | degree 27 | 10 | -5 | 5 |
| 127 | Stage3 blk9 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 128 | Stage3 blk9 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 129 | Stage3 blk9 | Bootstrap | → level 10 | 3 | boot | 10 |
| 130 | Stage3 blk9 | AppReLU | degree 27 | 10 | -5 | 5 |
| 131 | Stage4 blk1 main | ConvBN4_1a | conv(PP_CONV4A) | 5 | -2 | 3 |
| 132 | Stage4 blk1 main | Bootstrap | → level 10 | 3 | boot | 10 |
| 133 | Stage4 blk1 main | AppReLU | degree 27 | 10 | -5 | 5 |
| 134 | Stage4 blk1 main | ConvBN4_1b | conv(PP_CONV4) | 5 | -2 | 3 |
| 135 | Stage4 blk1 skip | ConvBN4_s2 | conv(PP_CONV4A) | 5 | -2 | 3 |
| 136 | Stage4 blk1 skip | DownSamp2 | downsamp(PP_CONV4A) | 3 | -1 | 2 **← min** |
| 137 | Stage4 blk1 | Add(skip) | min(3,2) | 3 | 0 | 2 **← min** |
| 138 | Stage4 blk1 | Bootstrap | → level 10 | 2 | boot | 10 |
| 139 | Stage4 blk1 | AppReLU | degree 27 | 10 | -5 | 5 |
| 140 | Stage4 blk2 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 141 | Stage4 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 142 | Stage4 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 143 | Stage4 blk2 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 144 | Stage4 blk2 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 145 | Stage4 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 146 | Stage4 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 147 | Stage4 blk3 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 148 | Stage4 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 149 | Stage4 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 150 | Stage4 blk3 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 151 | Stage4 blk3 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 152 | Stage4 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 153 | Stage4 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 154 | Stage4 blk4 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 155 | Stage4 blk4 | Bootstrap | → level 10 | 3 | boot | 10 |
| 156 | Stage4 blk4 | AppReLU | degree 27 | 10 | -5 | 5 |
| 157 | Stage4 blk4 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 158 | Stage4 blk4 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 159 | Stage4 blk4 | Bootstrap | → level 10 | 3 | boot | 10 |
| 160 | Stage4 blk4 | AppReLU | degree 27 | 10 | -5 | 5 |
| 161 | Stage4 blk5 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 162 | Stage4 blk5 | Bootstrap | → level 10 | 3 | boot | 10 |
| 163 | Stage4 blk5 | AppReLU | degree 27 | 10 | -5 | 5 |
| 164 | Stage4 blk5 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 165 | Stage4 blk5 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 166 | Stage4 blk5 | Bootstrap | → level 10 | 3 | boot | 10 |
| 167 | Stage4 blk5 | AppReLU | degree 27 | 10 | -5 | 5 |
| 168 | Stage4 blk6 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 169 | Stage4 blk6 | Bootstrap | → level 10 | 3 | boot | 10 |
| 170 | Stage4 blk6 | AppReLU | degree 27 | 10 | -5 | 5 |
| 171 | Stage4 blk6 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 172 | Stage4 blk6 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 173 | Stage4 blk6 | Bootstrap | → level 10 | 3 | boot | 10 |
| 174 | Stage4 blk6 | AppReLU | degree 27 | 10 | -5 | 5 |
| 175 | Stage4 blk7 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 176 | Stage4 blk7 | Bootstrap | → level 10 | 3 | boot | 10 |
| 177 | Stage4 blk7 | AppReLU | degree 27 | 10 | -5 | 5 |
| 178 | Stage4 blk7 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 179 | Stage4 blk7 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 180 | Stage4 blk7 | Bootstrap | → level 10 | 3 | boot | 10 |
| 181 | Stage4 blk7 | AppReLU | degree 27 | 10 | -5 | 5 |
| 182 | Stage4 blk8 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 183 | Stage4 blk8 | Bootstrap | → level 10 | 3 | boot | 10 |
| 184 | Stage4 blk8 | AppReLU | degree 27 | 10 | -5 | 5 |
| 185 | Stage4 blk8 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 186 | Stage4 blk8 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 187 | Stage4 blk8 | Bootstrap | → level 10 | 3 | boot | 10 |
| 188 | Stage4 blk8 | AppReLU | degree 27 | 10 | -5 | 5 |
| 189 | Stage4 blk9 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 190 | Stage4 blk9 | Bootstrap | → level 10 | 3 | boot | 10 |
| 191 | Stage4 blk9 | AppReLU | degree 27 | 10 | -5 | 5 |
| 192 | Stage4 blk9 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 193 | Stage4 blk9 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 194 | Stage4 blk9 | Bootstrap | → level 10 | 3 | boot | 10 |
| 195 | Stage4 blk9 | AppReLU | degree 27 | 10 | -5 | 5 |
| 196 | Head | AvgPool | avg_pool(PP_POOL) | 5 | -1 | 4 |
| 197 | Head | FC | fc(64→10) | 4 | -1 | 3 |

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
