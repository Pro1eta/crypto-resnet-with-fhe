# ResNet-32 乘法深度分析

## CKKS 参数

| 参数 | 值 |
|------|----|
| levels_before | 10 |
| bootstrap_depth | 18 |
| circuit_depth | 28 |
| 输入层级 | circuit_depth - 4 = 24 |
| 自举后层级 | 10 |
| n (block 数) | 5 |

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
| 38 | Stage3 blk1 main | ConvBN3_1a | conv(PP_CONV3A) | 5 | -2 | 3 |
| 39 | Stage3 blk1 main | Bootstrap | → level 10 | 3 | boot | 10 |
| 40 | Stage3 blk1 main | AppReLU | degree 27 | 10 | -5 | 5 |
| 41 | Stage3 blk1 main | ConvBN3_1b | conv(PP_CONV3) | 5 | -2 | 3 |
| 42 | Stage3 blk1 skip | ConvBN3_s1 | conv(PP_CONV3A) | 5 | -2 | 3 |
| 43 | Stage3 blk1 skip | DownSamp1 | downsamp(PP_CONV3A) | 3 | -1 | 2 **← min** |
| 44 | Stage3 blk1 | Add(skip) | min(3,2) | 3 | 0 | 2 **← min** |
| 45 | Stage3 blk1 | Bootstrap | → level 10 | 2 | boot | 10 |
| 46 | Stage3 blk1 | AppReLU | degree 27 | 10 | -5 | 5 |
| 47 | Stage3 blk2 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 48 | Stage3 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 49 | Stage3 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 50 | Stage3 blk2 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 51 | Stage3 blk2 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 52 | Stage3 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 53 | Stage3 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 54 | Stage3 blk3 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 55 | Stage3 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 56 | Stage3 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 57 | Stage3 blk3 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 58 | Stage3 blk3 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 59 | Stage3 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 60 | Stage3 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 61 | Stage3 blk4 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 62 | Stage3 blk4 | Bootstrap | → level 10 | 3 | boot | 10 |
| 63 | Stage3 blk4 | AppReLU | degree 27 | 10 | -5 | 5 |
| 64 | Stage3 blk4 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 65 | Stage3 blk4 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 66 | Stage3 blk4 | Bootstrap | → level 10 | 3 | boot | 10 |
| 67 | Stage3 blk4 | AppReLU | degree 27 | 10 | -5 | 5 |
| 68 | Stage3 blk5 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 69 | Stage3 blk5 | Bootstrap | → level 10 | 3 | boot | 10 |
| 70 | Stage3 blk5 | AppReLU | degree 27 | 10 | -5 | 5 |
| 71 | Stage3 blk5 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 72 | Stage3 blk5 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 73 | Stage3 blk5 | Bootstrap | → level 10 | 3 | boot | 10 |
| 74 | Stage3 blk5 | AppReLU | degree 27 | 10 | -5 | 5 |
| 75 | Stage4 blk1 main | ConvBN4_1a | conv(PP_CONV4A) | 5 | -2 | 3 |
| 76 | Stage4 blk1 main | Bootstrap | → level 10 | 3 | boot | 10 |
| 77 | Stage4 blk1 main | AppReLU | degree 27 | 10 | -5 | 5 |
| 78 | Stage4 blk1 main | ConvBN4_1b | conv(PP_CONV4) | 5 | -2 | 3 |
| 79 | Stage4 blk1 skip | ConvBN4_s2 | conv(PP_CONV4A) | 5 | -2 | 3 |
| 80 | Stage4 blk1 skip | DownSamp2 | downsamp(PP_CONV4A) | 3 | -1 | 2 **← min** |
| 81 | Stage4 blk1 | Add(skip) | min(3,2) | 3 | 0 | 2 **← min** |
| 82 | Stage4 blk1 | Bootstrap | → level 10 | 2 | boot | 10 |
| 83 | Stage4 blk1 | AppReLU | degree 27 | 10 | -5 | 5 |
| 84 | Stage4 blk2 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 85 | Stage4 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 86 | Stage4 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 87 | Stage4 blk2 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 88 | Stage4 blk2 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 89 | Stage4 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 90 | Stage4 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 91 | Stage4 blk3 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 92 | Stage4 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 93 | Stage4 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 94 | Stage4 blk3 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 95 | Stage4 blk3 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 96 | Stage4 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 97 | Stage4 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 98 | Stage4 blk4 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 99 | Stage4 blk4 | Bootstrap | → level 10 | 3 | boot | 10 |
| 100 | Stage4 blk4 | AppReLU | degree 27 | 10 | -5 | 5 |
| 101 | Stage4 blk4 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 102 | Stage4 blk4 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 103 | Stage4 blk4 | Bootstrap | → level 10 | 3 | boot | 10 |
| 104 | Stage4 blk4 | AppReLU | degree 27 | 10 | -5 | 5 |
| 105 | Stage4 blk5 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 106 | Stage4 blk5 | Bootstrap | → level 10 | 3 | boot | 10 |
| 107 | Stage4 blk5 | AppReLU | degree 27 | 10 | -5 | 5 |
| 108 | Stage4 blk5 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 109 | Stage4 blk5 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 110 | Stage4 blk5 | Bootstrap | → level 10 | 3 | boot | 10 |
| 111 | Stage4 blk5 | AppReLU | degree 27 | 10 | -5 | 5 |
| 112 | Head | AvgPool | avg_pool(PP_POOL) | 5 | -1 | 4 |
| 113 | Head | FC | fc(64→10) | 4 | -1 | 3 |

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
