# ResNet-20 乘法深度分析

## CKKS 参数

| 参数 | 值 |
|------|----|
| levels_before | 10 |
| bootstrap_depth | 18 |
| circuit_depth | 28 |
| 输入层级 | circuit_depth - 4 = 24 |
| 自举后层级 | 10 |
| n (block 数) | 3 |

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
| 24 | Stage3 blk1 main | ConvBN3_1a | conv(PP_CONV3A) | 5 | -2 | 3 |
| 25 | Stage3 blk1 main | Bootstrap | → level 10 | 3 | boot | 10 |
| 26 | Stage3 blk1 main | AppReLU | degree 27 | 10 | -5 | 5 |
| 27 | Stage3 blk1 main | ConvBN3_1b | conv(PP_CONV3) | 5 | -2 | 3 |
| 28 | Stage3 blk1 skip | ConvBN3_s1 | conv(PP_CONV3A) | 5 | -2 | 3 |
| 29 | Stage3 blk1 skip | DownSamp1 | downsamp(PP_CONV3A) | 3 | -1 | 2 **← min** |
| 30 | Stage3 blk1 | Add(skip) | min(3,2) | 3 | 0 | 2 **← min** |
| 31 | Stage3 blk1 | Bootstrap | → level 10 | 2 | boot | 10 |
| 32 | Stage3 blk1 | AppReLU | degree 27 | 10 | -5 | 5 |
| 33 | Stage3 blk2 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 34 | Stage3 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 35 | Stage3 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 36 | Stage3 blk2 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 37 | Stage3 blk2 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 38 | Stage3 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 39 | Stage3 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 40 | Stage3 blk3 | ConvBN3a | conv(PP_CONV3) | 5 | -2 | 3 |
| 41 | Stage3 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 42 | Stage3 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 43 | Stage3 blk3 | ConvBN3b | conv(PP_CONV3) | 5 | -2 | 3 |
| 44 | Stage3 blk3 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 45 | Stage3 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 46 | Stage3 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 47 | Stage4 blk1 main | ConvBN4_1a | conv(PP_CONV4A) | 5 | -2 | 3 |
| 48 | Stage4 blk1 main | Bootstrap | → level 10 | 3 | boot | 10 |
| 49 | Stage4 blk1 main | AppReLU | degree 27 | 10 | -5 | 5 |
| 50 | Stage4 blk1 main | ConvBN4_1b | conv(PP_CONV4) | 5 | -2 | 3 |
| 51 | Stage4 blk1 skip | ConvBN4_s2 | conv(PP_CONV4A) | 5 | -2 | 3 |
| 52 | Stage4 blk1 skip | DownSamp2 | downsamp(PP_CONV4A) | 3 | -1 | 2 **← min** |
| 53 | Stage4 blk1 | Add(skip) | min(3,2) | 3 | 0 | 2 **← min** |
| 54 | Stage4 blk1 | Bootstrap | → level 10 | 2 | boot | 10 |
| 55 | Stage4 blk1 | AppReLU | degree 27 | 10 | -5 | 5 |
| 56 | Stage4 blk2 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 57 | Stage4 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 58 | Stage4 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 59 | Stage4 blk2 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 60 | Stage4 blk2 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 61 | Stage4 blk2 | Bootstrap | → level 10 | 3 | boot | 10 |
| 62 | Stage4 blk2 | AppReLU | degree 27 | 10 | -5 | 5 |
| 63 | Stage4 blk3 | ConvBN4a | conv(PP_CONV4) | 5 | -2 | 3 |
| 64 | Stage4 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 65 | Stage4 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 66 | Stage4 blk3 | ConvBN4b | conv(PP_CONV4) | 5 | -2 | 3 |
| 67 | Stage4 blk3 | Add(skip) | min(3,5) | 3 | 0 | 3 |
| 68 | Stage4 blk3 | Bootstrap | → level 10 | 3 | boot | 10 |
| 69 | Stage4 blk3 | AppReLU | degree 27 | 10 | -5 | 5 |
| 70 | Head | AvgPool | avg_pool(PP_POOL) | 5 | -1 | 4 |
| 71 | Head | FC | fc(64→10) | 4 | -1 | 3 |

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
