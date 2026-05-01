#pragma once
#include "fhe_context.h"
#include "packing.h"

// Algorithm 5：DownSamp
// 在 stage 边界调用（ki=1→2 或 ki=2→4），将 stride=2 后的稀疏数据重新紧凑打包
Ctxt downsamp(const FHEContext& ctx, const Ctxt& in, const PackParams& pp);
