#pragma once
#include "fhe_context.h"
#include "packing.h"

// Algorithm 6：AvgPool + 重排索引
Ctxt avg_pool(const FHEContext& ctx, const Ctxt& in, const PackParams& pp);
