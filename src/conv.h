#pragma once
#include "fhe_context.h"
#include "packing.h"
#include "weight_loader.h"

// Algorithm 7：MultParConvBN
// 权重文件已预处理为 ParMultWgt 格式，直接读取编码为 Ptxt
Ctxt mult_par_conv_bn(const FHEContext& ctx, const Ctxt& in,
                      const PackParams& pp,
                      const vector<Ptxt>& weights,  // fh*fw*q 个权重明文
                      const Ptxt& bn_bias);          // 1/B*(i''-c''*m'')
