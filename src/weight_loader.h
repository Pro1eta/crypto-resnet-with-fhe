#pragma once
#include "fhe_context.h"
#include "packing.h"

// 权重文件已预处理为 ParMultWgt 格式，直接读取编码为 Ptxt
// 每层权重文件命名约定：weights/<layer_name>-w<i3>-k<i1*fw+i2>.bin
// BN 偏置文件：weights/<layer_name>-bn_bias.bin

struct LayerWeights {
    vector<Ptxt> w;   // fh*fw*q 个权重明文
    Ptxt bn_bias;     // 1/B*(i'' - c''*m'')
};

LayerWeights load_conv_weights(const FHEContext& ctx, const string& layer_name,
                                const PackParams& pp, int level,
                                const string& weights_dir = "weights");
