#pragma once
#include "fhe_context.h"
#include "packing.h"

// ResNet 推理入口
// n 控制每个 stage 的 block 数：ResNet-20=3, 32=5, 44=7, 56=9, 110=18
int resnet_infer(FHEContext& ctx, const Ctxt& input,
                 int n, const string& weights_dir, const string& keys_dir);
