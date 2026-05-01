#pragma once
#include "fhe_context.h"

// 全连接层：对角线方法（论文 Section 7.2）
// in：AvgPool 输出的一维密文（ci 个有效值顺序排列）
// weight_path：权重文件路径（已预处理为对角线格式）
Ctxt fc_layer(const FHEContext& ctx, const Ctxt& in,
              const string& weight_path, int in_dim, int out_dim);
