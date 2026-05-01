#pragma once
#include "fhe_context.h"

// AppReLU：minimax 近似多项式组合，精度 α=13，度数 {15,15,27}（论文 Section 7.1）
// scale：进入前的缩放系数（明文训练时得到），用于将值域映射到 [-1,1]
Ctxt app_relu(const FHEContext& ctx, const Ctxt& in, double scale);
