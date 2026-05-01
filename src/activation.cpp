#include "activation.h"

// AppReLU：使用 OpenFHE 的 EvalChebyshevFunction 实现近似 ReLU
// scale 是明文训练时得到的放缩系数，使值域在 [-1,1] 内
Ctxt app_relu(const FHEContext& ctx, const Ctxt& in, double scale) {
    return ctx.cc->EvalChebyshevFunction(
        [scale](double x) -> double { return x < 0 ? 0.0 : x / scale; },
        in, -1.0, 1.0, 27);
}
