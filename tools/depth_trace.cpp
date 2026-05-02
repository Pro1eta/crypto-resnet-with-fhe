// 计算 FHE ResNet 推理管线中每一步的乘法深度消耗
// 对照 src/ 中各算法的 ct-pt 乘法链自动推导
// 编译：g++ -std=c++17 -O2 tools/depth_trace.cpp -o tools/depth_trace
// 运行：./tools/depth_trace [n]    n=3(ResNet20) / 5(32) / 9(56) / 18(110)

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ========== 与 fhe_context.cpp 对齐的参数 ==========

static const int LEVELS_BEFORE = 10;
// GetBootstrapDepth({4,4}, SPARSE_TERNARY)
//   = GetModDepthInternal(SPARSE) + 4 + 4
//   = (GetDepthByDegree(44) + R_SPARSE) + 8
//   = (7 + 3) + 8 = 18
static const int BOOT_DEPTH = 18;
static const int CIRCUIT_DEPTH = LEVELS_BEFORE + BOOT_DEPTH; // = 28

// ========== 与 resnet.cpp 对齐的 PackParams ==========

struct PP {
    std::string label;
    int hi, wi, ci, ho, wo, co, ki, ko, ti, to_, pi, po, q, s;
    int fh = 3, fw = 3;
};

static PP C1  = {"PP_CONV1",  32,32, 3, 32,32,16, 1,1, 3,16, 8,2, 2,1};
static PP C2  = {"PP_CONV2",  32,32,16, 32,32,16, 1,1,16,16, 2,2, 8,1};
static PP C3A = {"PP_CONV3A", 32,32,16, 16,16,32, 1,2,16, 8, 2,4,16,2};
static PP C3  = {"PP_CONV3",  16,16,32, 16,16,32, 2,2, 8, 8, 4,4, 8,1};
static PP C4A = {"PP_CONV4A", 16,16,32,  8, 8,64, 2,4, 8, 4, 4,8,16,2};
static PP C4  = {"PP_CONV4",   8, 8,64,  8, 8,64, 4,4, 4, 4, 8,8, 8,1};
static PP PL  = {"PP_POOL",    8, 8,64,  1, 1,64, 4,4, 4, 4, 0,0, 0,1};

// ========== 从算法逻辑推导的深度代价 ==========

// mult_par_conv_bn (conv.cpp Algorithm 7):
//   乘法链: ctx.mul(rot, weight)  ← 第1次 ct-pt 乘，深度 +1
//           sum_slots(...)        ← 仅旋转+加法，深度 +0
//           ctx.mul(rotated, sel) ← 第2次 ct-pt 乘，深度 +1
//           log2(po) 旋转累加     ← 仅旋转+加法，深度 +0
//           EvalAdd(out, bn_bias) ← ct-pt 加法，深度 +0
//   总计: 2
int conv_bn_depth() { return 2; }

// downsamp (downsample.cpp Algorithm 5):
//   乘法链: ctx.mul(in, sel)      ← 1次 ct-pt 乘，深度 +1
//           旋转+加法+旋转+累加    ← 仅旋转+加法，深度 +0
//   总计: 1
int downsamp_depth() { return 1; }

// avg_pool (avgpool.cpp Algorithm 6):
//   乘法链: log2(wi)+log2(hi) 旋转累加 ← 仅旋转+加法，深度 +0
//           ctx.mul(rot_a, sel)          ← 1次 ct-pt 乘，深度 +1
//   总计: 1
int avgpool_depth() { return 1; }

// fc_layer (fc.cpp 对角线方法):
//   乘法链: ctx.mul(rot_in, diag) ← 1次 ct-pt 乘，深度 +1
//           所有 d 的结果通过加法累加 ← 同层级，深度 +0
//   总计: 1
int fc_depth() { return 1; }

// app_relu (activation.cpp):
//   EvalChebyshevFunction(degree=27, a=-1, b=1)
//   OpenFHE PS 参数: ComputeDegreesPS(27) → k=4, m=3
//   深度 = GetDepthByDegree(27) - 1 = 6 - 1 = 5
//   (-1 因为 a=-1, b=1 无需线性变换)
int relu_depth(int degree = 27) {
    // 对照 OpenFHE GetDepthByDegree 表
    // degree [14,27] → 6，a=-1,b=1 省 1 级 → 5
    switch (degree) {
        case 15: return 5;
        case 27: return 5;
        case 59: return 6;
        case 119: return 7;
        default: return (int)std::ceil(std::log2(degree + 1));
    }
}

// ========== 层级追踪器 ==========

struct Step {
    std::string stage;
    std::string operation;
    std::string detail;
    int input_level;
    int cost;
    int output_level;
};

class Tracer {
public:
    int level;
    int min_level = 999;
    std::string min_location;
    std::vector<Step> steps;

    Tracer(int start) : level(start) {}

    void op(const std::string& stage, const std::string& name,
            const std::string& detail, int cost) {
        int in = level;
        level -= cost;
        steps.push_back({stage, name, detail, in, cost, level});
        if (level < min_level) {
            min_level = level;
            min_location = stage + " / " + name;
        }
    }

    void boot(const std::string& stage) {
        int in = level;
        level = LEVELS_BEFORE;
        steps.push_back({stage, "Bootstrap", "→ level " + std::to_string(LEVELS_BEFORE), in, 0, level});
    }

    int save() const { return level; }
    void restore(int l) { level = l; }

    void join(const std::string& stage, int skip_level) {
        int joined = std::min(level, skip_level);
        steps.push_back({stage, "Add(skip)", "min(" + std::to_string(level) + "," + std::to_string(skip_level) + ")", level, 0, joined});
        level = joined;
        if (level < min_level) {
            min_level = level;
            min_location = stage + " / Add(skip)";
        }
    }
};

// ========== 模拟推理管线 (对照 resnet.cpp) ==========

void trace_resnet(Tracer& t, int n) {
    int D_conv = conv_bn_depth();
    int D_relu = relu_depth(27);
    int D_ds   = downsamp_depth();
    int D_pool = avgpool_depth();
    int D_fc   = fc_depth();

    // ConvBN1 → AppReLU（无 Bootstrap）
    t.op("Stage1", "ConvBN1", "conv(" + C1.label + ") = " + std::to_string(D_conv) + " levels", D_conv);
    t.op("Stage1", "AppReLU", "degree 27 = " + std::to_string(D_relu) + " levels", D_relu);

    // Stage2：n 个 identity block
    for (int b = 1; b <= n; b++) {
        std::string tag = "Stage2 blk" + std::to_string(b);
        int skip = t.save();
        t.op(tag, "ConvBN2a", "conv(" + C2.label + ")", D_conv);
        t.boot(tag); t.op(tag, "AppReLU", "degree 27", D_relu);
        t.op(tag, "ConvBN2b", "conv(" + C2.label + ")", D_conv);
        t.join(tag, skip);
        t.boot(tag); t.op(tag, "AppReLU", "degree 27", D_relu);
    }

    // Stage3 block1（转换块，含 skip conv + downsamp）
    {
        std::string tag = "Stage3 blk1";
        int skip_level = t.save();

        // 主路径
        t.op(tag + " main", "ConvBN3_1a", "conv(" + C3A.label + ")", D_conv);
        t.boot(tag + " main"); t.op(tag + " main", "AppReLU", "degree 27", D_relu);
        t.op(tag + " main", "ConvBN3_1b", "conv(" + C3.label + ")", D_conv);
        int main_level = t.save();

        // Skip 路径
        t.restore(skip_level);
        t.op(tag + " skip", "ConvBN3_s1", "conv(" + C3A.label + ")", D_conv);
        t.op(tag + " skip", "DownSamp1", "downsamp(" + C3A.label + ")", D_ds);
        int skip_out = t.save();

        // 合并
        t.restore(main_level);
        t.join(tag, skip_out);
        t.boot(tag); t.op(tag, "AppReLU", "degree 27", D_relu);
    }

    // Stage3 identity blocks
    for (int b = 2; b <= n; b++) {
        std::string tag = "Stage3 blk" + std::to_string(b);
        int skip = t.save();
        t.op(tag, "ConvBN3a", "conv(" + C3.label + ")", D_conv);
        t.boot(tag); t.op(tag, "AppReLU", "degree 27", D_relu);
        t.op(tag, "ConvBN3b", "conv(" + C3.label + ")", D_conv);
        t.join(tag, skip);
        t.boot(tag); t.op(tag, "AppReLU", "degree 27", D_relu);
    }

    // Stage4 block1（转换块，含 skip conv + downsamp）
    {
        std::string tag = "Stage4 blk1";
        int skip_level = t.save();

        // 主路径
        t.op(tag + " main", "ConvBN4_1a", "conv(" + C4A.label + ")", D_conv);
        t.boot(tag + " main"); t.op(tag + " main", "AppReLU", "degree 27", D_relu);
        t.op(tag + " main", "ConvBN4_1b", "conv(" + C4.label + ")", D_conv);
        int main_level = t.save();

        // Skip 路径
        t.restore(skip_level);
        t.op(tag + " skip", "ConvBN4_s2", "conv(" + C4A.label + ")", D_conv);
        t.op(tag + " skip", "DownSamp2", "downsamp(" + C4A.label + ")", D_ds);
        int skip_out = t.save();

        // 合并
        t.restore(main_level);
        t.join(tag, skip_out);
        t.boot(tag); t.op(tag, "AppReLU", "degree 27", D_relu);
    }

    // Stage4 identity blocks
    for (int b = 2; b <= n; b++) {
        std::string tag = "Stage4 blk" + std::to_string(b);
        int skip = t.save();
        t.op(tag, "ConvBN4a", "conv(" + C4.label + ")", D_conv);
        t.boot(tag); t.op(tag, "AppReLU", "degree 27", D_relu);
        t.op(tag, "ConvBN4b", "conv(" + C4.label + ")", D_conv);
        t.join(tag, skip);
        t.boot(tag); t.op(tag, "AppReLU", "degree 27", D_relu);
    }

    // AvgPool → FC
    t.op("Head", "AvgPool", "avg_pool(" + PL.label + ")", D_pool);
    t.op("Head", "FC", "fc(64→10)", D_fc);
}

// ========== 输出 Markdown ==========

void write_markdown(const Tracer& t, int n, const std::string& path) {
    std::ofstream f(path);
    int resnet_id = 6*n + 2;

    f << "# ResNet-" << resnet_id << " 乘法深度分析\n\n";

    f << "## CKKS 参数\n\n";
    f << "| 参数 | 值 |\n|------|----|\n";
    f << "| levels_before | " << LEVELS_BEFORE << " |\n";
    f << "| bootstrap_depth | " << BOOT_DEPTH << " |\n";
    f << "| circuit_depth | " << CIRCUIT_DEPTH << " |\n";
    f << "| 输入层级 | circuit_depth - 4 = " << CIRCUIT_DEPTH - 4 << " |\n";
    f << "| 自举后层级 | " << LEVELS_BEFORE << " |\n";
    f << "| n (block 数) | " << n << " |\n\n";

    f << "## 各操作深度代价（从算法推导）\n\n";
    f << "| 操作 | 深度代价 | 推导依据 |\n";
    f << "|------|---------|----------|\n";
    f << "| ConvBN (Algorithm 7) | 2 | ① `ctx.mul(rot, weight)` +1；② `ctx.mul(rotated, sel)` +1 |\n";
    f << "| DownSamp (Algorithm 5) | 1 | ① `ctx.mul(in, sel)` +1 |\n";
    f << "| AvgPool (Algorithm 6) | 1 | ① `ctx.mul(rot_a, sel)` +1 |\n";
    f << "| FC (对角线方法) | 1 | ① `ctx.mul(rot_in, diag)` +1 |\n";
    f << "| AppReLU (degree 27) | " << relu_depth(27)
      << " | Chebyshev PS: k=4, m=3, GetDepthByDegree(27)=6, a=-1,b=1 省 1 |\n";
    f << "| Bootstrap | → " << LEVELS_BEFORE
      << " | 恢复至 levels_before |\n";
    f << "| Add (skip) | 0 | min(主路径, skip 路径) |\n";
    f << "| Rot / SumSlots | 0 | 仅槽位旋转+加法 |\n\n";

    f << "## 逐层深度追踪\n\n";
    f << "| # | Stage | 操作 | 说明 | 输入层级 | 消耗 | 输出层级 |\n";
    f << "|---|-------|------|------|---------|------|----------|\n";

    for (size_t i = 0; i < t.steps.size(); i++) {
        auto& s = t.steps[i];
        f << "| " << i + 1
          << " | " << s.stage
          << " | " << s.operation
          << " | " << s.detail
          << " | " << s.input_level
          << " | " << (s.cost > 0 ? "-" + std::to_string(s.cost) : (s.operation == "Bootstrap" ? "boot" : "0"))
          << " | " << s.output_level;
        if (s.output_level == t.min_level)
            f << " **← min**";
        f << " |\n";
    }

    f << "\n## 关键路径分析\n\n";
    f << "- **最低层级**: " << t.min_level
      << "（位于 " << t.min_location << "）\n";
    if (t.min_level >= 1)
        f << "- **状态**: 安全（余量 " << t.min_level << " 级）\n";
    else if (t.min_level == 0)
        f << "- **状态**: 边界（层级为 0，任何额外消耗将导致 DropLastElement）\n";
    else
        f << "- **状态**: **不安全**（层级 < 0，将触发 DropLastElement 错误）\n";

    f << "- **自举间最大消耗**: AppReLU(" << relu_depth(27)
      << ") + ConvBN(2) = " << relu_depth(27) + 2 << " 级\n";
    f << "- **转换块 skip 路径**: ConvBN(2) + DownSamp(1) = 3 级，"
      << "入口层级 " << LEVELS_BEFORE - relu_depth(27)
      << "，出口层级 " << LEVELS_BEFORE - relu_depth(27) - 3 << "\n";

    f << "\n## 自举间层级预算\n\n";
    f << "```\n";
    f << "Bootstrap 输出: level " << LEVELS_BEFORE << "\n";
    f << "  ├─ AppReLU:   -" << relu_depth(27) << " → level " << LEVELS_BEFORE - relu_depth(27) << "\n";
    f << "  ├─ ConvBN:    -2 → level " << LEVELS_BEFORE - relu_depth(27) - 2 << "  (进入下一次 Bootstrap)\n";
    f << "  │\n";
    f << "  └─ 转换块 skip 路径（最深）:\n";
    f << "     ├─ ConvBN:    -2 → level " << LEVELS_BEFORE - relu_depth(27) - 2 << "\n";
    f << "     └─ DownSamp:  -1 → level " << LEVELS_BEFORE - relu_depth(27) - 3 << "  ← 全局最低\n";
    f << "```\n";

    f.close();
}

int main(int argc, char* argv[]) {
    int n = 3;
    if (argc >= 2) n = std::atoi(argv[1]);

    int start_level = CIRCUIT_DEPTH - 4;

    Tracer t(start_level);
    trace_resnet(t, n);

    std::string path = "tools/depth_trace_resnet" + std::to_string(6*n+2) + ".md";
    write_markdown(t, n, path);

    std::cout << "ResNet-" << 6*n+2 << " 深度追踪完成\n";
    std::cout << "  circuit_depth = " << CIRCUIT_DEPTH << "\n";
    std::cout << "  起始层级      = " << start_level << "\n";
    std::cout << "  最低层级      = " << t.min_level
              << " (" << t.min_location << ")\n";
    std::cout << "  输出 → " << path << "\n";
    return 0;
}
