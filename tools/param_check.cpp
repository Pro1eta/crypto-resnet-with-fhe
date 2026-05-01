// 参数校验工具：计算三种 ResNet 每层参数，检查 po → pi 衔接
// 逻辑与 src/packing.cpp 完全等价
// 编译：g++ -std=c++17 -O2 param_check.cpp -o param_check && ./param_check

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

const int NT = 32768;  // 2^15

static int get_ti(int ci, int ki) { int d = ki * ki; return (ci + d - 1) / d; }
static int get_to(int co, int ko) { int d = ko * ko; return (co + d - 1) / d; }

static int get_pi(int ki, int hi, int wi, int ti) {
    int denom = ki * ki * hi * wi * ti;
    if (denom == 0) return 1;
    return 1 << static_cast<int>(std::log2(NT / denom));
}

static int get_po(int ko, int ho, int wo, int to) {
    int denom = ko * ko * ho * wo * to;
    if (denom == 0) return 1;
    return 1 << static_cast<int>(std::log2(NT / denom));
}

static int get_q(int co, int pi) { return (co + pi - 1) / pi; }

struct Layer {
    std::string name;
    std::string type;
    int s;           // 卷积步长
    int ci, co, ki, ko, ti, to, pi, po, q;
    int hi, wi, ho, wo;
    int boot_slots;
};

static void add_convbn(std::vector<Layer>& layers, const std::string& name,
                       int s, int ci, int co, int ki, int ko,
                       int hi, int wi, int ho, int wo) {
    Layer l;
    l.name = name;
    l.type = "ConvBN";
    l.s = s;
    l.ci = ci; l.co = co; l.ki = ki; l.ko = ko;
    l.hi = hi; l.wi = wi; l.ho = ho; l.wo = wo;
    l.ti = get_ti(ci, ki);
    l.to = get_to(co, ko);
    l.pi = get_pi(ki, hi, wi, l.ti);
    l.po = get_po(ko, ho, wo, l.to);
    l.q = get_q(co, l.pi);
    l.boot_slots = 0;
    layers.push_back(l);
}

static void add_boot(std::vector<Layer>& layers, int slots, int ki) {
    Layer l;
    l.name = "Boot(" + std::to_string(slots) + ")";
    l.type = "Boot";
    l.s = l.ci = l.co = l.ti = l.to = l.pi = l.po = l.q = l.hi = l.wi = l.ho = l.wo = 0;
    l.ki = l.ko = ki;
    l.boot_slots = slots;
    layers.push_back(l);
}

static void add_relu(std::vector<Layer>& layers, int ki) {
    Layer l;
    l.name = "AppReLU";
    l.type = "AppReLU";
    l.s = l.ci = l.co = l.ti = l.to = l.pi = l.po = l.q = l.hi = l.wi = l.ho = l.wo = 0;
    l.ki = l.ko = ki;
    l.boot_slots = 0;
    layers.push_back(l);
}

static void add_avgpool(std::vector<Layer>& layers, int ki, int ci, int hi, int wi) {
    Layer l;
    l.name = "AvgPool";
    l.type = "AvgPool";
    l.s = 0;
    l.ci = ci; l.co = ci;
    l.hi = hi; l.wi = wi; l.ho = 1; l.wo = 1;
    l.ki = l.ko = ki;
    l.ti = get_ti(ci, ki); l.to = l.ti;
    l.pi = l.po = l.q = 0;
    l.boot_slots = 0;
    layers.push_back(l);
}

static void add_fc(std::vector<Layer>& layers, int in_dim, int out_dim) {
    Layer l;
    l.name = "FC";
    l.type = "FC layer";
    l.s = l.hi = l.wi = l.ho = l.wo = l.ki = l.ko = l.ti = l.to = l.pi = l.po = l.q = 0;
    l.ci = in_dim; l.co = out_dim;
    l.boot_slots = 0;
    layers.push_back(l);
}

static std::vector<Layer> build_architecture(int n) {
    std::vector<Layer> layers;

    // Stage 1: ConvBN1 (3→16, 32×32, s=1, ki=1)
    add_convbn(layers, "ConvBN1",      1,  3, 16, 1, 1,  32, 32, 32, 32);
    add_relu(layers, 1);

    // Stage 2: n blocks, 16→16, 32×32, s=1, ki=1
    for (int b = 1; b <= n; b++) {
        add_convbn(layers, "ConvBN2_" + std::to_string(b) + "a",
                   1, 16, 16, 1, 1,  32, 32, 32, 32);
        add_boot(layers, 16384, 1);
        add_relu(layers, 1);
        add_convbn(layers, "ConvBN2_" + std::to_string(b) + "b",
                   1, 16, 16, 1, 1,  32, 32, 32, 32);
        add_boot(layers, 16384, 1);
        add_relu(layers, 1);
    }

    // Stage 3 block 1: downsample (16→32, s=2, ki=1→2)
    add_convbn(layers, "ConvBN3_1a",       2, 16, 32, 1, 2,  32, 32, 16, 16);
    add_boot(layers, 8192, 2);
    add_relu(layers, 2);
    add_convbn(layers, "ConvBN3_1b",       1, 32, 32, 2, 2,  16, 16, 16, 16);
    add_convbn(layers, "ConvBN_s1 (skip)", 2, 16, 32, 1, 2,  32, 32, 16, 16);
    add_boot(layers, 8192, 2);
    add_relu(layers, 2);

    // Stage 3 blocks 2..n: identity (32→32, s=1, ki=2)
    for (int b = 2; b <= n; b++) {
        add_convbn(layers, "ConvBN3_" + std::to_string(b) + "a",
                   1, 32, 32, 2, 2,  16, 16, 16, 16);
        add_boot(layers, 8192, 2);
        add_relu(layers, 2);
        add_convbn(layers, "ConvBN3_" + std::to_string(b) + "b",
                   1, 32, 32, 2, 2,  16, 16, 16, 16);
        add_boot(layers, 8192, 2);
        add_relu(layers, 2);
    }

    // Stage 4 block 1: downsample (32→64, s=2, ki=2→4)
    add_convbn(layers, "ConvBN4_1a",       2, 32, 64, 2, 4,  16, 16, 8, 8);
    add_boot(layers, 4096, 4);
    add_relu(layers, 4);
    add_convbn(layers, "ConvBN4_1b",       1, 64, 64, 4, 4,  8, 8, 8, 8);
    add_convbn(layers, "ConvBN_s2 (skip)", 2, 32, 64, 2, 4,  16, 16, 8, 8);
    add_boot(layers, 4096, 4);
    add_relu(layers, 4);

    // Stage 4 blocks 2..n: identity (64→64, s=1, ki=4)
    for (int b = 2; b <= n; b++) {
        add_convbn(layers, "ConvBN4_" + std::to_string(b) + "a",
                   1, 64, 64, 4, 4,  8, 8, 8, 8);
        add_boot(layers, 4096, 4);
        add_relu(layers, 4);
        add_convbn(layers, "ConvBN4_" + std::to_string(b) + "b",
                   1, 64, 64, 4, 4,  8, 8, 8, 8);
        add_boot(layers, 4096, 4);
        add_relu(layers, 4);
    }

    // Head: AvgPool + FC
    add_avgpool(layers, 4, 64, 8, 8);
    add_fc(layers, 64, 10);

    return layers;
}

// 判断 ConvBN 在残差块中的角色
static bool is_conv_a(const std::string& name) {
    if (name.find("skip") != std::string::npos) return false;
    return name.size() >= 2 && name[name.size()-1] == 'a';
}
static bool is_conv_b(const std::string& name) {
    return name.size() >= 2 && name[name.size()-1] == 'b';
}
static bool is_skip(const std::string& name) {
    return name.find("skip") != std::string::npos;
}

// 将层参数写入 Markdown 表格
static void write_md_table(std::ofstream& out,
                           const std::vector<Layer>& layers,
                           bool show_all) {
    // 表头
    out << "| # | 层名称 | 类型 | s | ci | co | hi | wi | ho | wo | ki | ko | ti | to | pi | po | q | Boot | 检查 |\n";
    out << "|---|--------|------|---|---|----|----|----|----|----|----|----|----|----|----|----|----|-----|------|\n";

    // DAG 状态机
    enum BlkState { OUTSIDE, AFTER_A, AFTER_B };
    BlkState st = OUTSIDE;
    int entry_po = -1, entry_ko = -1;
    int a_po = -1;
    int b_po = -1, b_ko = -1;

    int idx = 0;
    auto v = [](int x) -> std::string { return x == 0 ? "-" : std::to_string(x); };

    for (const auto& l : layers) {
        if (!show_all && l.type == "AppReLU") continue;

        idx++;
        std::string check;

        if (l.type == "ConvBN") {
            std::string tag;
            if (!is_conv_a(l.name) && !is_conv_b(l.name) && !is_skip(l.name)) {
                tag = "入口";
                check = "✓";
                entry_po = l.po; entry_ko = l.ko;
                st = OUTSIDE;
            } else if (is_conv_a(l.name)) {
                if (st == AFTER_B) { entry_po = b_po; entry_ko = b_ko; }
                tag = "a";
                bool ok = (entry_po < 0) || (l.pi == entry_po);
                check = ok ? "✓" : ("✗ pi≠" + std::to_string(entry_po));
                a_po = l.po;
                st = AFTER_A;
            } else if (is_conv_b(l.name)) {
                tag = "b";
                bool ok = (l.pi == a_po);
                check = ok ? "✓" : ("✗ pi≠" + std::to_string(a_po));
                b_po = l.po; b_ko = l.ko;
                st = AFTER_B;
            } else if (is_skip(l.name)) {
                tag = "skip";
                bool m_pi = (l.pi == entry_po);
                bool m_po = (l.po == b_po && l.ko == b_ko);
                std::string pi_s = m_pi ? "pi✓" : ("pi✗" + std::to_string(entry_po));
                std::string po_s = m_po ? "po✓" : ("po✗" + std::to_string(b_po));
                check = pi_s + " " + po_s;
                entry_po = b_po; entry_ko = b_ko;
                st = OUTSIDE;
            }
            check = "ConvBN [" + tag + "] " + check;
        } else if (l.type == "Boot") {
            check = "Boot";
        } else if (l.type == "AppReLU") {
            check = "AppReLU";
        } else if (l.type == "AvgPool") {
            check = "AvgPool";
        } else if (l.type == "FC layer") {
            check = "FC";
        }

        out << "| " << idx
            << " | " << l.name
            << " | " << l.type
            << " | " << v(l.s)
            << " | " << v(l.ci)
            << " | " << v(l.co)
            << " | " << v(l.hi)
            << " | " << v(l.wi)
            << " | " << v(l.ho)
            << " | " << v(l.wo)
            << " | " << v(l.ki)
            << " | " << v(l.ko)
            << " | " << v(l.ti)
            << " | " << v(l.to)
            << " | " << v(l.pi)
            << " | " << v(l.po)
            << " | " << v(l.q)
            << " | " << v(l.boot_slots)
            << " | " << check
            << " |\n";
    }
    out << "\n";
}

// 打印 ConvBN 衔接摘要到 Markdown
static void write_md_summary(std::ofstream& out,
                              const std::vector<Layer>& layers,
                              const std::string& title) {
    out << "### " << title << " — ConvBN po→pi 衔接\n\n";
    out << "| 层 | ki | pi | ko | po | 上游来源 | 状态 |\n";
    out << "|-----|----|----|----|----|----------|------|\n";

    enum BlkState { OUTSIDE, AFTER_A, AFTER_B };
    BlkState st = OUTSIDE;
    int entry_po = -1, entry_ko = -1;
    int a_po = -1, a_ko = -1;
    int b_po = -1, b_ko = -1;

    for (const auto& l : layers) {
        if (l.type != "ConvBN") continue;

        if (!is_conv_a(l.name) && !is_conv_b(l.name) && !is_skip(l.name)) {
            out << "| " << l.name
                << " | " << l.ki << " | " << l.pi
                << " | " << l.ko << " | " << l.po
                << " | 初始卷积 | — |\n";
            entry_po = l.po; entry_ko = l.ko;
            continue;
        }

        if (is_conv_a(l.name)) {
            if (st == AFTER_B) { entry_po = b_po; entry_ko = b_ko; }
            bool ok = (entry_po < 0) || (l.pi == entry_po);
            out << "| " << l.name
                << " | " << l.ki << " | " << l.pi
                << " | " << l.ko << " | " << l.po
                << " | 块入口 po=" << entry_po
                << " | " << (ok ? "✓" : "✗") << " |\n";
            a_po = l.po; a_ko = l.ko;
            st = AFTER_A;
            continue;
        }

        if (is_conv_b(l.name)) {
            bool ok = (l.pi == a_po);
            out << "| " << l.name
                << " | " << l.ki << " | " << l.pi
                << " | " << l.ko << " | " << l.po
                << " | ConvBN_a po=" << a_po
                << " | " << (ok ? "✓" : "✗") << " |\n";
            b_po = l.po; b_ko = l.ko;
            st = AFTER_B;
            continue;
        }

        if (is_skip(l.name)) {
            bool m_pi = (l.pi == entry_po);
            bool m_po = (l.po == b_po && l.ko == b_ko);
            std::string status;
            if (m_pi && m_po) status = "pi✓ po✓";
            else if (!m_pi && !m_po) status = "pi✗ po✗";
            else if (!m_pi) status = "pi✗ po✓";
            else status = "pi✓ po✗";

            out << "| " << l.name
                << " | " << l.ki << " | " << l.pi
                << " | " << l.ko << " | " << l.po
                << " | 块入口 po=" << entry_po
                << " | **" << status << "** |\n";
            entry_po = b_po; entry_ko = b_ko;
            st = OUTSIDE;
            continue;
        }
    }
    out << "\n";
}

int main() {
    std::ofstream out("tools/params.md");
    if (!out) {
        std::cerr << "无法写入 tools/params.md\n";
        return 1;
    }

    out << "# ResNet 参数总览\n\n";
    out << "nt = " << NT << " (2^15 槽位)\n\n";
    out << "各参数含义：`ci/co` 通道数, `hi/wi/ho/wo` 特征图尺寸, `ki/ko` 间隙, `s` 步长, ";
    out << "`ti/to` 复用通道组数, `pi/po` 并行副本数, `q` 并行通道组数, `Boot` 自举槽位数\n\n";

    for (int n : {5, 9, 18}) {
        std::string arch;
        if (n == 5) arch = "ResNet-32";
        else if (n == 9) arch = "ResNet-56";
        else arch = "ResNet-110";

        out << "## " << arch << " (n=" << n << " blocks/stage)\n\n";
        auto layers = build_architecture(n);

        out << "### 完整层序\n\n";
        write_md_table(out, layers, true);

        write_md_summary(out, layers, arch);
    }

    out.close();
    std::cout << "已生成 tools/params.md\n";
    return 0;
}
