#pragma once
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "openfhe.h"

using namespace lbcrypto;
using namespace std;
using namespace std::chrono;

using Ptxt = Plaintext;
using Ctxt = Ciphertext<DCRTPoly>;

namespace utils {

inline auto start_time() { return steady_clock::now(); }

inline void print_duration(time_point<steady_clock, nanoseconds> t0, const string& label) {
    auto ms = duration_cast<milliseconds>(steady_clock::now() - t0);
    auto s = duration_cast<seconds>(ms); ms -= duration_cast<milliseconds>(s);
    auto m = duration_cast<minutes>(s);  s  -= duration_cast<seconds>(m);
    if (m.count() < 1)
        cout << "[" << label << "] " << s.count() << "." << ms.count() << "s\n";
    else
        cout << "[" << label << "] " << m.count() << "m" << s.count() << "s\n";
}

// 读取逗号分隔的浮点数文件
inline vector<double> read_bin(const string& path, double scale = 1.0) {
    vector<double> v;
    ifstream f(path);
    if (!f) { cerr << "无法打开: " << path << "\n"; return v; }
    string line, tok;
    while (getline(f, line)) {
        istringstream ss(line);
        while (getline(ss, tok, ',')) {
            try { v.push_back(stod(tok) * scale); } catch (...) {}
        }
    }
    return v;
}

inline int get_relu_depth(int deg) {
    switch (deg) {
        case 15: return 4;
        case 27: return 5;
        case 59: return 6;
        case 119: return 7;
        default: cerr << "无效 ReLU 度数: " << deg << "\n"; exit(1);
    }
}

} // namespace utils
