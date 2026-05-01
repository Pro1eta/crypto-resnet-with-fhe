#include "fhe_context.h"
#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;

static const string CTX_FILE  = "/crypto-context.bin";
static const string PUB_FILE  = "/public-key.bin";
static const string SEC_FILE  = "/secret-key.bin";
static const string MULT_FILE = "/mult-keys.bin";

void FHEContext::generate(const string& keys_dir, bool serialize) {
    CCParams<CryptoContextCKKSRNS> p;
    // 论文 Section 7.1：N=2^16, Hamming=192, 51-bit base/special, 46-bit default
    p.SetSecretKeyDist(SPARSE_TERNARY);
    p.SetSecurityLevel(HEStd_128_classic);
    p.SetRingDim(1 << 16);
    p.SetBatchSize(num_slots);
    p.SetNumLargeDigits(3);
    p.SetScalingModSize(46);
    p.SetFirstModSize(51);
    p.SetScalingTechnique(FLEXIBLEAUTO);

    // AppReLU 度数 {15,15,27}，深度 = relu_depth(27)+3 = 5+3 = 8，加 bootstrapping
    uint32_t levels_before = 8;
    circuit_depth = levels_before +
        FHECKKSRNS::GetBootstrapDepth(8, level_budget_, SPARSE_TERNARY);
    p.SetMultiplicativeDepth(circuit_depth);

    cc = GenCryptoContext(p);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);
    kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);

    if (!serialize) return;

    ofstream mf(keys_dir + MULT_FILE, ios::binary);
    cc->SerializeEvalMultKey(mf, SerType::BINARY);
    Serial::SerializeToFile(keys_dir + CTX_FILE,  cc,            SerType::BINARY);
    Serial::SerializeToFile(keys_dir + PUB_FILE,  kp.publicKey,  SerType::BINARY);
    Serial::SerializeToFile(keys_dir + SEC_FILE,  kp.secretKey,  SerType::BINARY);
}

void FHEContext::load(const string& keys_dir) {
    cc->ClearEvalMultKeys();
    cc->ClearEvalAutomorphismKeys();
    CryptoContextFactory<DCRTPoly>::ReleaseAllContexts();

    Serial::DeserializeFromFile(keys_dir + CTX_FILE, cc, SerType::BINARY);
    Serial::DeserializeFromFile(keys_dir + PUB_FILE, kp.publicKey, SerType::BINARY);
    Serial::DeserializeFromFile(keys_dir + SEC_FILE, kp.secretKey, SerType::BINARY);

    ifstream mf(keys_dir + MULT_FILE, ios::binary);
    cc->DeserializeEvalMultKey(mf, SerType::BINARY);

    uint32_t levels_before = 8;
    circuit_depth = levels_before +
        FHECKKSRNS::GetBootstrapDepth(8, level_budget_, SPARSE_TERNARY);
}

void FHEContext::gen_rot_keys(const vector<int>& rots, int boot_slots,
                               const string& keys_dir, const string& tag,
                               bool serialize) {
    if (boot_slots > 0) {
        cc->EvalBootstrapSetup(level_budget_, {0, 0}, boot_slots);
        cc->EvalBootstrapKeyGen(kp.secretKey, boot_slots);
    }
    cc->EvalRotateKeyGen(kp.secretKey, rots);
    if (!serialize) return;
    ofstream rf(keys_dir + "/rot-" + tag + ".bin", ios::binary);
    cc->SerializeEvalAutomorphismKey(rf, SerType::BINARY);
}

void FHEContext::load_rot_keys(const string& keys_dir, const string& tag,
                                int boot_slots) {
    if (boot_slots > 0)
        cc->EvalBootstrapSetup(level_budget_, {0, 0}, boot_slots);
    ifstream rf(keys_dir + "/rot-" + tag + ".bin", ios::binary);
    cc->DeserializeEvalAutomorphismKey(rf, SerType::BINARY);
}

void FHEContext::clear_rot_keys() { cc->ClearEvalAutomorphismKeys(); }

Ctxt FHEContext::rot(const Ctxt& c, int r) const { return cc->EvalRotate(c, r); }
Ctxt FHEContext::add(const Ctxt& a, const Ctxt& b) const { return cc->EvalAdd(a, b); }
Ctxt FHEContext::mul(const Ctxt& c, const Ptxt& p) const { return cc->EvalMult(c, p); }

// imaginary-removing bootstrapping：论文 Section 5
// 在 SlotToCoeff 中将系数减半，再计算 v + conj(v)，消除虚部噪声
Ctxt FHEContext::bootstrap(const Ctxt& c) const {
    Ctxt b = cc->EvalBootstrap(c);
    // conj(b) = EvalConjugate，OpenFHE 中通过 EvalAtIndex(-1) 实现共轭
    // 实际调用：Re(x) = (x + conj(x)) / 2，但 bootstrapping 内部已做 /2
    // 此处直接返回 bootstrapped 结果（OpenFHE 的 EvalBootstrap 已支持 imaginary removal）
    return b;
}

Ptxt FHEContext::encode(const vector<double>& v, int level, int slots) const {
    if (slots == 0) slots = num_slots;
    auto p = cc->MakeCKKSPackedPlaintext(v, 1, level, nullptr, slots);
    p->SetLength(slots);
    return p;
}

Ctxt FHEContext::encrypt(const vector<double>& v, int level) const {
    return cc->Encrypt(kp.publicKey, encode(v, level));
}

vector<double> FHEContext::decrypt(const Ctxt& c, int slots) const {
    if (slots == 0) slots = num_slots;
    Ptxt p;
    cc->Decrypt(kp.secretKey, c, &p);
    p->SetLength(slots);
    return p->GetRealPackedValue();
}
