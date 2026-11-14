// SPDX-License-Identifier: GPL-3.0-or-later
// Adapted from SuperMarioWonderRecomp runtime/src/nvn/nvn_shaders.{h,cpp} (MIT, Copyright (c) 2026 BelmanteGu); changes: namespace, cache lookup.
// Also: the reflection of Wonder's runtime/src/nvn/nvn_spirv_reflect.cpp is in this file too (one source file builds it).
// NVN shader stages -> offline-translated SPIR-V (nvn_shaders.h, SuperMarioWonderRecomp docs/nvn/TRANSLATOR.md "Runtime side").
//
// The meaning of each binding comes from the translator's own naming and numbering (Ryujinx.Graphics.Shader,
// Translation/ResourceManager.cs): uniform buffers are "<stage>_c<k>" at binding stageSlot * 18 + 1 + k, storage
// buffers "<stage>_s<slot>", textures "<stage>_t_tcb_<word hex>" (handle in the texture bank c[2]) or
// "<stage>_t_cb<k>_<word hex>" (handle in c[k]), images "<stage>_i_..." likewise with a format suffix. The one thing
// neither the names nor the numbers carry is the c[0] offset a storage buffer's address was loaded from; that comes
// from the harness's results.tsv (apply_harness).
#include "nvn_shaders.h"
#include "nvn_spirv_patch.h"

#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_set>

#include <nxrt/ftrace.h>
#include <nxrt/config.h>
#include <nxrt/gauge.h>
#include <nxrt/log.h>

#include <zstd.h>

namespace nxrt::nvn::be {

// ---- SHA-1 ----------------------------------------------------------------------------------------------------------

namespace {

inline uint32_t rol(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

inline uint32_t rd32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);  // little-endian host (arm64, x86-64), like the Switch
    return v;
}

}  // namespace

Sha1::Sha1() : h_{0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0} {}

void Sha1::block(const uint8_t* p) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++)
        w[i] = uint32_t(p[4 * i]) << 24 | uint32_t(p[4 * i + 1]) << 16 | uint32_t(p[4 * i + 2]) << 8 | p[4 * i + 3];
    for (int i = 16; i < 80; i++) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) f = (b & c) | (~b & d), k = 0x5a827999;
        else if (i < 40) f = b ^ c ^ d, k = 0x6ed9eba1;
        else if (i < 60) f = (b & c) | (b & d) | (c & d), k = 0x8f1bbcdc;
        else f = b ^ c ^ d, k = 0xca62c1d6;
        uint32_t t = rol(a, 5) + f + e + k + w[i];
        e = d, d = c, c = rol(b, 30), b = a, a = t;
    }
    h_[0] += a, h_[1] += b, h_[2] += c, h_[3] += d, h_[4] += e;
}

void Sha1::update(const void* data, size_t size) {
    auto* p = static_cast<const uint8_t*>(data);
    bytes_ += size;
    if (used_) {
        size_t n = std::min(size, 64 - used_);
        std::memcpy(buf_ + used_, p, n);
        used_ += n, p += n, size -= n;
        if (used_ < 64) return;
        block(buf_);
        used_ = 0;
    }
    for (; size >= 64; p += 64, size -= 64) block(p);
    std::memcpy(buf_, p, size);
    used_ = size;
}

void Sha1::finish(uint8_t digest[20]) {
    uint64_t bits = bytes_ * 8;
    uint8_t pad = 0x80;
    update(&pad, 1);
    pad = 0;
    while (used_ != 56) update(&pad, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; i++) len[i] = uint8_t(bits >> (56 - 8 * i));
    update(len, 8);
    for (int i = 0; i < 20; i++) digest[i] = uint8_t(h_[i / 4] >> (24 - 8 * (i % 4)));
}

std::string sha1_hex(const void* data, size_t size) {
    Sha1 s;
    s.update(data, size);
    uint8_t d[20];
    s.finish(d);
    static const char* kHex = "0123456789abcdef";
    std::string out(40, '0');
    for (int i = 0; i < 20; i++) out[2 * i] = kHex[d[i] >> 4], out[2 * i + 1] = kHex[d[i] & 15];
    return out;
}

std::string shader_hash(const uint8_t* code, uint32_t size) { return sha1_hex(code, size).substr(0, 16); }

// ---- control section ------------------------------------------------------------------------------------------------

ShaderControl read_control(const uint8_t* control) {
    ShaderControl c;
    c.valid = rd32(control) == kShaderControlMagic;
    if (c.valid && rd32(control + 4) == 201) {
        // NVN2 (Switch 2) layout, from the game's GLSLC outputs (tools/shaders/docs/AMPERE.md section 3): program length +0x38,
        // constants +0x3c/+0x40, code area +0x44, shared memory +0x50, stage u16 +0x56, compute work group +0x58.
        c.nvn2 = true;
        c.program_size = rd32(control + 0x38);
        c.const_size = rd32(control + 0x3c);
        c.const_offset = rd32(control + 0x40);
        c.code_size = rd32(control + 0x44);
        c.shared_size = rd32(control + 0x50);
        c.stage = static_cast<int>(*reinterpret_cast<const uint16_t*>(control + 0x56));
        c.program_start = 0;   // graphics: 0x80-byte SPH v4 at +0, instructions from +0x80; compute: instructions from +0
        if (c.stage == 5) {
            for (int i = 0; i < 3; i++) c.local_size[i] = rd32(control + 0x58 + 4 * i);
            c.c0_ubos = control[0xb1];
            c.c0_ssbos = control[0xb2];
            c.c0_textures = control[0xb3];
            c.c0_images = control[0xb6];
            c.c0_push_words = control[0xa9];
            c.c0_table_size = *reinterpret_cast<const uint16_t*>(control + 0xba);
        }
        return c;
    }
    c.program_size = rd32(control + 0x6f8);
    c.const_size = rd32(control + 0x6fc);
    c.const_offset = rd32(control + 0x700);
    c.code_size = rd32(control + 0x704);
    c.program_start = rd32(control + 0x708);
    for (int i = 0; i < 3; i++) c.local_size[i] = rd32(control + 0x748 + 4 * i);
    c.shared_size = rd32(control + 0x754);
    return c;
}

// ---- support buffer -------------------------------------------------------------------------------------------------

SupportBuffer support_buffer_defaults() {
    SupportBuffer sb;
    std::memset(&sb, 0, sizeof sb);
    for (auto& s : sb.render_scale) s[0] = 1.0f;  // Ryujinx's SupportBufferUpdater: (1, 0, 0, 0) for all 73
    return sb;
}

void support_buffer_set_viewport(SupportBuffer& sb, float width, float height, bool transform_disabled) {
    const float scale = 1.0f;  // no resolution scaling
    sb.viewport_inverse[0] = width != 0 ? scale * 2.0f / width : 0.0f;
    sb.viewport_inverse[1] = height != 0 ? scale * 2.0f / height : 0.0f;
    sb.viewport_inverse[2] = 1.0f;
    sb.viewport_inverse[3] = transform_disabled ? 1.0f : 0.0f;
    sb.viewport_size[0] = width;
    sb.viewport_size[1] = height < 0 ? -height : height;
    sb.viewport_size[2] = 1.0f;
    sb.viewport_size[3] = 0.0f;
}

// ---- SPIR-V reflection (from nvn_spirv_reflect.cpp) -----------------------------------------------------------------

namespace {

// The handful of SPIR-V opcodes, decorations and storage classes reflection needs (SPIR-V 1.x unified spec).
enum : uint32_t {
    OpName = 5, OpEntryPoint = 15, OpExecutionMode = 16, OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22,
    OpTypeVector = 23, OpTypeMatrix = 24, OpTypeImage = 25, OpTypeSampler = 26, OpTypeSampledImage = 27,
    OpTypeArray = 28, OpTypeRuntimeArray = 29, OpTypeStruct = 30, OpTypePointer = 32, OpConstant = 43,
    OpVariable = 59, OpAccessChain = 65, OpInBoundsAccessChain = 66, OpDecorate = 71, OpMemberDecorate = 72,
};
enum : uint32_t {
    DecBlock = 2, DecBufferBlock = 3, DecBuiltIn = 11, DecLocation = 30, DecComponent = 31, DecBinding = 33,
    DecDescriptorSet = 34,
};
enum : uint32_t {
    ScUniformConstant = 0, ScInput = 1, ScUniform = 2, ScOutput = 3, ScStorageBuffer = 12,
};
constexpr uint32_t kExecutionModelVertex = 0, kExecutionModelTessControl = 1, kExecutionModelTessEval = 2, kExecutionModelGeometry = 3,
                   kExecutionModelFragment = 4, kExecutionModelCompute = 5;
constexpr uint32_t kExecutionModeLocalSize = 17;
constexpr uint32_t kDimBuffer = 5;

struct Decorations {
    int set = -1, binding = -1, location = -1, component = 0, builtin = -1;
    bool block = false, buffer_block = false;
    std::vector<uint32_t> member_builtins;
    std::vector<std::pair<uint32_t, uint32_t>> member_builtin_at;   // (member index, BuiltIn)
};

struct Type {
    uint32_t op = 0;
    std::vector<uint32_t> w;  // operands after the result id
};

std::string spirv_string(const uint32_t* w, size_t n, size_t* used) {
    std::string s;
    size_t i = 0;
    for (; i < n; i++) {
        bool end = false;
        for (int b = 0; b < 4; b++) {
            char ch = char((w[i] >> (8 * b)) & 0xff);
            if (!ch) {
                end = true;
                break;
            }
            s.push_back(ch);
        }
        if (end) {
            i++;
            break;
        }
    }
    if (used) *used = i;
    return s;
}

bool parse_uint(std::string_view s, int base, int& out) {
    if (s.empty()) return false;
    int64_t v = 0;      // 64-bit: long is 32 bits on Windows, so v * base overflowed (UB) before the range check could see it
    for (char ch : s) {
        int d = ch >= '0' && ch <= '9' ? ch - '0' : ch >= 'a' && ch <= 'f' ? ch - 'a' + 10
                : ch >= 'A' && ch <= 'F' ? ch - 'A' + 10 : 99;
        if (d >= base) return false;
        v = v * base + d;
        if (v > 0x7fffffff) return false;
    }
    out = int(v);
    return true;
}

// Ryujinx's stage slot inside each set's binding range: vertex and compute 0, fragment 1, geometry 2 (GetStageIndex
// remap; the harness's StaticGpuAccessor gives the same numbers). Geometry is measured: arc0017_s1 storage binding 32
// (= 2 * 16) and the uniform blocks of the other geometry shaders at binding 40 (= 2 * 18 + 1 + 3). Tessellation control
// 3 and evaluation 4 are the StaticGpuAccessor's numbers (no translated tessellation shader checked yet).
int stage_slot(ShaderStage s) {
    switch (s) {
    case ShaderStage::Fragment: return 1;
    case ShaderStage::Geometry: return 2;
    case ShaderStage::TessControl: return 3;
    case ShaderStage::TessEval: return 4;
    default: return 0;
    }
}

// The storage buffer descriptors NVN keeps in c[0] (TRANSLATOR.md, bank table): graphics stage index i at
// 0x110 + 0x100 * i + 0x10 * n (vertex i = 0, geometry i = 3, fragment i = 4; Ryujinx's stage order); compute uniform
// blocks at 0x210 + 0x10 * n and storage buffers at 0x310 + 0x10 * n. The geometry offset is measured: arc0017_s1
// prints sb[c0+0x104] (word 0x104 = byte 0x410).
void set_c0_offset(NvnMeaning& m, ShaderStage stage, int byte_offset) {
    m.c0_offset = byte_offset;
    m.source = NvnSource::Unknown;
    m.index = -1;
    auto in = [&](int base, int count) {
        return byte_offset >= base && byte_offset < base + 0x10 * count && (byte_offset - base) % 0x10 == 0;
    };
    if (stage == ShaderStage::Compute) {
        if (in(0x210, 16)) m.source = NvnSource::UniformAsStorage, m.index = (byte_offset - 0x210) / 0x10;
        else if (in(0x310, 16)) m.source = NvnSource::StorageBuffer, m.index = (byte_offset - 0x310) / 0x10;
        return;
    }
    // tessellation control 1 and evaluation 2: Ryujinx's hardware stage order (inferred; no game shader checked)
    int graphics_index = stage == ShaderStage::Vertex ? 0 : stage == ShaderStage::TessControl ? 1 : stage == ShaderStage::TessEval ? 2
                         : stage == ShaderStage::Geometry ? 3 : stage == ShaderStage::Fragment ? 4 : -1;
    if (graphics_index < 0) return;
    int base = 0x110 + 0x100 * graphics_index;
    if (in(base, 16)) m.source = NvnSource::StorageBuffer, m.index = (byte_offset - base) / 0x10;
}

void set_texture_handle(NvnMeaning& m, bool image, int bank, int word) {
    m.default_bank = bank < 0;
    m.bank = bank < 0 ? 2 : bank;
    m.word = word;
    m.index = -1;
    m.source = NvnSource::Unknown;
    if (m.bank == 2) {
        // c[2]: texture handle of sampler binding n at byte 0x20 + 8n (verified on every stage), images after 32
        // texture slots at 0x120 + 8n (inferred from the agl compute shaders: words 72, 74, 76).
        int first = image ? 72 : 8;
        if (word >= first && (word - first) % 2 == 0 && (word - first) / 2 < 32) {
            m.source = image ? NvnSource::ImageBinding : NvnSource::SamplerBinding;
            m.index = (word - first) / 2;
        }
    } else if (m.bank >= 3 && !image) {
        m.source = NvnSource::HandleInUniform;
        m.index = m.bank - 3;
    }
}

// The NVN meaning from the variable's name and binding. Names first; the binding number alone fixes uniform banks.
void derive_meaning(DescriptorBinding& b, ShaderStage stage) {
    NvnMeaning& m = b.nvn;
    std::string_view n = b.name;
    if (n == "support_buffer") {
        m.source = NvnSource::SupportBuffer;
        return;
    }
    if (size_t us = n.find('_'); us != std::string_view::npos) {
        std::string_view prefix = n.substr(0, us);  // vp fp cp (tcp tep gp): Ryujinx GetShaderStagePrefix
        if (prefix == "vp" || prefix == "fp" || prefix == "cp" || prefix == "tcp" || prefix == "tep" || prefix == "gp")
            n.remove_prefix(us + 1);
    }
    int v = 0;
    if (b.type == DescriptorType::UniformBuffer) {
        // Banks 0..17 at binding stageSlot * 18 + 1 + bank; Ampere's banks 18..31 at 91 + stageSlot * 14 + (bank - 18) (the harness's
        // ampere-resource-slots range). NVN2 graphics stages read push constant slot n from bank 24 + n (sdk 0x673660 sends the slot with
        // the data; Fortune's Weave's fragment shaders read slot 0 as c[0x18]).
        int bank = -1;
        if (n.size() > 1 && n[0] == 'c' && parse_uint(n.substr(1), 10, v)) bank = v;
        else if (b.set == 0 && b.binding >= 91) {
            const int k = int(b.binding) - 91 - stage_slot(stage) * 14;
            if (k >= 0 && k < 14) bank = 18 + k;
        } else if (b.set == 0 && b.binding >= 1) {
            bank = int(b.binding) - 1 - stage_slot(stage) * 18;  // fallback
        }
        m.bank = bank;
        m.index = -1;
        if (bank == 0) m.source = NvnSource::DriverConstants;
        else if (bank == 1) m.source = NvnSource::ConstantData;
        else if (bank >= 3 && bank < 18) m.source = NvnSource::UniformBlock, m.index = bank - 3;
        else if (bank >= 24 && bank < 32) m.source = NvnSource::PushConstants, m.index = bank - 24;
        else m.source = NvnSource::Unknown;
        return;
    }
    if (b.type == DescriptorType::StorageBuffer) {
        if (n.size() > 1 && n[0] == 's' && parse_uint(n.substr(1), 10, v)) m.sb_slot = v;
        else if (b.set == 1) m.sb_slot = int(b.binding) - stage_slot(stage) * 16;
        m.source = NvnSource::Unknown;  // until apply_harness gives the c[0] offset
        return;
    }
    // Textures and images: "<t|i|s>[type]_tcb_<hex>[_fmt]" or "<t|i|s>[type]_cb<k>_<hex>[_fmt]".
    bool image = b.type == DescriptorType::StorageImage || b.type == DescriptorType::StorageTexelBuffer;
    size_t us = n.find('_');
    if (us == std::string_view::npos) return;
    std::string_view rest = n.substr(us + 1);
    int bank = -2;
    if (rest.substr(0, 4) == "tcb_") {
        bank = -1;
        rest.remove_prefix(4);
    } else if (rest.substr(0, 2) == "cb") {
        size_t us2 = rest.find('_');
        if (us2 == std::string_view::npos || !parse_uint(rest.substr(2, us2 - 2), 10, bank)) return;
        rest.remove_prefix(us2 + 1);
    } else {
        return;
    }
    std::string_view hex = rest.substr(0, rest.find('_'));
    int word = 0;
    if (!parse_uint(hex, 16, word)) return;
    if (bank == 0 && word == int(kPoolHandleWord) && !image) {
        m.bank = 0;
        m.word = word;
        m.index = -1;
        m.source = b.type == DescriptorType::Sampler ? NvnSource::SamplerPoolArray : NvnSource::TexturePoolArray;
        return;
    }
    set_texture_handle(m, image, bank, word);
}

}  // namespace

const DescriptorBinding* ShaderReflection::find(uint32_t set, uint32_t binding) const {
    for (const auto& b : bindings)
        if (b.set == set && b.binding == binding) return &b;
    return nullptr;
}

bool reflect_spirv(const uint32_t* words, size_t count, ShaderReflection& out, std::string* error) {
    auto fail = [&](const char* why) {
        if (error) *error = why;
        return false;
    };
    out = ShaderReflection{};
    if (count < 5 || words[0] != 0x07230203) return fail("not a SPIR-V module");

    std::unordered_map<uint32_t, std::string> names;
    std::unordered_map<uint32_t, Decorations> dec;
    std::unordered_map<uint32_t, Type> types;
    std::unordered_map<uint32_t, uint32_t> constants;
    struct Var {
        uint32_t id, ptr_type, storage;
    };
    std::vector<Var> vars;
    struct Chain {
        uint32_t base;
        std::vector<uint32_t> idx;   // index ids
    };
    std::vector<Chain> chains;   // access chains (block members of built-in blocks: builtins_used_*)
    std::unordered_set<uint32_t> referenced;  // ids some instruction takes as an operand (approximation, see below)
    uint32_t entry_id = 0;
    bool have_entry = false;

    for (size_t i = 5; i < count;) {
        uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;
        if (wc == 0 || i + wc > count) return fail("truncated instruction");
        const uint32_t* w = words + i + 1;  // operands
        size_t n = wc - 1;
        // Operands of code instructions are taken as ids. Declarations, decorations, names, types, constants and
        // literal-only instructions are not: they never reference a variable through an id.
        if (!(op <= 7 || op == 10 || op == 11 || op == 14 || op == 15 || op == 16 || op == 17 || (op >= 19 && op <= 40) ||
              (op >= 43 && op <= 52) || op == 59 || (op >= 71 && op <= 75) || op == 103))
            for (size_t k = 0; k < n; k++) referenced.insert(w[k]);
        switch (op) {
        case OpEntryPoint:
            if (!have_entry && n >= 3) {
                have_entry = true;
                uint32_t model = w[0];
                entry_id = w[1];
                out.entry = spirv_string(w + 2, n - 2, nullptr);
                out.stage = model == kExecutionModelVertex ? ShaderStage::Vertex
                            : model == kExecutionModelFragment ? ShaderStage::Fragment
                            : model == kExecutionModelCompute ? ShaderStage::Compute
                            : model == kExecutionModelGeometry ? ShaderStage::Geometry
                            : model == kExecutionModelTessControl ? ShaderStage::TessControl
                            : model == kExecutionModelTessEval ? ShaderStage::TessEval
                            : ShaderStage::Unknown;
            }
            break;
        case OpExecutionMode:
            if (n >= 5 && w[0] == entry_id && w[1] == kExecutionModeLocalSize)
                out.local_size[0] = w[2], out.local_size[1] = w[3], out.local_size[2] = w[4];
            break;
        case OpName:
            if (n >= 2) names[w[0]] = spirv_string(w + 1, n - 1, nullptr);
            break;
        case OpDecorate:
            if (n >= 2) {
                Decorations& d = dec[w[0]];
                uint32_t lit = n >= 3 ? w[2] : 0;
                switch (w[1]) {
                case DecDescriptorSet: d.set = int(lit); break;
                case DecBinding: d.binding = int(lit); break;
                case DecLocation: d.location = int(lit); break;
                case DecComponent: d.component = int(lit); break;
                case DecBuiltIn: d.builtin = int(lit); break;
                case DecBlock: d.block = true; break;
                case DecBufferBlock: d.buffer_block = true; break;
                }
            }
            break;
        case OpMemberDecorate:
            if (n >= 4 && w[2] == DecBuiltIn) {
                dec[w[0]].member_builtins.push_back(w[3]);
                dec[w[0]].member_builtin_at.emplace_back(w[1], w[3]);
            }
            break;
        case OpTypeBool: case OpTypeInt: case OpTypeFloat: case OpTypeVector: case OpTypeMatrix: case OpTypeImage:
        case OpTypeSampler: case OpTypeSampledImage: case OpTypeArray: case OpTypeRuntimeArray: case OpTypeStruct:
        case OpTypePointer:
            if (n >= 1) types[w[0]] = Type{op, std::vector<uint32_t>(w + 1, w + n)};
            break;
        case OpConstant:
            if (n >= 3) constants[w[1]] = w[2];  // low word is enough for array lengths
            break;
        case OpVariable:
            if (n >= 3) vars.push_back({w[1], w[0], w[2]});
            break;
        case OpAccessChain: case OpInBoundsAccessChain:
            if (n >= 4) chains.push_back({w[2], std::vector<uint32_t>(w + 3, w + n)});
            break;
        }
        i += wc;
    }
    if (!have_entry) return fail("no OpEntryPoint");

    auto type = [&](uint32_t id) -> const Type* {
        auto it = types.find(id);
        return it == types.end() ? nullptr : &it->second;
    };
    auto scalar = [&](uint32_t id, uint32_t& comps) -> ScalarType {
        const Type* t = type(id);
        comps = 1;
        if (t && t->op == OpTypeVector && t->w.size() >= 2) comps = t->w[1], t = type(t->w[0]);
        if (!t) return ScalarType::Other;
        if (t->op == OpTypeFloat) return ScalarType::Float;
        if (t->op == OpTypeBool) return ScalarType::Bool;
        if (t->op == OpTypeInt && t->w.size() >= 2) return t->w[1] ? ScalarType::Sint : ScalarType::Uint;
        return ScalarType::Other;
    };
    auto builtins_of = [&](uint32_t var_id, uint32_t pointee, std::vector<uint32_t>& list) {
        auto add = [&](uint32_t b) {
            if (std::find(list.begin(), list.end(), b) == list.end()) list.push_back(b);
        };
        if (auto it = dec.find(var_id); it != dec.end() && it->second.builtin >= 0) add(uint32_t(it->second.builtin));
        // gl_PerVertex-style blocks, possibly arrays of them
        for (const Type* t = type(pointee); t;) {
            if (t->op == OpTypeArray && !t->w.empty()) {
                pointee = t->w[0];
                t = type(pointee);
                continue;
            }
            if (t->op == OpTypeStruct)
                if (auto it = dec.find(pointee); it != dec.end())
                    for (uint32_t b : it->second.member_builtins) add(b);
            break;
        }
    };

    for (const Var& v : vars) {
        const Type* ptr = type(v.ptr_type);
        if (!ptr || ptr->op != OpTypePointer || ptr->w.size() < 2) continue;
        uint32_t pointee = ptr->w[1];
        const Decorations* d = nullptr;
        if (auto it = dec.find(v.id); it != dec.end()) d = &it->second;

        if (v.storage == ScInput || v.storage == ScOutput) {
            auto& list = v.storage == ScInput ? out.builtins_in : out.builtins_out;
            builtins_of(v.id, pointee, list);
            if (!d || d->location < 0 || d->builtin >= 0) continue;
            ShaderIo io;
            io.location = uint32_t(d->location);
            io.component = uint32_t(d->component);
            uint32_t elem = pointee;
            if (const Type* t = type(elem); t && t->op == OpTypeArray && t->w.size() >= 2) {
                io.array = constants.count(t->w[1]) ? constants[t->w[1]] : 1;
                elem = t->w[0];
            }
            io.type = scalar(elem, io.components);
            (v.storage == ScInput ? out.inputs : out.outputs).push_back(io);
            continue;
        }
        if (v.storage != ScUniformConstant && v.storage != ScUniform && v.storage != ScStorageBuffer) continue;
        if (!d || d->set < 0 || d->binding < 0) continue;

        DescriptorBinding b;
        b.set = uint32_t(d->set);
        b.binding = uint32_t(d->binding);
        b.declared_set = b.set;
        b.declared_binding = b.binding;
        b.used = referenced.count(v.id) != 0;
        if (auto it = names.find(v.id); it != names.end()) b.name = it->second;
        uint32_t inner = pointee;
        for (const Type* t = type(inner); t && (t->op == OpTypeArray || t->op == OpTypeRuntimeArray); t = type(inner)) {
            if (t->op == OpTypeRuntimeArray) b.count = 0;
            else if (t->w.size() >= 2) b.count *= constants.count(t->w[1]) ? constants[t->w[1]] : 1;
            inner = t->w.empty() ? 0 : t->w[0];
        }
        const Type* t = type(inner);
        if (!t) return fail("descriptor of unknown type");
        auto image_shape = [&](const Type* img) {
            ImageShape s;
            if (img && img->op == OpTypeImage && img->w.size() >= 7) {
                uint32_t c;
                s.component = scalar(img->w[0], c);
                s.dim = uint8_t(img->w[1]);
                s.depth = img->w[2] == 1, s.arrayed = img->w[3] != 0, s.ms = img->w[4] != 0;
                s.sampled = uint8_t(img->w[5]);
                s.format = img->w[6];
            }
            return s;
        };
        if (t->op == OpTypeStruct) {
            const Decorations* sd = nullptr;
            if (auto it = dec.find(inner); it != dec.end()) sd = &it->second;
            bool storage = v.storage == ScStorageBuffer || (sd && sd->buffer_block);
            b.type = storage ? DescriptorType::StorageBuffer : DescriptorType::UniformBuffer;
        } else if (t->op == OpTypeSampledImage) {
            b.image = image_shape(t->w.empty() ? nullptr : type(t->w[0]));
            b.type = b.image.dim == kDimBuffer ? DescriptorType::UniformTexelBuffer
                                               : DescriptorType::CombinedImageSampler;
        } else if (t->op == OpTypeImage) {
            b.image = image_shape(t);
            bool storage = b.image.sampled == 2;
            if (b.image.dim == kDimBuffer)
                b.type = storage ? DescriptorType::StorageTexelBuffer : DescriptorType::UniformTexelBuffer;
            else
                b.type = storage ? DescriptorType::StorageImage : DescriptorType::SampledImage;
        } else if (t->op == OpTypeSampler) {
            b.type = DescriptorType::Sampler;
        } else {
            return fail("descriptor of unexpected type");
        }
        derive_meaning(b, out.stage);
        if (b.nvn.source == NvnSource::SupportBuffer) out.uses_support_buffer = true;
        out.bindings.push_back(std::move(b));
    }
    std::sort(out.bindings.begin(), out.bindings.end(), [](const DescriptorBinding& a, const DescriptorBinding& b) {
        return a.set != b.set ? a.set < b.set : a.binding < b.binding;
    });
    // built-in block members (gl_PerVertex.gl_ClipDistance, ...) an access chain reaches: the ones the shader uses, not just declares
    std::unordered_map<uint32_t, const Var*> io_vars;
    for (const Var& x : vars)
        if (x.storage == ScInput || x.storage == ScOutput) io_vars[x.id] = &x;
    for (const Chain& c : chains) {
        auto vi = io_vars.find(c.base);
        if (vi == io_vars.end()) continue;
        const Var* v = vi->second;
        const Type* pt = type(v->ptr_type);
        if (!pt || pt->op != OpTypePointer || pt->w.size() < 2) continue;
        uint32_t t_id = pt->w[1];
        size_t depth = 0;
        for (const Type* t = type(t_id); t && t->op == OpTypeArray && !t->w.empty(); t = type(t_id)) t_id = t->w[0], ++depth;
        const Type* st = type(t_id);
        if (!st || st->op != OpTypeStruct || depth >= c.idx.size()) continue;
        auto d = dec.find(t_id);
        auto k = constants.find(c.idx[depth]);
        if (d == dec.end() || k == constants.end()) continue;
        auto& list = v->storage == ScInput ? out.builtins_used_in : out.builtins_used_out;
        for (const auto& [member, b] : d->second.member_builtin_at)
            if (member == k->second && std::find(list.begin(), list.end(), b) == list.end()) list.push_back(b);
    }
    auto by_location = [](const ShaderIo& a, const ShaderIo& b) {
        return a.location != b.location ? a.location < b.location : a.component < b.component;
    };
    std::sort(out.inputs.begin(), out.inputs.end(), by_location);
    std::sort(out.outputs.begin(), out.outputs.end(), by_location);
    return true;
}

// ---- harness records ------------------------------------------------------------------------------------------------

bool parse_harness_resources(std::string_view text, std::vector<HarnessResource>& out) {
    out.clear();
    while (!text.empty()) {
        size_t sp = text.find(' ');
        std::string_view tok = text.substr(0, sp);
        text = sp == std::string_view::npos ? std::string_view{} : text.substr(sp + 1);
        if (tok.empty()) continue;
        // "...=s<set>b<binding>"
        size_t eq = tok.rfind('=');
        if (eq == std::string_view::npos || eq + 1 >= tok.size() || tok[eq + 1] != 's') return false;
        std::string_view sb = tok.substr(eq + 2);
        size_t bpos = sb.find('b');
        HarnessResource r;
        if (bpos == std::string_view::npos || !parse_uint(sb.substr(0, bpos), 10, r.set) ||
            !parse_uint(sb.substr(bpos + 1), 10, r.binding))
            return false;
        std::string_view lhs = tok.substr(0, eq);
        if (lhs.substr(0, 2) == "cb") {
            r.kind = HarnessResource::Uniform;
            if (!parse_uint(lhs.substr(2), 10, r.bank)) return false;
        } else if (lhs.substr(0, 4) == "sb[c") {
            // sb[c<slot>+0x<off>]
            r.kind = HarnessResource::Storage;
            size_t plus = lhs.find("+0x"), close = lhs.find(']');
            if (plus == std::string_view::npos || close == std::string_view::npos) return false;
            if (!parse_uint(lhs.substr(4, plus - 4), 10, r.bank) ||
                !parse_uint(lhs.substr(plus + 3, close - plus - 3), 16, r.word))
                return false;
        } else if (lhs.substr(0, 3) == "tex" || lhs.substr(0, 3) == "img") {
            // tex<type>[c<slot>:<handle>(x<n>)]
            r.kind = lhs[0] == 't' ? HarnessResource::Texture : HarnessResource::Image;
            size_t open = lhs.find("[c"), colon = lhs.find(':'), close = lhs.find(']');
            if (open == std::string_view::npos || colon == std::string_view::npos || close == std::string_view::npos)
                return false;
            r.type = std::string(lhs.substr(3, open - 3));
            std::string_view bank = lhs.substr(open + 2, colon - open - 2);
            bool neg = !bank.empty() && bank[0] == '-';
            if (!parse_uint(neg ? bank.substr(1) : bank, 10, r.bank)) return false;
            if (neg) r.bank = -r.bank;
            std::string_view handle = lhs.substr(colon + 1, close - colon - 1);
            size_t x = handle.find('x');
            if (x != std::string_view::npos) {
                if (!parse_uint(handle.substr(x + 1), 10, r.array_length)) return false;
                handle = handle.substr(0, x);
            }
            if (!parse_uint(handle, 10, r.word)) return false;
        } else {
            return false;
        }
        out.push_back(std::move(r));
    }
    return true;
}

size_t parse_results_tsv(std::string_view text, std::unordered_map<std::string, std::string>& out) {
    size_t lines = 0;
    int resources_col = -1;
    while (!text.empty()) {
        size_t nl = text.find('\n');
        std::string_view line = text.substr(0, nl);
        text = nl == std::string_view::npos ? std::string_view{} : text.substr(nl + 1);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        std::vector<std::string_view> cols;
        for (size_t start = 0;;) {
            size_t tab = line.find('\t', start);
            cols.push_back(line.substr(start, tab == std::string_view::npos ? std::string_view::npos : tab - start));
            if (tab == std::string_view::npos) break;
            start = tab + 1;
        }
        if (resources_col < 0) {  // header line names the columns
            for (size_t i = 0; i < cols.size(); i++)
                if (cols[i] == "resources") resources_col = int(i);
            if (resources_col < 0) return 0;
            continue;
        }
        if (cols.size() <= size_t(resources_col) || cols[0].empty()) continue;
        out[std::string(cols[0])] = std::string(cols[resources_col]);
        lines++;
    }
    return lines;
}

bool apply_harness(ShaderReflection& refl, const std::vector<HarnessResource>& resources) {
    bool ok = true;
    for (const HarnessResource& r : resources) {
        DescriptorBinding* b = nullptr;
        for (auto& x : refl.bindings)
            if (int(x.declared_set) == r.set && int(x.declared_binding) == r.binding) b = &x;
        if (!b) {
            ok = false;
            continue;
        }
        switch (r.kind) {
        case HarnessResource::Storage:
            if (b->type != DescriptorType::StorageBuffer || r.bank != 0) {
                ok = false;
                break;
            }
            set_c0_offset(b->nvn, refl.stage, r.word * 4);  // the harness prints 32-bit words
            break;
        case HarnessResource::Texture:
        case HarnessResource::Image:
            if (b->nvn.word < 0) set_texture_handle(b->nvn, r.kind == HarnessResource::Image, r.bank, r.word);
            break;
        case HarnessResource::Uniform:
            break;  // name and binding already say it
        }
    }
    return ok;
}

std::string sampler_type_text(const ImageShape& s) {
    static const char* kDims[] = {"Texture1D", "Texture2D", "Texture3D", "TextureCube", "?", "TextureBuffer"};
    std::string t = s.dim < 6 ? kDims[s.dim] : "?";
    if (s.arrayed) t += "|Array";
    if (s.ms) t += "|Multisample";
    if (s.depth) t += "|Shadow";
    return t;
}

std::string describe_like_harness(const ShaderReflection& refl) {
    std::string out;
    char tmp[160];
    for (const auto& b : refl.bindings) {
        if (!b.used) continue;  // declared, never referenced: the harness does not list it
        const NvnMeaning& m = b.nvn;
        tmp[0] = 0;
        const uint32_t set = b.declared_set, binding = b.declared_binding;
        switch (b.type) {
        case DescriptorType::UniformBuffer:
            if (m.source != NvnSource::SupportBuffer)
                std::snprintf(tmp, sizeof tmp, "cb%d=s%ub%u", m.bank, set, binding);
            break;
        case DescriptorType::StorageBuffer:
            if (m.c0_offset >= 0)
                std::snprintf(tmp, sizeof tmp, "sb[c0+0x%x]=s%ub%u", m.c0_offset / 4, set, binding);
            else
                std::snprintf(tmp, sizeof tmp, "sb[?]=s%ub%u", set, binding);
            break;
        default: {
            bool image = b.type == DescriptorType::StorageImage || b.type == DescriptorType::StorageTexelBuffer;
            std::string arr = b.count != 1 ? "x" + std::to_string(b.count) : "";
            const std::string type = b.type == DescriptorType::Sampler ? "None" : sampler_type_text(b.image);   // the sampler pool array
            std::snprintf(tmp, sizeof tmp, "%s%s[c%d:%d%s]=s%ub%u", image ? "img" : "tex", type.c_str(), m.default_bank ? -1 : m.bank, m.word,
                          arr.c_str(), set, binding);
        }
        }
        if (!tmp[0]) continue;
        if (!out.empty()) out += ' ';
        out += tmp;
    }
    return out;
}

// ---- one stage ------------------------------------------------------------------------------------------------------

namespace {
// NVN2 compute shaders read their resources from a per-program table in c[0] (Switch 1 used 0x210/0x310 and c[2]):
// 0x10 header bytes, 4 * control byte +0xa9 bytes of push constant data (nvnCommandBufferPushComputeConstants), then 0x10 per
// uniform block descriptor, 0x10 per storage buffer descriptor, 8 per texture handle, 8 per image handle, slot n = NVN binding n.
// The counts are control bytes +0xb1/+0xb2/+0xb3/+0xb6, and the table size at u16 +0xba equals
// 0x10 + 4 * push words + 0x10 * (ubos + ssbos) + 8 * (textures + images) in all 3,142 compute blocks of Pokopia's corpus (no push
// words there) and all 118 of Fortune's Weave's (28 push words in 38 of them, whose storage buffer and texture words the translated code
// reads all fall in the shifted ranges). Checked on e764a3f2 (sb at 0x10), 62619a57 (3 ubos: sb at 0x40, texture at 0x50), 7dbc445a
// (texture 0x40, image 0x48), 0034791c (2 ubos, 30 textures from 0x30 by GLSLC binding, image at 0x120) and Fortune's Weave's
// 09b66614 (28 push words, 1 ubo: sb at 0x90, image at 0xa0).
void apply_nvn2_compute_table(ShaderReflection& refl, const ShaderControl& c) {
    if (!c.nvn2 || c.stage != 5) return;
    const int ub0 = 0x10 + 4 * c.c0_push_words, sb0 = ub0 + 0x10 * c.c0_ubos, tex0 = sb0 + 0x10 * c.c0_ssbos;
    const int img0 = tex0 + 8 * c.c0_textures, end = img0 + 8 * c.c0_images;
    for (DescriptorBinding& b : refl.bindings) {
        NvnMeaning& m = b.nvn;
        if (m.source == NvnSource::TexturePoolArray || m.source == NvnSource::SamplerPoolArray) continue;
        if (b.type == DescriptorType::StorageBuffer) {
            const int off = m.c0_offset;
            if (off < ub0 || off >= tex0 || off % 0x10) continue;
            if (off < sb0) m.source = NvnSource::UniformAsStorage, m.index = (off - ub0) / 0x10;
            else m.source = NvnSource::StorageBuffer, m.index = (off - sb0) / 0x10;
        } else if (b.type != DescriptorType::UniformBuffer && m.bank == 0 && m.word >= 0 && m.word < 0x4000) {
            const int off = m.word * 4;
            const bool image = b.type == DescriptorType::StorageImage || b.type == DescriptorType::StorageTexelBuffer;
            if (!image && off >= img0 && off < end && off % 8 == 0) {
                // a texture read through an image slot's handle (texture id, sampler 0): the shader samples the image it writes (23 of
                // Fortune's Weave's 118 compute shaders, all with push words)
                m.source = NvnSource::ImageBinding;
                m.index = (off - img0) / 8;
                continue;
            }
            const int base = image ? img0 : tex0, limit = image ? end : img0;
            if (off < base || off >= limit || off % 8) continue;
            m.source = image ? NvnSource::ImageBinding : NvnSource::SamplerBinding;
            m.index = (off - base) / 8;
        }
    }
}
}  // namespace

int pool_binding_of(const DescriptorBinding& b) {
    if (b.nvn.source == NvnSource::SamplerPoolArray) return b.type == DescriptorType::Sampler ? int(kPoolSamplers) : -1;
    if (b.nvn.source != NvnSource::TexturePoolArray || b.type != DescriptorType::SampledImage) return -1;
    const ImageShape& s = b.image;
    if (s.arrayed || s.ms || s.component != ScalarType::Float) return -1;
    return s.dim == 1 ? int(kPoolTex2D) : s.dim == 3 ? int(kPoolTexCube) : -1;
}

bool move_pool_arrays(std::vector<uint32_t>& w, ShaderReflection& refl, std::string* error) {
    auto fail = [&](std::string why) {
        if (error) *error = std::move(why);
        return false;
    };
    struct Move { uint32_t set, binding, to; bool done; };
    std::vector<Move> moves;
    for (const DescriptorBinding& b : refl.bindings) {
        if (b.nvn.source != NvnSource::TexturePoolArray && b.nvn.source != NvnSource::SamplerPoolArray) continue;
        const int to = pool_binding_of(b);
        if (to < 0) return fail("pool array " + b.name + " has an image shape the pool set does not have");
        const uint32_t cap = to == int(kPoolSamplers) ? kPoolSamplerCount : kPoolTextureCount;
        if (b.count == 0 || b.count > cap) return fail("pool array " + b.name + " is longer than the pool set's");
        moves.push_back({b.declared_set, b.declared_binding, uint32_t(to), false});   // the words as translated (also when read again)
    }
    if (moves.empty()) return true;
    // id -> word index of the literal of its DescriptorSet / Binding decoration
    std::unordered_map<uint32_t, std::pair<size_t, size_t>> at;
    for (size_t i = 5; i < w.size();) {
        const uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
        if (wc == 0 || i + wc > w.size()) return fail("truncated instruction");
        if (op == OpDecorate && wc >= 4 && (w[i + 2] == DecDescriptorSet || w[i + 2] == DecBinding)) {
            auto& e = at.try_emplace(w[i + 1], SIZE_MAX, SIZE_MAX).first->second;
            (w[i + 2] == DecDescriptorSet ? e.first : e.second) = i + 3;
        }
        i += wc;
    }
    for (auto& [id, e] : at) {
        if (e.first == SIZE_MAX || e.second == SIZE_MAX) continue;
        for (Move& mv : moves)
            if (!mv.done && w[e.first] == mv.set && w[e.second] == mv.binding) {
                w[e.first] = kPoolSet;
                w[e.second] = mv.to;
                mv.done = true;
                break;
            }
    }
    for (const Move& mv : moves)
        if (!mv.done) return fail("pool array decoration not found");
    for (DescriptorBinding& b : refl.bindings)
        for (const Move& mv : moves)
            if (b.declared_set == mv.set && b.declared_binding == mv.binding) {
                b.set = kPoolSet;
                b.binding = mv.to;
                break;
            }
    std::sort(refl.bindings.begin(), refl.bindings.end(), [](const DescriptorBinding& a, const DescriptorBinding& b) {
        return a.set != b.set ? a.set < b.set : a.binding < b.binding;
    });
    return true;
}

bool make_shader_info(const uint8_t* code, const uint8_t* control, std::vector<uint32_t> spirv,
                      const std::string* resources, ShaderInfo& out, std::string* error, const ShaderReflection* pre_refl) {
    auto fail = [&](std::string why) {
        if (error) *error = std::move(why);
        return false;
    };
    out = ShaderInfo{};
    out.control = read_control(control);
    const ShaderControl& c = out.control;
    if (!c.valid) return fail("control section without magic 0x98761234");
    if (c.code_size == 0 || c.code_size > 16u << 20) return fail("implausible code block size");
    out.hash = shader_hash(code, c.code_size);
    if (pre_refl) out.refl = *pre_refl;
    else if (!reflect_spirv(spirv.data(), spirv.size(), out.refl, error)) return false;
    out.stage = out.refl.stage;
    out.spirv = std::move(spirv);
    if (c.const_size) {
        if (uint64_t(c.const_offset) + c.const_size > c.code_size) return fail("constant data outside the code block");
        out.constants.assign(code + c.const_offset, code + c.const_offset + c.const_size);
    }
    if (out.stage == ShaderStage::Compute)
        for (int i = 0; i < 3; i++) out.local_size[i] = std::max<uint32_t>(1, c.local_size[i]);
    if (resources && !pre_refl) {
        std::vector<HarnessResource> list;
        if (!parse_harness_resources(*resources, list)) return fail("unreadable resources record");
        if (!apply_harness(out.refl, list)) return fail("resources record does not match the SPIR-V");
    }
    apply_nvn2_compute_table(out.refl, c);
    return move_pool_arrays(out.spirv, out.refl, error);
}

const char* stage_name(ShaderStage s) {
    switch (s) {
    case ShaderStage::Vertex: return "vertex";
    case ShaderStage::Fragment: return "fragment";
    case ShaderStage::Compute: return "compute";
    case ShaderStage::Geometry: return "geometry";
    case ShaderStage::TessControl: return "tessellationcontrol";      // the harness's spelling (results.tsv)
    case ShaderStage::TessEval: return "tessellationevaluation";
    default: return "unknown";
    }
}

const char* source_name(NvnSource s) {
    switch (s) {
    case NvnSource::SupportBuffer: return "support buffer";
    case NvnSource::DriverConstants: return "c[0] driver constants";
    case NvnSource::ConstantData: return "c[1] constant data";
    case NvnSource::UniformBlock: return "uniform block";
    case NvnSource::StorageBuffer: return "storage buffer";
    case NvnSource::UniformAsStorage: return "uniform block (as storage)";
    case NvnSource::SamplerBinding: return "sampler binding";
    case NvnSource::ImageBinding: return "image binding";
    case NvnSource::HandleInUniform: return "handle in uniform block";
    case NvnSource::PushConstants: return "push constants";
    case NvnSource::TexturePoolArray: return "texture pool array";
    case NvnSource::SamplerPoolArray: return "sampler pool array";
    default: return "unknown";
    }
}

// ---- the runtime cache ----------------------------------------------------------------------------------------------

namespace {

bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream s;
    s << f.rdbuf();
    out = s.str();
    return true;
}

// Tab-separated column `col` (0-based) of one line; empty if the line has fewer columns.
std::string column_text(std::string_view line, int col) {
    size_t start = 0;
    for (int i = 0; i < col; i++) {
        size_t tab = line.find('\t', start);
        if (tab == std::string_view::npos) return {};
        start = tab + 1;
    }
    size_t tab = line.find('\t', start);
    return std::string(line.substr(start, tab == std::string_view::npos ? std::string_view::npos : tab - start));
}

// ---- the preload pack ---------------------------------------------------------------------------------------------
// <dir>.preload.pack (shader_cache/spirv.preload.pack: beside the cache directory, not in it; under Wine a file name that is not there
// costs a case-insensitive scan of its whole directory, 3 s for the 99k-file cache on a microSD card) holds the .spv files of the order
// file in one file, in that order, each one zstd frame. Read as thousands of
// separate files from a microSD card (the Steam Deck), the boot preload ran at ~15 MB/s and the game sat on a black screen for a minute
// or more; the pack is ~4x smaller and is read front to back (pack_prefetch).
// Layout, little endian: PackHeader, the frames, then `count` PackDiskEntry. line_hash is FNV-1a 64 of the shader's results.tsv line when
// the entry was written: an entry whose line has changed since (the shader was translated again) is not used, its .spv file is.
// The runtime writes the pack itself: when a run's boot preload is done and the order file lists shaders the pack lacks, a background
// thread (lowest priority) writes <dir>.preload.pack.new from the pack's own frames and the missing .spv files; the next start renames it to
// preload.pack before opening it (the open pack is never replaced under a running game). [nvn] shader_pack = 0: no pack, read or written.
constexpr char kPackMagic[8] = {'N', 'X', 'S', 'P', 'A', 'C', 'K', '1'};
constexpr uint32_t kPackVersion = 1;
constexpr int kPackLevel = 6;            // zstd level: ~3.8x on SPIR-V at ~200 MB/s (level 19 is ~5x at ~5 MB/s)
struct PackHeader {
    char magic[8];
    uint32_t version;
    uint32_t count;
    uint64_t index_offset;
    uint64_t reserved;
};
struct PackDiskEntry {
    char hash[16];
    uint64_t line_hash;
    uint64_t offset;
    uint32_t csize;
    uint32_t usize;
};
static_assert(sizeof(PackHeader) == 32 && sizeof(PackDiskEntry) == 40, "preload.pack layout");

// <dir>.preload.pack<suffix>; a trailing separator of the cache directory is dropped, so the file stays beside the directory.
std::string pack_path(std::string dir, const char* suffix) {
    while (!dir.empty() && (dir.back() == '/' || dir.back() == '\\')) dir.pop_back();
    return dir + ".preload.pack" + suffix;
}

bool pack_enabled() {
    static const bool on = nxrt::config_get_bool("nvn", "shader_pack", true);
    return on;
}

uint64_t fnv1a64(std::string_view s) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 0x100000001b3ull;
    }
    return h;
}

std::wstring widen(const std::string& s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(std::max(n, 0)), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

// A positional read: threads share the handle (under Wine a pread; Windows serialises reads on a synchronous handle, still correct).
bool read_at(void* file, uint64_t off, void* dst, uint32_t n) {
    OVERLAPPED ov = {};
    ov.Offset = static_cast<DWORD>(off);
    ov.OffsetHigh = static_cast<DWORD>(off >> 32);
    DWORD got = 0;
    return ReadFile(static_cast<HANDLE>(file), dst, n, &got, &ov) && got == n;
}

uint64_t now_ms() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

}  // namespace

// open(), lock_ held: results_ and lines_ are in place.
void ShaderCache::pack_open() {
    if (!pack_enabled()) return;
    const std::wstring path = widen(pack_path(dir_, "")), fresh = widen(pack_path(dir_, ".new"));
    if (GetFileAttributesW(fresh.c_str()) != INVALID_FILE_ATTRIBUTES && !MoveFileExW(fresh.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING))
        NXLOG_W("nvn", "shader pack: preload.pack.new could not replace preload.pack (error %lu)", static_cast<unsigned long>(GetLastError()));
    HANDLE f = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return;
    PackHeader h = {};
    LARGE_INTEGER size = {};
    const bool head_ok = GetFileSizeEx(f, &size) && read_at(f, 0, &h, sizeof h) && !std::memcmp(h.magic, kPackMagic, sizeof h.magic) &&
                         h.version == kPackVersion && h.count <= 1000000 && h.index_offset >= sizeof h &&
                         h.index_offset + uint64_t{h.count} * sizeof(PackDiskEntry) == static_cast<uint64_t>(size.QuadPart);
    std::vector<PackDiskEntry> idx(head_ok ? h.count : 0);
    if (!head_ok || (h.count && !read_at(f, h.index_offset, idx.data(), static_cast<uint32_t>(idx.size() * sizeof(PackDiskEntry))))) {
        NXLOG_W("nvn", "shader pack: preload.pack is not a pack this runtime reads: the .spv files are read (and a new pack written)");
        CloseHandle(f);
        return;
    }
    size_t stale = 0;
    uint64_t bytes = 0;
    for (const PackDiskEntry& e : idx) {
        std::string hash(e.hash, sizeof e.hash);
        auto ln = lines_.find(hash);
        if (ln == lines_.end() || fnv1a64(std::string_view(results_.data() + ln->second.first, ln->second.second)) != e.line_hash ||
            e.offset < sizeof h || e.csize == 0 || e.offset + e.csize > h.index_offset || e.usize == 0 || e.usize % 4 || e.usize > (64u << 20)) {
            stale++;
            continue;
        }
        bytes += e.usize;
        pack_.emplace(std::move(hash), PackEntry{e.offset, e.csize, e.usize});
    }
    pack_file_ = f;
    pack_data_end_ = h.index_offset;
    NXLOG_I("nvn", "shader pack: preload.pack has %zu shaders (%llu MiB of SPIR-V in %llu MiB)%s", pack_.size(),
            static_cast<unsigned long long>(bytes >> 20), static_cast<unsigned long long>(h.index_offset >> 20),
            stale ? (", " + std::to_string(stale) + " translated again since and read from their .spv files").c_str() : "");
}

bool ShaderCache::pack_read(const std::string& hash, std::string& out, bool* from_mem) {
    if (!pack_file_) return false;
    auto it = pack_.find(hash);
    if (it == pack_.end()) return false;
    const PackEntry& e = it->second;
    thread_local std::string comp;
    const char* src = nullptr;
    if (pack_mem_ && e.offset + e.csize <= pack_mem_end_.load(std::memory_order_acquire)) {
        src = pack_mem_.get() + e.offset;                            // resident ([nvn] shader_pack_resident): no disk access
        if (from_mem) *from_mem = true;
    } else {
        comp.resize(e.csize);
        if (!read_at(pack_file_, e.offset, comp.data(), e.csize)) return false;
        src = comp.data();
    }
    out.resize(e.usize);
    const size_t n = ZSTD_decompress(out.data(), out.size(), src, e.csize);
    if (ZSTD_isError(n) || n != e.usize) {
        out.clear();
        return false;
    }
    return true;
}

// Sampled images the shader reads only as integer bits become uint images (spirv_predict_int_fetches): through the float alias of an
// R32UI texture NVIDIA flushes the denormal bit patterns to zero (the clustered light grid read empty: issue #89). Done here, before the
// reflection, so the binding is reflected as a uint image and gets a UINT view; the pack and the .spv files keep the translator's words.
// [nvn] int_fetch_retype = false: off (A/B).
static void retype_int_fetches(const std::string& hash, std::string& bytes) {
    static const bool on = nxrt::config_get_bool("nvn", "int_fetch_retype", true);
    if (!on || bytes.size() < 20 || bytes.size() % 4) return;
    std::vector<uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    const std::vector<gpu::SpvIntSampler> slots = gpu::spirv_predict_int_fetches(words);
    if (slots.empty()) return;
    std::vector<uint32_t> patched;
    static std::atomic<int> done{0}, failed{0};
    if (!gpu::spirv_int_samplers(words, slots, patched)) {
        if (failed.fetch_add(1, std::memory_order_relaxed) < 4)
            NXLOG_W("nvn", "shader %s: integer fetch through set %u binding %u left as a float image (the rewrite declined the module)", hash.c_str(),
                    slots[0].set, slots[0].binding);
        return;
    }
    bytes.assign(reinterpret_cast<const char*>(patched.data()), patched.size() * 4);
    if (done.fetch_add(1, std::memory_order_relaxed) < 3)
        NXLOG_I("nvn", "shader %s: %zu sampled image(s) read only as integer bits (set %u binding %u first) are uint images", hash.c_str(), slots.size(),
                slots[0].set, slots[0].binding);
}

bool ShaderCache::read_spv(const std::string& hash, std::string& out, int* source) {
    bool mem = false;
    if (pack_read(hash, out, &mem)) {
        pre_from_pack_.fetch_add(1, std::memory_order_relaxed);
        if (source) *source = mem ? 1 : 2;
        retype_int_fetches(hash, out);
        return true;
    }
    if (source) *source = 3;
    pre_from_files_.fetch_add(1, std::memory_order_relaxed);
    if (!read_file(dir_ + "/" + hash + ".spv", out)) return false;
    retype_int_fetches(hash, out);
    return true;
}

// Reads the frames front to back on its own handle, ahead of the workers: one sequential read of the card, whose data the workers' reads
// then find in the operating system's cache.
void ShaderCache::pack_prefetch() {
    log_set_thread_name("nvn-pack-read");
    const uint64_t t0 = now_ms();
    HANDLE f = CreateFileW(widen(pack_path(dir_, "")).c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                           FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (f == INVALID_HANDLE_VALUE) return;
    // pack_mem_ (start_preload, [nvn] shader_pack_resident): the frames are read into memory that stays
    std::vector<char> buf(pack_mem_ ? 0 : (8u << 20));
    uint64_t off = 0;
    while (off < pack_data_end_ && !pack_stop_.load(std::memory_order_relaxed)) {
        DWORD got = 0;
        const DWORD want = static_cast<DWORD>(std::min<uint64_t>(8u << 20, pack_data_end_ - off));
        if (!ReadFile(f, pack_mem_ ? pack_mem_.get() + off : buf.data(), want, &got, nullptr) || got == 0) break;
        off += got;
        if (pack_mem_) pack_mem_end_.store(off, std::memory_order_release);
    }
    CloseHandle(f);
    const uint64_t ms = std::max<uint64_t>(1, now_ms() - t0);
    NXLOG_I("nvn", "shader pack: read %llu MiB in %.1f s (%.0f MB/s)%s", static_cast<unsigned long long>(off >> 20), ms / 1000.0,
            static_cast<double>(off) / 1000.0 / static_cast<double>(ms), pack_mem_ ? ", kept in memory" : "");
}

// The last preload worker, as it ends.
void ShaderCache::preload_done() {
    NXLOG_I("nvn", "shader preload: done after %.1f s (%llu .spv reads from preload.pack, %llu from separate files so far)",
            (now_ms() - pre_start_ms_) / 1000.0, static_cast<unsigned long long>(pre_from_pack_.load()),
            static_cast<unsigned long long>(pre_from_files_.load()));
}

// Started with the preload when the pack lacks shaders of the order file; writes once the preload is done (or after 90 s: workers wait
// while the prepared shaders the game has not taken fill [nvn] shader_preload_mb, and may then never end), so the boot reads come first.
void ShaderCache::pack_build(size_t missing) {
    log_set_thread_name("nvn-pack-write");
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    // A big build (the whole cache, [nvn] shader_pack_all) reads thousands of separate files: in background mode (very low I/O and memory
    // priority), so the game's own reads of the same disk go first, and not before the boot is over.
    const bool big = missing >= 500;
    if (big) SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN);
    while ((pre_live_.load() > 0 && now_ms() - pre_start_ms_ < 90000) || (big && now_ms() - pre_start_ms_ < 30000)) {
        if (pack_stop_.load()) return;
        Sleep(250);
    }
    const uint64_t t0 = now_ms();
    const std::string tmp_path = pack_path(dir_, ".tmp");
    const std::wstring tmp = widen(tmp_path), out = widen(pack_path(dir_, ".new"));
    NXLOG_I("nvn", "shader pack: writing preload.pack.new (%zu shaders: %zu of the order file%s; %zu not in preload.pack yet)", pack_list_.size(),
            pre_list_.size(), pack_list_.size() > pre_list_.size() ? ", then the rest of the cache" : "", missing);
    HANDLE f = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        NXLOG_W("nvn", "shader pack: cannot create %s (error %lu)", tmp_path.c_str(), static_cast<unsigned long>(GetLastError()));
        return;
    }
    auto put = [f](const void* p, size_t n) {
        DWORD w = 0;
        return WriteFile(f, p, static_cast<DWORD>(n), &w, nullptr) && w == n;
    };
    PackHeader h = {};
    std::memcpy(h.magic, kPackMagic, sizeof h.magic);
    h.version = kPackVersion;
    bool ok = put(&h, sizeof h);     // count and index_offset are filled in at the end
    uint64_t off = sizeof h, raw_total = 0;
    std::vector<PackDiskEntry> idx;
    idx.reserve(pack_list_.size());
    std::unordered_set<std::string> seen;
    std::string raw, comp;
    size_t reused = 0, compressed = 0, skipped = 0;
    ZSTD_CCtx* cc = ZSTD_createCCtx();
    for (const std::string& hash : pack_list_) {
        if (!ok || !cc) break;
        if (pack_stop_.load(std::memory_order_relaxed)) {
            ok = false;
            break;
        }
        if (hash.size() != sizeof(PackDiskEntry::hash) || !seen.insert(hash).second) continue;
        PackDiskEntry e = {};
        std::memcpy(e.hash, hash.data(), sizeof e.hash);
        {
            std::lock_guard lk(lock_);
            auto ln = lines_.find(hash);
            if (ln == lines_.end()) {
                skipped++;
                continue;
            }
            e.line_hash = fnv1a64(std::string_view(results_.data() + ln->second.first, ln->second.second));
        }
        if (auto it = pack_.find(hash); it != pack_.end()) {          // still current in the old pack: its frame as it is
            comp.resize(it->second.csize);
            if (pack_mem_ && it->second.offset + it->second.csize <= pack_mem_end_.load(std::memory_order_acquire)) {
                std::memcpy(comp.data(), pack_mem_.get() + it->second.offset, it->second.csize);
            } else if (!read_at(pack_file_, it->second.offset, comp.data(), it->second.csize)) {
                skipped++;
                continue;
            }
            e.usize = it->second.usize;
            reused++;
        } else {
            if (!read_file(dir_ + "/" + hash + ".spv", raw) || raw.empty() || raw.size() % 4 || raw.size() > (64u << 20)) {
                skipped++;
                continue;
            }
            comp.resize(ZSTD_compressBound(raw.size()));
            const size_t n = ZSTD_compressCCtx(cc, comp.data(), comp.size(), raw.data(), raw.size(), kPackLevel);
            if (ZSTD_isError(n)) {
                skipped++;
                continue;
            }
            comp.resize(n);
            e.usize = static_cast<uint32_t>(raw.size());
            compressed++;
        }
        e.offset = off;
        e.csize = static_cast<uint32_t>(comp.size());
        ok = put(comp.data(), comp.size());
        off += comp.size();
        raw_total += e.usize;
        idx.push_back(e);
    }
    if (cc) ZSTD_freeCCtx(cc);
    ok = ok && cc && !idx.empty();
    if (ok) {
        h.count = static_cast<uint32_t>(idx.size());
        h.index_offset = off;
        LARGE_INTEGER zero = {};
        ok = put(idx.data(), idx.size() * sizeof(PackDiskEntry)) && SetFilePointerEx(f, zero, nullptr, FILE_BEGIN) && put(&h, sizeof h);
    }
    ok = ok && FlushFileBuffers(f);   // on the card before the rename: a card pulled early leaves the old pack, never half a new one
    CloseHandle(f);
    if (!ok || !MoveFileExW(tmp.c_str(), out.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        DeleteFileW(tmp.c_str());
        if (!pack_stop_.load()) NXLOG_W("nvn", "shader pack: writing preload.pack.new failed (error %lu)", static_cast<unsigned long>(GetLastError()));
        return;
    }
    NXLOG_I("nvn", "shader pack: wrote preload.pack.new in %.1f s: %zu shaders (%zu from preload.pack, %zu compressed, %zu skipped), "
                   "%llu MiB of SPIR-V in %llu MiB; it is used from the next start", (now_ms() - t0) / 1000.0, idx.size(), reused, compressed, skipped,
            static_cast<unsigned long long>(raw_total >> 20), static_cast<unsigned long long>(off >> 20));
}

bool ShaderCache::open(const std::string& dir) {
    std::lock_guard lk(lock_);
    if (open_) return false;  // one directory per cache: loaded ShaderInfos keep their addresses
    std::string text;
    if (!read_file(dir + "/results.tsv", text)) {
        NXLOG_W("nvn", "shader cache %s: no readable results.tsv", dir.c_str());
        return false;
    }
    results_ = std::move(text);
    int resources_col = -1;
    bool header = true;
    size_t pos = 0;
    while (pos < results_.size()) {
        size_t nl = results_.find('\n', pos);
        size_t next = nl == std::string::npos ? results_.size() : nl + 1;
        size_t end = nl == std::string::npos ? results_.size() : nl;
        if (end > pos && results_[end - 1] == '\r') end--;
        std::string_view line(results_.data() + pos, end - pos);
        if (header) {
            header = false;
            int col = 0;
            for (size_t start = 0;; col++) {
                size_t tab = line.find('\t', start);
                const std::string_view name = line.substr(start, tab == std::string_view::npos ? std::string_view::npos : tab - start);
                if (name == "resources") resources_col = col;
                if (name == "spv_bytes") spv_bytes_col_ = col;
                if (tab == std::string_view::npos) break;
                start = tab + 1;
            }
            if (resources_col < 0) {
                NXLOG_W("nvn", "shader cache %s: results.tsv has no resources column", dir.c_str());
                results_.clear();
                return false;
            }
        } else if (!line.empty()) {
            std::string hash(line.substr(0, line.find('\t')));
            if (!hash.empty()) lines_.emplace(std::move(hash), std::make_pair(pos, end - pos));
        }
        pos = next;
    }
    resources_col_ = resources_col;
    dir_ = dir;
    open_ = true;
    NXLOG_I("nvn", "shader cache %s: %zu records", dir.c_str(), lines_.size());
    pack_open();
    start_preload();
    return true;
}

// Called from open() with lock_ held: reads the order file and starts the workers (they take lock_ only to read results_).
void ShaderCache::start_preload() {
    order_path_ = nxrt::config_get_str("nvn", "shader_order", "shader_order.txt");
    if (order_path_.empty()) return;
    std::string text;
    if (read_file(order_path_, text)) {
        size_t pos = 0;
        while (pos < text.size()) {
            size_t nl = text.find('\n', pos);
            if (nl == std::string::npos) nl = text.size();
            std::string h = text.substr(pos, nl - pos);
            if (!h.empty() && h.back() == '\r') h.pop_back();
            if (!h.empty() && order_known_.insert(h).second) pre_list_.push_back(std::move(h));
            pos = nl + 1;
        }
    }
    const int64_t threads = std::clamp<int64_t>(nxrt::config_get_int("nvn", "shader_preload_threads", 6), 0, 16);
    pre_budget_ = static_cast<size_t>(std::max<int64_t>(64, nxrt::config_get_int("nvn", "shader_preload_mb", 1024))) << 20;
    // New lines of the order file are appended every few seconds by their own thread: a batch held until it was full (77 shaders) or until a
    // shutdown that a closed window never runs was lost, and the next start preloaded none of those shaders.
    order_thread_ = std::thread([this] { order_writer(); });
    // [nvn] shader_pack_all (default on): the pack also holds every other shader of results.tsv that has SPIR-V, after the order file's.
    static const bool pack_all = nxrt::config_get_bool("nvn", "shader_pack_all", true);
    pack_list_ = pre_list_;
    if (pack_all) {
        std::vector<std::string> rest;
        for (const auto& [h, ln] : lines_) {
            if (order_known_.count(h)) continue;
            if (spv_bytes_col_ >= 0) {
                const std::string b = column_text(std::string_view(results_.data() + ln.first, ln.second), spv_bytes_col_);
                if (b.empty() || b == "0") continue;          // not translated: no .spv
            }
            rest.push_back(h);
        }
        std::sort(rest.begin(), rest.end());
        pack_list_.insert(pack_list_.end(), rest.begin(), rest.end());
    }
    pre_start_ms_ = now_ms();
    static const bool resident = nxrt::config_get_bool("nvn", "shader_pack_resident", true);
    const bool prefetch = pack_file_ && !pack_.empty() && (resident || (!pre_list_.empty() && threads > 0));
    if (prefetch && resident) pack_mem_.reset(new (std::nothrow) char[pack_data_end_]);   // before any reader can look at it
    if (!pre_list_.empty() && threads > 0) {
        NXLOG_I("nvn", "shader preload: %zu shaders from %s on %lld threads", pre_list_.size(), order_path_.c_str(),
                static_cast<long long>(threads));
        pre_live_.store(static_cast<int>(threads));
    }
    if (prefetch) pack_prefetch_thread_ = std::thread([this] { pack_prefetch(); });
    if (pack_enabled()) {
        size_t missing = 0;
        for (const std::string& h : pack_list_)
            if (!pack_.count(h)) missing++;
        // A handful of new shaders is not worth rewriting a few hundred MB on the card: they are read from their files until more come.
        if (missing >= 32 || (missing && missing * 50 >= pack_list_.size())) pack_build_thread_ = std::thread([this, missing] { pack_build(missing); });
    }
    if (pre_list_.empty() || threads == 0) return;
    for (int64_t i = 0; i < threads; ++i) pre_threads_.emplace_back([this] { preload_worker(); });
}

// Appends the order file's new lines (note_order_locked) every 2 s, and the rest when the cache is destroyed. No file I/O on the game thread.
void ShaderCache::order_writer() {
    log_set_thread_name("nvn-order-write");
    std::unique_lock lk(lock_);
    for (;;) {
        order_cv_.wait_for(lk, std::chrono::seconds(2), [this] { return order_stop_; });
        std::string lines;
        lines.swap(order_pending_);
        const bool stop = order_stop_;
        lk.unlock();
        flush_order(lines);
        lk.lock();
        if (stop) return;
    }
}

void ShaderCache::preload_worker() {
    struct LastOut {
        ShaderCache* c;
        ~LastOut() {
            if (c->pre_live_.fetch_sub(1) == 1) c->preload_done();
        }
    } last_out{this};
    for (;;) {
        std::string hash;
        {
            std::unique_lock lk(pre_mu_);
            // Stay inside the memory budget: the entries the game has not taken yet are all in memory.
            pre_cv_.wait(lk, [this] { return pre_stop_ || pre_bytes_ < pre_budget_ || pre_next_ >= pre_list_.size(); });
            if (pre_stop_ || pre_next_ >= pre_list_.size()) return;
            hash = pre_list_[pre_next_++];
            pre_inflight_.insert(hash);
        }
        std::unique_ptr<Preloaded> p = preload_one(hash);
        {
            std::lock_guard lk(pre_mu_);
            pre_inflight_.erase(hash);
            if (p) {
                pre_bytes_ += p->words.size() * 4;
                pre_.emplace(hash, std::move(p));
            }
        }
        pre_cv_.notify_all();   // find() may be waiting for this very hash
    }
}

// One shader: the file read and the reflection, no lock held for them. Null if the game already has it or needs its own load.
std::unique_ptr<ShaderCache::Preloaded> ShaderCache::preload_one(const std::string& hash) {
    NXFT("shader.preload");
    {
        std::lock_guard lk(lock_);
        if (by_hash_.count(hash)) return nullptr;   // the game got there first
    }
    std::string bytes;
    if (!read_spv(hash, bytes) || bytes.empty() || bytes.size() % 4) return nullptr;
    auto p = std::make_unique<Preloaded>();
    p->words.resize(bytes.size() / 4);
    std::memcpy(p->words.data(), bytes.data(), bytes.size());
    bytes = std::string();
    if (!reflect_spirv(p->words.data(), p->words.size(), p->refl, nullptr)) return nullptr;
    std::string record;
    bool have_record = false;
    {
        std::lock_guard lk(lock_);
        if (by_hash_.count(hash)) return nullptr;
        if (auto it = lines_.find(hash); it != lines_.end()) {
            record = column_text(std::string_view(results_.data() + it->second.first, it->second.second), resources_col_);
            have_record = true;
        }
    }
    if (have_record) {
        std::vector<HarnessResource> list;
        if (!parse_harness_resources(record, list) || !apply_harness(p->refl, list)) return nullptr;   // the game's own load reports it
    }
    return p;
}

// Waits while a worker is reading `hash` (one read, not two), then hands over the prepared entry if there is one.
std::unique_ptr<ShaderCache::Preloaded> ShaderCache::take_preloaded(const std::string& hash) {
    std::unique_lock lk(pre_mu_);
    pre_cv_.wait(lk, [&] { return pre_stop_ || !pre_inflight_.count(hash); });
    auto it = pre_.find(hash);
    if (it == pre_.end()) return nullptr;
    auto p = std::move(it->second);
    pre_.erase(it);
    pre_bytes_ -= std::min(pre_bytes_, p->words.size() * 4);
    lk.unlock();
    pre_cv_.notify_all();   // a paused worker may continue
    return p;
}

// lock_ held. New hashes are collected for the order file in batches (a few thousand in a whole session); a full batch is moved to
// `flush` and the caller appends it after releasing lock_ (no file I/O under the lock).
void ShaderCache::note_order_locked(const std::string& hash, std::string& flush) {
    if (order_path_.empty() || !order_known_.insert(hash).second) return;
    order_pending_ += hash;
    order_pending_ += '\n';
    if (order_thread_.joinable()) return;            // order_writer appends it within 2 s
    if (order_pending_.size() < 32 * 41) return;
    flush.swap(order_pending_);
    order_pending_.clear();
}

void ShaderCache::flush_order(const std::string& lines) {
    if (lines.empty()) return;
    std::lock_guard lk(order_io_mu_);
    std::ofstream f(order_path_, std::ios::binary | std::ios::app);
    if (f) f << lines;
}

ShaderCache::~ShaderCache() {
    {
        std::lock_guard lk(pre_mu_);
        pre_stop_ = true;
    }
    pre_cv_.notify_all();
    pack_stop_.store(true);
    {
        std::lock_guard lk(lock_);
        order_stop_ = true;
    }
    order_cv_.notify_all();
    if (order_thread_.joinable()) order_thread_.join();
    for (std::thread& t : pre_threads_)
        if (t.joinable()) t.join();
    if (pack_prefetch_thread_.joinable()) pack_prefetch_thread_.join();
    if (pack_build_thread_.joinable()) pack_build_thread_.join();
    if (pack_file_) CloseHandle(static_cast<HANDLE>(pack_file_));
    if (!order_pending_.empty() && !order_path_.empty()) {
        std::ofstream f(order_path_, std::ios::binary | std::ios::app);
        if (f) f << order_pending_;
    }
}

namespace {
void dump_missing_block(const std::string& hash, const uint8_t* code, const uint8_t* control, uint32_t code_size);
}

const ShaderInfo* ShaderCache::find(const uint8_t* code, const uint8_t* control) {
    if (!code || !control) return nullptr;
    ShaderControl c = read_control(control);
    if (!c.valid || c.code_size == 0 || c.code_size > 16u << 20) {
        std::lock_guard lk(lock_);
        if (bad_controls_.insert(reinterpret_cast<uintptr_t>(control)).second)
            NXLOG_W("nvn", "shader control %p: bad section (code size %#x)", (const void*)control, c.code_size);
        return nullptr;
    }
    std::string hash = shader_hash(code, c.code_size);  // a pass over the block: outside the lock
    std::string record;
    bool have_record = false, is_open;
    {
        std::lock_guard lk(lock_);
        is_open = open_;
        if (is_open) {
            if (auto it = by_hash_.find(hash); it != by_hash_.end()) return it->second.get();
            if (auto ln = lines_.find(hash); ln != lines_.end()) {
                record = column_text(std::string_view(results_.data() + ln->second.first, ln->second.second), resources_col_);
                have_record = true;
            }
        }
    }
    if (!is_open) {   // no cache at all: every block is missing, and still goes to [nvn] dump_missing
        dump_missing_block(hash, code, control, c.code_size);
        return nullptr;
    }
    return load(hash, code, control, c.code_size, have_record ? &record : nullptr);
}

namespace {
// [nvn] dump_missing (NXRT_NVN_DUMP_MISSING): a folder that receives every block the cache has no translation for, in the layout of
// the shader corpus (code_<hash>.bin, ctrl_<hash>.bin, manifest.tsv), so `spirv_harness --batch <folder>` can translate it.
// The stage is the word at control +0x714 (0 vertex, 1 fragment, 2 geometry, 5 compute).
void dump_missing_block(const std::string& hash, const uint8_t* code, const uint8_t* control, uint32_t code_size) {
    static const std::string dir = nxrt::config_get_str("nvn", "dump_missing", "");
    if (dir.empty()) return;
    static std::mutex mu;
    static std::unordered_set<std::string> done;
    std::lock_guard<std::mutex> lk(mu);
    if (!done.insert(hash).second) return;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const ShaderControl c = read_control(control);
    const uint32_t kCtrlBytes = c.nvn2 ? std::max<uint32_t>(0xd8, rd32(control + 0x30)) : 0x880;
    uint32_t stage_word = 0;
    if (c.nvn2) stage_word = static_cast<uint32_t>(c.stage);
    else std::memcpy(&stage_word, control + 0x714, 4);
    const char* stage = stage_word == 0 ? "vertex" : stage_word == 1 ? "fragment" : stage_word == 2 ? "geometry" : stage_word == 3 ? "tess_control"
                      : stage_word == 4 ? "tess_eval" : stage_word == 5 ? "compute" : "unknown";
    {
        std::ofstream f(dir + "/code_" + hash + ".bin", std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(code), code_size);
    }
    {
        std::ofstream f(dir + "/ctrl_" + hash + ".bin", std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(control), kCtrlBytes);
    }
    const std::string manifest = dir + "/manifest.tsv";
    const bool fresh = !std::filesystem::exists(manifest, ec);
    std::ofstream m(manifest, std::ios::app);
    if (fresh) m << "id\tstage\tcode_off\tcode_size\tctrl_off\tctrl_size\tprogram_start\tprogram_bytes\tconst_off\tconst_size\tcode_sha1_16\tcode_file\tctrl_file\n";
    char row[512];
    std::snprintf(row, sizeof row, "run_%s\t%s\t0x0\t0x%x\t0x0\t0x%x\t0x%x\t0x%x\t0x%x\t0x%x\t%s\tcode_%s.bin\tctrl_%s.bin\n", hash.c_str(), stage, code_size,
                  kCtrlBytes, c.program_start, c.program_size, c.const_offset, c.const_size, hash.c_str(), hash.c_str(), hash.c_str());
    m << row;
}
}  // namespace

const ShaderInfo* ShaderCache::load(const std::string& hash, const uint8_t* code, const uint8_t* control, uint32_t code_size,
                                    const std::string* record_ptr) {
    // No file I/O under lock_. dir_ is written once, by open(), before the cache is used.
    auto miss = [&](const std::string& why) -> const ShaderInfo* {
        {
            std::lock_guard lk(lock_);
            auto r = by_hash_.emplace(hash, nullptr);
            if (!r.second) return r.first->second.get();   // another thread settled this hash while we were loading
            missing_++;
        }
        NXLOG_W("nvn", "shader %s (code %#x bytes): %s", hash.c_str(), code_size, why.c_str());
        return nullptr;
    };
    NXFT("shader.load");
    const auto t0 = std::chrono::steady_clock::now();
    int source = 0;                                          // 0 preloaded words, 1 the pack in memory, 2 the pack file, 3 a .spv file
    std::unique_ptr<Preloaded> pre = take_preloaded(hash);   // waits for a worker that is reading this file: one read, not two
    std::vector<uint32_t> words;
    if (pre) {
        words = std::move(pre->words);
    } else {
        std::string bytes;
        if (!read_spv(hash, bytes, &source) || bytes.empty() || bytes.size() % 4) {
            dump_missing_block(hash, code, control, code_size);
            return miss("no usable " + dir_ + "/" + hash + ".spv");
        }
        words.resize(bytes.size() / 4);
        std::memcpy(words.data(), bytes.data(), bytes.size());
    }

    auto info = std::make_unique<ShaderInfo>();
    std::string error;
    if (!make_shader_info(code, control, std::move(words), record_ptr, *info, &error, pre ? &pre->refl : nullptr)) return miss(error);
    std::string flush;
    const ShaderInfo* out;
    {
        std::lock_guard lk(lock_);
        auto r = by_hash_.emplace(hash, std::move(info));   // a second reader of the same file loses here and its copy is dropped
        out = r.first->second.get();
        if (r.second) {
            loaded_++;
            note_order_locked(hash, flush);
        }
    }
    flush_order(flush);
    note_load(source, std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
    return out;
}

// What the game's shader loads cost (they run on the thread that sets the program, usually the game thread): counts by source, time, and a
// line every 10 s while new ones come. The hitch line gets shader_loads / shader_file_reads (nxrt/gauge.h).
namespace {
std::atomic<uint64_t> g_hitch_loads{0}, g_hitch_file_reads{0}, g_hitch_load_ms{0};
std::atomic<uint64_t> g_hitch_load_us_acc{0};
}  // namespace

void ShaderCache::note_load(int source, int64_t us) {
    static const bool registered = [] {
        nxrt::hitch_add_counter("shader_loads", []() -> uint64_t { return g_hitch_loads.load(std::memory_order_relaxed); });
        nxrt::hitch_add_counter("shader_file_reads", []() -> uint64_t { return g_hitch_file_reads.load(std::memory_order_relaxed); });
        nxrt::hitch_add_counter("shader_load_ms", []() -> uint64_t { return g_hitch_load_ms.load(std::memory_order_relaxed); });
        return true;
    }();
    (void)registered;
    const uint64_t u = static_cast<uint64_t>(std::max<int64_t>(0, us));
    g_hitch_loads.fetch_add(1, std::memory_order_relaxed);
    if (source == 3) g_hitch_file_reads.fetch_add(1, std::memory_order_relaxed);
    g_hitch_load_ms.store((g_hitch_load_us_acc.fetch_add(u, std::memory_order_relaxed) + u) / 1000, std::memory_order_relaxed);
    const uint64_t n = ld_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    (source == 0 ? ld_pre_ : source == 1 ? ld_pack_mem_ : source == 2 ? ld_pack_ : ld_files_).fetch_add(1, std::memory_order_relaxed);
    ld_us_.fetch_add(u, std::memory_order_relaxed);
    if (source == 3) ld_files_us_.fetch_add(u, std::memory_order_relaxed);
    uint64_t w = ld_worst_us_.load(std::memory_order_relaxed);
    while (u > w && !ld_worst_us_.compare_exchange_weak(w, u, std::memory_order_relaxed)) {}
    if (u >= 20000)
        NXLOG_W("nvn", "shader load took %.1f ms on thread %lu (%s)", u / 1000.0, static_cast<unsigned long>(GetCurrentThreadId()),
                source == 0 ? "preloaded" : source == 1 ? "pack in memory" : source == 2 ? "read from the pack file" : "read from its .spv file");
    const uint64_t now = now_ms();
    uint64_t last = ld_logged_ms_.load(std::memory_order_relaxed);
    if (now - last < 10000 || !ld_logged_ms_.compare_exchange_strong(last, now)) return;
    const uint64_t since = n - ld_logged_count_.exchange(n);
    NXLOG_I("nvn", "shader loads by the game: %llu (+%llu in the last 10 s): preloaded %llu, pack in memory %llu, pack file %llu, .spv files %llu "
                   "(%.0f ms); %.0f ms in all, worst %.1f ms",
            static_cast<unsigned long long>(n), static_cast<unsigned long long>(since), static_cast<unsigned long long>(ld_pre_.load()),
            static_cast<unsigned long long>(ld_pack_mem_.load()), static_cast<unsigned long long>(ld_pack_.load()),
            static_cast<unsigned long long>(ld_files_.load()), ld_files_us_.load() / 1000.0, ld_us_.load() / 1000.0, ld_worst_us_.load() / 1000.0);
}

const ShaderInfo* ShaderCache::load_stub(const std::string& hash) {
    std::string dir, record;
    bool have_record = false;
    ShaderInfo* existing = nullptr;
    {
        std::lock_guard lk(lock_);
        if (!open_) return nullptr;
        auto it = stubs_.find(hash);
        if (it != stubs_.end()) {
            if (!it->second) return nullptr;
            existing = it->second.get();
            if (!existing->spirv.empty()) return existing;        // checked under the lock: another thread may be putting the words back
        } else if (auto ln = lines_.find(hash); ln != lines_.end()) {
            record = column_text(std::string_view(results_.data() + ln->second.first, ln->second.second), resources_col_);
            have_record = true;
        }
        dir = dir_;
    }
    std::string bytes;
    if (!read_spv(hash, bytes) || bytes.empty() || bytes.size() % 4) {
        if (!existing) {
            std::lock_guard lk(lock_);
            stubs_.emplace(hash, nullptr);
        }
        return nullptr;
    }
    std::vector<uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    if (existing) {                         // the words were dropped: put them back (the reflection is kept)
        ShaderReflection again = existing->refl;
        if (!move_pool_arrays(words, again)) return existing;
        std::lock_guard lk(lock_);
        if (existing->spirv.empty()) existing->spirv = std::move(words);
        return existing;
    }
    auto info = std::make_unique<ShaderInfo>();
    info->hash = hash;
    std::string error;
    if (!reflect_spirv(words.data(), words.size(), info->refl, &error)) info.reset();
    if (info && have_record) {
        std::vector<HarnessResource> list;
        if (!parse_harness_resources(record, list) || !apply_harness(info->refl, list)) info.reset();
    }
    if (info && !move_pool_arrays(words, info->refl, &error)) info.reset();
    if (info) {
        info->stage = info->refl.stage;
        info->spirv = std::move(words);
    }
    std::lock_guard lk(lock_);
    if (auto it = stubs_.find(hash); it != stubs_.end()) {
        // Another thread made the stub first (and its words may have been dropped since): hand it ours. They sit in info when the
        // reflection worked, in words otherwise.
        std::vector<uint32_t>& mine = info ? info->spirv : words;
        if (it->second && it->second->spirv.empty() && !mine.empty()) it->second->spirv = std::move(mine);
        return it->second.get();
    }
    return stubs_.emplace(hash, std::move(info)).first->second.get();
}

void ShaderCache::release_stub_spirv(const ShaderInfo* s) {
    std::lock_guard lk(lock_);
    auto it = stubs_.find(s->hash);
    if (it == stubs_.end() || it->second.get() != s) return;
    std::vector<uint32_t>().swap(it->second->spirv);
}

const ShaderInfo* ShaderCache::find_by_hash(const std::string& hash) {
    std::lock_guard lk(lock_);
    auto it = by_hash_.find(hash);
    return it == by_hash_.end() ? nullptr : it->second.get();
}

size_t ShaderCache::loaded() const {
    std::lock_guard lk(lock_);
    return loaded_;
}

size_t ShaderCache::missing() const {
    std::lock_guard lk(lock_);
    return missing_;
}

bool ShaderCache::is_open() const {
    std::lock_guard lk(lock_);
    return open_;
}

const std::string& ShaderCache::dir() const { return dir_; }

// The folder of the running executable, or "" when it cannot be read.
static std::string exe_folder() {
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return {};
    return std::filesystem::path(buf).parent_path().string();
}

namespace {
// Gauges (nxrt/gauge.h) for the process cache. Declared after the cache in shader_cache(), so it is destroyed before the cache.
struct ShaderCacheGauges {
    explicit ShaderCacheGauges(const ShaderCache& c) {
        nxrt::gauge_register("nvn.shaders.loaded", [&c] { return static_cast<int64_t>(c.loaded()); });
        nxrt::gauge_register("nvn.shaders.missing", [&c] { return static_cast<int64_t>(c.missing()); });
    }
    ~ShaderCacheGauges() {
        nxrt::gauge_unregister("nvn.shaders.loaded");
        nxrt::gauge_unregister("nvn.shaders.missing");
    }
};
}  // namespace

ShaderCache& shader_cache() {
    static ShaderCache cache;
    static ShaderCacheGauges gauges(cache);
    static std::once_flag once;
    std::call_once(once, [] {
        std::string configured = nxrt::config_get_str("nvn", "shader_cache", "");
        if (!configured.empty()) {
            cache.open(configured);  // logs its own failure
            return;
        }
        // The working directory first, then the folder of the executable.
        std::vector<std::string> candidates = {"./shader_cache/spirv"};
        if (std::string exe = exe_folder(); !exe.empty()) candidates.push_back(exe + "/shader_cache/spirv");
        for (const std::string& dir : candidates) {
            std::error_code ec;
            if (!std::filesystem::exists(dir + "/results.tsv", ec)) continue;
            if (cache.open(dir)) return;
        }
        NXLOG_W("nvn", "no shader cache: set [nvn] shader_cache (NXRT_NVN_SHADER_CACHE) to a folder with results.tsv");
    });
    return cache;
}

}  // namespace nxrt::nvn::be
