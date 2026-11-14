// SPDX-License-Identifier: GPL-3.0-or-later
// Adapted from SuperMarioWonderRecomp runtime/src/nvn/nvn_shaders.{h,cpp} (MIT, Copyright (c) 2026 BelmanteGu); changes: namespace, cache lookup.
//
// NVN shader stages -> the SPIR-V translated offline (SuperMarioWonderRecomp docs/nvn/TRANSLATOR.md, "Runtime side").
//
// Every Maxwell code block of a game was translated ahead of time by Ryujinx's translator into <cache dir>/<hash>.spv,
// <hash> = the first 16 hex digits of the SHA-1 of the whole code block (the control section's +0x704 bytes). The
// harness's record of each shader sits in <cache dir>/results.tsv. At nvnProgramSetShaders time the runtime hashes the
// block the game hands in, loads that file and needs to know what each descriptor of it stands for in NVN terms: which
// uniform block, which texture binding, which storage buffer. That is what this file provides.
//
// Everything above "the runtime cache" is plain C++ with no dependency on the rest of the runtime.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nxrt::nvn::be {

// ---- hashing ----------------------------------------------------------------------------------------------------

// SHA-1 (FIPS 180-4). Only a cache key here, not a security measure.
class Sha1 {
public:
    Sha1();
    void update(const void* data, size_t size);
    void finish(uint8_t digest[20]);  // the object is spent afterwards

private:
    void block(const uint8_t* p);
    uint32_t h_[5];
    uint64_t bytes_ = 0;
    uint8_t buf_[64];
    size_t used_ = 0;
};

std::string sha1_hex(const void* data, size_t size);                 // 40 lowercase hex digits
std::string shader_hash(const uint8_t* code, uint32_t size);          // the first 16 of them: the .spv file name

// ---- the NVN control section ----------------------------------------------------------------------------------------

constexpr uint32_t kShaderControlMagic = 0x98761234;

struct ShaderControl {
    bool valid = false;           // magic present
    uint32_t program_size = 0;    // +0x6f8: SPH + instructions, from program_start
    uint32_t const_size = 0;      // +0x6fc: constant data (c[1]) bytes, 0 = none
    uint32_t const_offset = 0;    // +0x700: its offset in the code block (0x100-aligned)
    uint32_t code_size = 0;       // +0x704: the whole code block, the bytes the hash covers
    uint32_t program_start = 0;   // +0x708: 0x30 graphics (SPH there), 0x100 compute
    uint32_t local_size[3] = {};  // +0x748: compute work-group size
    uint32_t shared_size = 0;     // +0x754: compute shared memory bytes (0 = none)
    bool nvn2 = false;            // Switch 2 control section (version word 201 at +4, 0xd8 bytes, Ampere SASS)
    int stage = -1;               // NVN2 only: u16 +0x56 (0 vs, 1 ps, 2 gs, 3 tcs, 4 tes, 5 cs)
    // NVN2 compute only: the slot counts of the program's c[0] resource table (bytes +0xb1 uniform blocks, +0xb2 storage
    // buffers, +0xb3 textures, +0xb6 images). See apply_nvn2_compute_table.
    uint8_t c0_ubos = 0, c0_ssbos = 0, c0_textures = 0, c0_images = 0;
    // NVN2 compute only: byte +0xa9, the 32-bit words of push constant data between the 0x10-byte header and the table (0 or 28 in
    // Fortune's Weave), and u16 +0xba, the bytes of header, push data and table together.
    uint8_t c0_push_words = 0;
    uint16_t c0_table_size = 0;
};

ShaderControl read_control(const uint8_t* control);

// ---- SPIR-V reflection ----------------------------------------------------------------------------------------------

enum class ShaderStage : uint8_t { Unknown, Vertex, Fragment, Compute, Geometry, TessControl, TessEval };

// Values are VkDescriptorType's (checked against the Vulkan headers in the runtime build).
enum class DescriptorType : uint32_t {
    Sampler = 0,
    CombinedImageSampler = 1,
    SampledImage = 2,
    StorageImage = 3,
    UniformTexelBuffer = 4,
    StorageTexelBuffer = 5,
    UniformBuffer = 6,
    StorageBuffer = 7,
};

enum class ScalarType : uint8_t { Other, Float, Sint, Uint, Bool };

// OpTypeImage of a texture or image binding. `dim` is SPIR-V's Dim: 0 1D, 1 2D, 2 3D, 3 Cube, 5 Buffer.
struct ImageShape {
    uint8_t dim = 0;
    bool depth = false, arrayed = false, ms = false;
    uint8_t sampled = 0;   // 1 sampled, 2 storage
    uint32_t format = 0;   // SPIR-V ImageFormat (0 Unknown)
    ScalarType component = ScalarType::Other;
};

// What a binding is in NVN terms. The translator's scheme (TRANSLATOR.md, "Where NVN puts resources"):
//   c[0] driver constants, c[1] the block's constant data, c[2] texture/image handles, c[3+n] uniform block n.
enum class NvnSource : uint8_t {
    Unknown,
    SupportBuffer,    // set 0 binding 0: the translator's own buffer (struct SupportBuffer below), filled by us
    DriverConstants,  // c[0]: NVN driver constants (compute reads its buffer descriptor table there)
    ConstantData,     // c[1]: this block's constant data (ShaderInfo::constants / control const_offset)
    UniformBlock,     // c[3+n]: what the game bound with nvnCommandBufferBindUniformBuffer(stage, n)
    StorageBuffer,    // the address in c[0] descriptor `c0_offset`: nvnCommandBufferBindStorageBuffer(stage, n)
    UniformAsStorage, // compute: the c[0] descriptor of uniform block n, read through a storage buffer
    SamplerBinding,   // texture whose handle is c[2] byte 0x20 + 8n: nvnCommandBufferBindTexture(stage, n)
    ImageBinding,     // image whose handle is c[2] byte 0x120 + 8n: nvnCommandBufferBindImage(stage, n) (inferred); on a texture binding of an
                      // NVN2 compute shader: that image sampled through its handle (apply_nvn2_compute_table)
    HandleInUniform,  // texture whose handle the game wrote into uniform block `index` (c[bank]), at byte word * 4
    PushConstants,    // c[24 + n] (NVN2 graphics): the data of nvnCommandBufferPushGraphicsConstants slot n = `index`
    TexturePoolArray, // the texture pool as an array indexed by texture id: a handle the translator could not pin to one binding
    SamplerPoolArray, // the sampler pool as an array indexed by sampler id (handle >> 20)
};

struct NvnMeaning {
    NvnSource source = NvnSource::Unknown;
    int index = -1;      // the NVN binding n (uniform block, storage buffer, sampler, image)
    int bank = -1;       // constant bank c[bank]: the buffer itself (uniforms) or the bank holding the handle
    int word = -1;       // textures/images: 32-bit word of c[bank] holding the handle
    int sb_slot = -1;    // storage buffers: the translator's slot (order found), from the name "<stage>_s<slot>"
    int c0_offset = -1;  // storage buffers: byte offset of the descriptor in c[0]; -1 = not known (see apply_harness)
    bool default_bank = false;  // texture handle bank given by the translator as "the texture bank" (-1), i.e. c[2]
};

struct DescriptorBinding {
    uint32_t set = 0, binding = 0;
    uint32_t declared_set = 0, declared_binding = 0;   // as translated (results.tsv); pool arrays are moved to kPoolSet (move_pool_arrays)
    DescriptorType type = DescriptorType::UniformBuffer;
    uint32_t count = 1;  // array length; 0 = runtime array
    std::string name;    // OpName, e.g. "fp_c3", "fp_t_tcb_8", "fp_t_cb7_20", "cp_s0", "cp_i_cb2_4A_r32ui"
    ImageShape image;    // texture and image bindings
    NvnMeaning nvn;
    // Some instruction references the variable. The translator declares c[0] (driver constants) in most shaders
    // without using it; the harness's record lists only used resources (checked on the whole cache), so
    // describe_like_harness skips the unused ones.
    bool used = false;
};

struct ShaderIo {
    uint32_t location = 0, component = 0;
    ScalarType type = ScalarType::Other;
    uint32_t components = 0;  // 1..4
    uint32_t array = 1;       // array length (1 = not an array)
};

struct ShaderReflection {
    ShaderStage stage = ShaderStage::Unknown;
    std::string entry;                             // entry point name ("main")
    uint32_t local_size[3] = {0, 0, 0};            // OpExecutionMode LocalSize (compute)
    std::vector<DescriptorBinding> bindings;       // sorted by (set, binding)
    std::vector<ShaderIo> inputs, outputs;         // Location-decorated Input/Output variables (not built-ins)
    std::vector<uint32_t> builtins_in, builtins_out;  // SPIR-V BuiltIn values (on variables or block members)
    std::vector<uint32_t> builtins_used_in, builtins_used_out;  // built-in block members an access chain reaches (used, not only declared)
    bool uses_support_buffer = false;

    const DescriptorBinding* find(uint32_t set, uint32_t binding) const;
    // Vertex: the attribute locations the shader reads. Fragment: the render targets it writes.
    const std::vector<ShaderIo>& vertex_inputs() const { return inputs; }
    const std::vector<ShaderIo>& fragment_outputs() const { return outputs; }
};

// Parses a SPIR-V module (`count` 32-bit words). False with a reason in `error` if the module is malformed.
bool reflect_spirv(const uint32_t* words, size_t count, ShaderReflection& out, std::string* error = nullptr);

// ---- the harness's own record (results.tsv, column "resources") -----------------------------------------------------
// The SPIR-V carries everything except where a storage buffer's address came from: the translator names storage
// buffers by slot ("vp_s0"), not by their c[0] offset. The harness printed that offset in results.tsv, so the
// runtime reads it from there. Format (tools/maxwell/spirv_harness/Program.cs, DescribeResources):
//   cb<slot>=s<set>b<binding>                      constant buffer c[slot]
//   sb[c<slot>+0x<off>]=s<set>b<binding>           storage buffer, address at c[slot] 32-bit word <off>
//   tex<type>[c<slot>:<word>(x<n>)]=s<set>b<binding>  texture, handle at word <word> of c[slot] (-1 = c[2])
//   img<type>[c<slot>:<word>(x<n>)]=s<set>b<binding>  image, likewise

struct HarnessResource {
    enum Kind : uint8_t { Uniform, Storage, Texture, Image } kind = Uniform;
    int set = -1, binding = -1;
    int bank = -1;           // c[bank] as printed (-1 = the default texture bank)
    int word = -1;           // storage: descriptor word in c[bank]; texture/image: handle word
    int array_length = 1;
    std::string type;        // Ryujinx SamplerType text, e.g. "Texture2D|Array|Shadow"
};

bool parse_harness_resources(std::string_view text, std::vector<HarnessResource>& out);
// results.tsv text -> hash -> resources column (every line; the cache indexes lines itself instead).
size_t parse_results_tsv(std::string_view text, std::unordered_map<std::string, std::string>& out);
// Fills the c[0] offset of each storage buffer (and anything the names did not give). False if a harness entry has
// no matching SPIR-V binding.
bool apply_harness(ShaderReflection& refl, const std::vector<HarnessResource>& resources);

// The "resources" text the harness would print for this reflection (after apply_harness); the test compares it.
std::string describe_like_harness(const ShaderReflection& refl);
std::string sampler_type_text(const ImageShape& s);  // Ryujinx SamplerType text of an image shape

// ---- texture and sampler pools as descriptor arrays ---------------------------------------------------------------------
// A handle the translator cannot pin to one constant-buffer word indexes the pools (harness `tex<Type>[c0:1073741824x65536]`, SPIR-V
// "<stage>_t2d_cb0_40000000" / "_tcube_" / "_s_" in set 2 from binding 640, texture id = handle & 0xfffff, sampler id = handle >> 20, both
// clamped to the array). The runtime moves these arrays to a set of their own, one per pool pair, shared by every program and kept up to
// date from the pools' registrations (nvn_pipeline.cpp, pool_set), at fixed bindings by image shape. Other shapes are refused at load.
constexpr uint32_t kPoolSet = 4;
constexpr uint32_t kPoolHandleWord = 0x40000000;     // the translator's word for "the pool" (Ryujinx TextureHandleType.Direct marker)
constexpr uint32_t kPoolTextureCount = 65536, kPoolSamplerCount = 4096;   // the arrays the harness declares (--texture-pool / --sampler-pool)
enum PoolBinding : uint32_t { kPoolSamplers = 0, kPoolTex2D = 1, kPoolTexCube = 2, kPoolBindingCount = 3 };
int pool_binding_of(const DescriptorBinding& b);    // its binding in kPoolSet; -1 when the shape has none
// Moves the pool arrays of a module to kPoolSet: rewrites their DescriptorSet / Binding decorations, found by the translator's numbers
// (declared_set / declared_binding, so words read again can be moved with a reflection moved before), and the reflection. False with a
// reason when an array has a shape or length the set does not have.
bool move_pool_arrays(std::vector<uint32_t>& spirv, ShaderReflection& refl, std::string* error = nullptr);

// ---- Ryujinx's support buffer ---------------------------------------------------------------------------------------
// Set 0, binding 0 of every stage (std140), declared by every translated shader. Layout copied from Ryujinx
// (Ryujinx.Graphics.Shader/SupportBuffer.cs; filling rules from Ryujinx.Graphics.Gpu/Memory/SupportBufferUpdater.cs),
// MIT License, Copyright (c) Ryujinx Team and Contributors. Every field is a 16-byte vec4 slot.
//
// With the answers the harness baked (TRANSLATOR.md, "What only the runtime knows") the shaders read only
// render_scale (texture size queries, texel fetches, gl_FragCoord.xy) and frag_scale_count (vertex shaders index their
// scales past the fragment ones): with every scale 1 those reads are identities. viewport_* are read only by variants
// with viewport transform disabled or Y negate, is_bgra only when the host has no BGRA formats (Vulkan has), alpha_test
// never (the alpha test is baked into the shader from the graphics state).
struct SupportBuffer {
    int32_t alpha_test[4];        // 0x000 FragmentAlphaTest (declared, not read)
    int32_t is_bgra[8][4];        // 0x010 FragmentIsBgra: x = 1 if render target i is BGRA (R/B swapped on output)
    float viewport_inverse[4];    // 0x090 2*scale/width, 2*scale/height, 1, 1 if viewport transform disabled else 0
    float viewport_size[4];       // 0x0a0 width, |height|, 1, 0 (gl_FragCoord.y flip under Y negate)
    int32_t frag_scale_count[4];  // 0x0b0 x = number of fragment texture scales (vertex scales follow them)
    float render_scale[73][4];    // 0x0c0 x only: [0] render target, [1..64] textures, [65..72] images
    int32_t tfe_offset[4];        // 0x550 transform feedback emulation (unused: no transform feedback)
    int32_t tfe_vertex_count[4];  // 0x560
};
static_assert(sizeof(SupportBuffer) == 0x570, "Ryujinx SupportBuffer is 1392 bytes");
static_assert(offsetof(SupportBuffer, is_bgra) == 0x10 && offsetof(SupportBuffer, viewport_inverse) == 0x90 &&
                  offsetof(SupportBuffer, viewport_size) == 0xa0 && offsetof(SupportBuffer, frag_scale_count) == 0xb0 &&
                  offsetof(SupportBuffer, render_scale) == 0xc0 && offsetof(SupportBuffer, tfe_offset) == 0x550 &&
                  offsetof(SupportBuffer, tfe_vertex_count) == 0x560,
              "offsets as in the SPIR-V's OpMemberDecorate");

SupportBuffer support_buffer_defaults();  // render scale 1 everywhere, no BGRA, no alpha test, zero viewport
// What Ryujinx writes for viewport 0 (SetViewportTransformDisable / SetViewportSize), with render scale 1.
void support_buffer_set_viewport(SupportBuffer& sb, float width, float height, bool transform_disabled = false);

// ---- one stage ------------------------------------------------------------------------------------------------------

struct ShaderInfo {
    std::string hash;
    ShaderStage stage = ShaderStage::Unknown;  // from the SPIR-V execution model
    std::vector<uint32_t> spirv;
    ShaderReflection refl;
    ShaderControl control;
    // c[1]: the constant data at code block + control.const_offset. Copied because the game may reuse the pool. The
    // SPIR-V declares every c[k] as vec4[4096]; bind at least const_size rounded up to 16.
    std::vector<uint8_t> constants;
    uint32_t local_size[3] = {1, 1, 1};  // compute: from control +0x748 (equals the SPIR-V's LocalSize)
    // Not computed by this port (the original fills them from nvn_spirv_patch.cpp, which is not ported). masks_known
    // stays false and the replay must compute them on use.
    bool masks_known = false;
    uint64_t dynamic_uniforms = 0, written_storage = 0;
    // A generated stage that reads push constants (the pass-through tessellation control shader: its levels), bytes from offset 0; the
    // program's pipeline layout then has that range for this stage. 0 = none (every translated shader).
    uint32_t push_bytes = 0;
};

// Builds a ShaderInfo from a code block, its control section and the SPIR-V words (no I/O). `resources` = the
// block's results.tsv resources column, or null (storage buffer c[0] offsets then stay -1).
// `pre_refl`: the reflection of these words with the resources record already applied (the boot preload made it); then neither
// is done again and `resources` is not read.
bool make_shader_info(const uint8_t* code, const uint8_t* control, std::vector<uint32_t> spirv,
                      const std::string* resources, ShaderInfo& out, std::string* error = nullptr,
                      const ShaderReflection* pre_refl = nullptr);

const char* stage_name(ShaderStage s);
const char* source_name(NvnSource s);

// ---- the runtime cache ----------------------------------------------------------------------------------------------

// A directory holding <hash>.spv files and results.tsv. Lookups are keyed by the hash of the code block.
class ShaderCache {
public:
    ShaderCache() = default;
    ~ShaderCache();
    ShaderCache(const ShaderCache&) = delete;
    ShaderCache& operator=(const ShaderCache&) = delete;

    // Opens `dir`: reads results.tsv once and indexes its lines by the hash in column 1. False (logged) if results.tsv
    // cannot be read or has no "resources" column. A cache opens once: a second open() returns false.
    bool open(const std::string& dir);
    // The stage whose code block is at `code` (guest = host memory) with control section `control`. Hashes the block,
    // loads <hash>.spv on first use and keeps the ShaderInfo for the life of the process (the pointer stays valid).
    // Null if there is no usable .spv (logged once per hash) or the control section is malformed.
    // Hashing costs a pass over the block: call it when the program is set, not per draw.
    const ShaderInfo* find(const uint8_t* code, const uint8_t* control);
    // A ShaderInfo that find() already loaded, or null. Never loads: a ShaderInfo needs the code block (its control
    // section and constant data come from it).
    const ShaderInfo* find_by_hash(const std::string& hash);

    // A ShaderInfo for pipeline building only (S24, the shader preparation): SPIR-V, stage and reflection from <hash>.spv and its results.tsv line,
    // no code block, no control section, no constant data. Kept apart from find(): the game's own ShaderInfos never come from here. The
    // pointer stays valid for the life of the cache; only the SPIR-V words can be dropped (release_stub_spirv) and are read again by
    // load_stub. Null if the hash has no usable .spv. Any thread.
    const ShaderInfo* load_stub(const std::string& hash);
    void release_stub_spirv(const ShaderInfo* s);   // frees the words; the caller guarantees nobody is using them (single threaded point)
    size_t loaded() const;   // ShaderInfos built so far
    size_t missing() const;  // hashes asked for that have no usable .spv (or do not reflect)
    bool is_open() const;
    const std::string& dir() const;

private:
    // No lock held while it runs: takes the preloaded words (waiting for a worker that is reading this very file) or reads the .spv
    // itself, builds the ShaderInfo, and only then inserts it under lock_ (if another thread inserted the hash first, that one wins).
    // `record`: the resources column of this block's results.tsv line, or null.
    const ShaderInfo* load(const std::string& hash, const uint8_t* code, const uint8_t* control, uint32_t code_size,
                           const std::string* record);
    // Boot preload ([nvn] shader_order, default shader_order.txt; [nvn] shader_preload_threads, default 6; [nvn] shader_preload_mb, default 1024
    // = how much prepared-but-untaken data the workers may hold): the hashes find() loaded,
    // in the order it first loaded them, are kept in the order file; the next run reads and reflects those .spv files on worker
    // threads from the moment the cache opens, ahead of the game's own program loads, which then only take the prepared words.
    struct Preloaded {
        std::vector<uint32_t> words;
        ShaderReflection refl;   // the resources record already applied
    };
    void start_preload();
    void preload_worker();
    std::unique_ptr<Preloaded> preload_one(const std::string& hash);
    void note_order_locked(const std::string& hash, std::string& flush);   // flush: lines to append, written by the caller after unlocking
    void flush_order(const std::string& lines);
    std::unique_ptr<Preloaded> take_preloaded(const std::string& hash);

    // The preload pack (<dir>.preload.pack, nvn_shaders.cpp "the preload pack"): the order file's .spv files in one file, zstd-compressed,
    // written by the runtime after a run's boot preload and read instead of the separate files by the runs after it.
    struct PackEntry {
        uint64_t offset;
        uint32_t csize, usize;
    };
    void pack_open();                                                  // open(), lock_ held
    bool pack_read(const std::string& hash, std::string& out, bool* from_mem = nullptr);   // any thread, no lock
    bool read_spv(const std::string& hash, std::string& out, int* source = nullptr);       // the pack, else <dir>/<hash>.spv; source 1 pack in memory, 2 pack file, 3 .spv file
    void pack_prefetch();                                              // own thread
    void preload_done();                                               // the last preload worker as it ends
    void pack_build(size_t missing);                                   // own thread
    void order_writer();                                               // own thread: appends new order-file lines every few seconds
    void note_load(int source, int64_t us);                            // load(): counters and the periodic line
    // [nvn] shader_pack_all: the pack holds every shader of results.tsv that has SPIR-V, the order file's first (pack_list_), so a shader the
    // game sets for the first time is never a separate file read on the game thread. [nvn] shader_pack_resident: the pack's data stays in memory
    // (pack_mem_, filled front to back by pack_prefetch; pack_mem_end_ = bytes valid), so such a load costs no disk access at all.
    std::vector<std::string> pack_list_;                               // what pack_build writes, in order (set in start_preload)
    std::unique_ptr<char[]> pack_mem_;
    std::atomic<uint64_t> pack_mem_end_{0};
    int spv_bytes_col_ = -1;                                           // column index of "spv_bytes" (-1: none; every line counts as packable)
    std::thread order_thread_;
    std::condition_variable order_cv_;
    bool order_stop_ = false;                                          // (lock_)
    // what the game's own loads (find -> load) cost, by source (preloaded words / the pack / separate .spv files)
    std::atomic<uint64_t> ld_count_{0}, ld_pre_{0}, ld_pack_{0}, ld_pack_mem_{0}, ld_files_{0}, ld_us_{0}, ld_worst_us_{0}, ld_files_us_{0};
    std::atomic<uint64_t> ld_logged_ms_{0}, ld_logged_count_{0};
    void* pack_file_ = nullptr;                                        // HANDLE of preload.pack; set in open(), then read-only
    uint64_t pack_data_end_ = 0;
    std::unordered_map<std::string, PackEntry> pack_;                  // the usable entries; set in open(), then read-only
    std::atomic<bool> pack_stop_{false};
    std::thread pack_prefetch_thread_, pack_build_thread_;
    std::atomic<int> pre_live_{0};                                     // preload workers still running
    std::atomic<uint64_t> pre_from_pack_{0}, pre_from_files_{0};       // .spv reads served by the pack / by separate files
    uint64_t pre_start_ms_ = 0;

    std::mutex pre_mu_;
    std::condition_variable pre_cv_;                                        // a worker finished a read / an entry was taken / stop
    std::unordered_set<std::string> pre_inflight_;                          // hashes a worker is reading right now (pre_mu_)
    size_t pre_bytes_ = 0;                                                  // words held in pre_ (pre_mu_)
    size_t pre_budget_ = 0;                                                 // workers pause above this
    std::mutex order_io_mu_;                                                // serialises appends to the order file
    std::unordered_map<std::string, std::unique_ptr<Preloaded>> pre_;     // prepared, not yet taken
    std::vector<std::string> pre_list_;                                     // the order file of the last run
    size_t pre_next_ = 0;                                                   // next pre_list_ entry for a worker (pre_mu_)
    bool pre_stop_ = false;
    std::vector<std::thread> pre_threads_;
    std::string order_path_;
    std::unordered_set<std::string> order_known_;                           // in the order file (lock_)
    std::string order_pending_;                                             // new lines not yet appended (lock_)

    mutable std::mutex lock_;
    std::string dir_;
    bool open_ = false;
    std::string results_;                                                   // results.tsv, read once
    int resources_col_ = -1;                                                // column index of "resources"
    std::unordered_map<std::string, std::pair<size_t, size_t>> lines_;     // hash -> (offset, length) in results_
    std::unordered_map<std::string, std::unique_ptr<ShaderInfo>> by_hash_; // null: no usable .spv (logged once)
    std::unordered_set<uintptr_t> bad_controls_;                           // malformed control pointers (logged once)
    std::unordered_map<std::string, std::unique_ptr<ShaderInfo>> stubs_;   // load_stub (see there); null: no usable .spv
    size_t loaded_ = 0, missing_ = 0;
};

// The process-wide cache. Opens itself on first use from config [nvn] shader_cache (environment NXRT_NVN_SHADER_CACHE);
// if that is empty, the first of ./shader_cache/spirv and <executable folder>/shader_cache/spirv that has a
// results.tsv. A failed open is logged once and every find returns null.
ShaderCache& shader_cache();

}  // namespace nxrt::nvn::be
