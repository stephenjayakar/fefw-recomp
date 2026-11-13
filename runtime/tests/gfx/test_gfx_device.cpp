// SPDX-License-Identifier: GPL-3.0-or-later
// Vulkan device layer: GPU choice (default discrete, override honoured), image-format checks before creation, validation as an
// oracle (it really reports), buffers and one-shot submits, leak-free create/destroy.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "nxrt/gfx.h"
#include "nxrt/log.h"
#include "test_framework.h"
NXTEST_GROUP(gfx_device)

using namespace nxrt::gfx;

namespace {

std::unique_ptr<Device> make_device(const char* gpu = "", int validation = 1) {
    DeviceDesc d;
    d.validation = validation;
    d.headless = 1;
    d.gpu = gpu;
    std::string err;
    auto dev = Device::create(d, &err);
    if (!dev) std::printf("    device creation failed: %s\n", err.c_str());
    return dev;
}

bool has_type(const Device& d, int type) {
    for (auto& g : d.gpus())
        if (g.type == type && g.can_present) return true;
    return false;
}

struct EnvOverride {
    std::string name;
    EnvOverride(const char* n, const char* v) : name(n) { _putenv_s(n, v); }
    ~EnvOverride() { _putenv_s(name.c_str(), ""); }
};

}  // namespace

NXTEST(gpu_rule_on_fake_lists) {
    const GpuCandidate igpu{"Intel(R) UHD Graphics 770", 1, 32ull << 30};
    const GpuCandidate dgpu{"NVIDIA GeForce RTX 3080", 2, 10ull << 30};
    const GpuCandidate dgpu2{"AMD Radeon RX 7900", 2, 20ull << 30};
    const GpuCandidate cpu{"llvmpipe", 4, 64ull << 30};
    bool matched = false;

    // Wonder's Windows port picked the first device, the iGPU; the rule must not depend on the order
    CHECK_EQ(pick_gpu({igpu, dgpu}, "", &matched), 1u);
    CHECK(matched);
    CHECK_EQ(pick_gpu({dgpu, igpu}, nullptr, &matched), 0u);
    CHECK_EQ(pick_gpu({cpu, igpu, dgpu}, "", nullptr), 2u);
    CHECK_EQ(pick_gpu({cpu, igpu}, "", nullptr), 1u);                 // no discrete: the integrated one, not the CPU
    CHECK_EQ(pick_gpu({igpu, dgpu, dgpu2}, "", nullptr), 2u);         // two discrete: more video memory
    CHECK_EQ(pick_gpu({}, "", nullptr), 0u);

    // overrides
    CHECK_EQ(pick_gpu({dgpu, igpu}, "integrated", &matched), 1u);
    CHECK(matched);
    CHECK_EQ(pick_gpu({igpu, dgpu}, "discrete", &matched), 1u);
    CHECK_EQ(pick_gpu({igpu, dgpu}, "0", &matched), 0u);
    CHECK_EQ(pick_gpu({igpu, dgpu}, "1", &matched), 1u);
    CHECK_EQ(pick_gpu({igpu, dgpu}, "uhd", &matched), 0u);
    CHECK_EQ(pick_gpu({igpu, dgpu}, "RTX 3080", &matched), 1u);
    CHECK_EQ(pick_gpu({igpu, dgpu}, "3080", &matched), 1u);          // more than two digits: a name part, not an index
    CHECK(matched);

    // an override that matches nothing falls back to the default and says so
    CHECK_EQ(pick_gpu({igpu, dgpu}, "voodoo", &matched), 1u);
    CHECK(!matched);
    CHECK_EQ(pick_gpu({igpu, dgpu}, "7", &matched), 1u);
    CHECK(!matched);
    CHECK_EQ(pick_gpu({igpu}, "discrete", &matched), 0u);
    CHECK(!matched);
}

NXTEST(default_device_is_the_discrete_gpu) {
    auto dev = make_device();
    REQUIRE(dev != nullptr);
    for (auto& g : dev->gpus())
        std::printf("    GPU %u: %s, type %d, %.1f GiB, usable %d%s\n", g.index, g.name.c_str(), g.type,
                    static_cast<double>(g.vram_bytes) / (1 << 30), g.can_present ? 1 : 0, g.index == dev->gpu().index ? "  <- chosen" : "");
    if (!has_type(*dev, 2)) {
        nxtest::skip("no discrete GPU on this machine");
        return;
    }
    CHECK_EQ(dev->gpu().type, 2);
    CHECK(dev->gpu_override_matched());
    CHECK_EQ(std::string(dev->properties().deviceName), dev->gpu().name);
}

NXTEST(override_picks_the_integrated_gpu_and_the_scene_style_device_works_on_it) {
    {
        auto probe = make_device();
        REQUIRE(probe != nullptr);
        if (!has_type(*probe, 1) || !has_type(*probe, 2)) {
            nxtest::skip("needs one integrated and one discrete GPU");
            return;
        }
    }
    auto dev = make_device("integrated");
    REQUIRE(dev != nullptr);
    CHECK_EQ(dev->gpu().type, 1);
    CHECK(dev->gpu_override_matched());
    // the device on the iGPU is a working device: a one-shot submit goes through and validation stays clean
    bool ran = false;
    CHECK(dev->submit_once([&](VkCommandBuffer) { ran = true; }));
    CHECK(ran);
    CHECK_EQ(dev->validation_stats().errors, 0ull);

    auto by_name = make_device("Intel");
    REQUIRE(by_name != nullptr);
    CHECK_EQ(by_name->gpu().type, 1);
    auto by_index = make_device(std::to_string(dev->gpu().index).c_str());
    REQUIRE(by_index != nullptr);
    CHECK_EQ(by_index->gpu().index, dev->gpu().index);
    auto discrete = make_device("discrete");
    REQUIRE(discrete != nullptr);
    CHECK_EQ(discrete->gpu().type, 2);
}

NXTEST(override_from_the_environment_and_a_bad_override) {
    {
        auto probe = make_device();
        REQUIRE(probe != nullptr);
        if (!has_type(*probe, 1)) {
            nxtest::skip("no integrated GPU to select");
            return;
        }
    }
    {
        EnvOverride e("NXRT_GFX_GPU", "integrated");
        auto dev = make_device("");   // empty override in the descriptor: the configuration decides
        REQUIRE(dev != nullptr);
        CHECK_EQ(dev->gpu().type, 1);
    }
    auto bad = make_device("no-such-gpu-xyz");
    REQUIRE(bad != nullptr);
    CHECK(!bad->gpu_override_matched());
    if (has_type(*bad, 2)) CHECK_EQ(bad->gpu().type, 2);   // falls back to the default rule
}

NXTEST(image_formats_are_checked_before_vkCreateImage) {
    auto dev = make_device();
    REQUIRE(dev != nullptr);
    std::printf("    %s: BC %d, ASTC-LDR %d, ETC2 %d\n", dev->properties().deviceName, dev->features().bc, dev->features().astc_ldr, dev->features().etc2);

    ImageDesc astc;
    astc.format = VK_FORMAT_ASTC_4x4_UNORM_BLOCK;   // what the console textures use; desktop NVIDIA has no ASTC
    astc.width = astc.height = 64;
    astc.name = "astc probe";
    Image img;
    std::string why;
    const FormatCheckStats before = dev->format_check_stats();
    const VkResult r = dev->create_image(astc, &img, &why);
    if (dev->features().astc_ldr) {
        CHECK_EQ(r, VK_SUCCESS);
        dev->destroy_image(img);
    } else {
        std::printf("    ASTC refused as expected: %s\n", why.c_str());
        CHECK_EQ(r, VK_ERROR_FORMAT_NOT_SUPPORTED);
        CHECK(!img);
        CHECK(!why.empty());
        CHECK_EQ(dev->format_check_stats().rejected, before.rejected + 1);
    }

    ImageDesc bc7;
    bc7.format = VK_FORMAT_BC7_UNORM_BLOCK;
    bc7.width = bc7.height = 256;
    bc7.mip_levels = 9;
    bc7.name = "bc7 probe";
    Image b;
    if (dev->features().bc) {
        CHECK_EQ(dev->create_image(bc7, &b, &why), VK_SUCCESS);
        CHECK(b.view != VK_NULL_HANDLE);
        dev->destroy_image(b);
    } else {
        CHECK_EQ(dev->create_image(bc7, &b, &why), VK_ERROR_FORMAT_NOT_SUPPORTED);
    }

    // limits: too big, too many mips, too many layers, bad sample count, a usage the format cannot have
    ImageDesc big;
    big.format = VK_FORMAT_R8G8B8A8_UNORM;
    big.width = 1u << 30;
    big.height = 1;
    CHECK(!dev->image_supported(big, &why));
    ImageDesc mips = big;
    mips.width = mips.height = 16;
    mips.mip_levels = 12;
    CHECK(!dev->image_supported(mips, &why));
    ImageDesc layers = mips;
    layers.mip_levels = 1;
    layers.array_layers = 1u << 20;
    CHECK(!dev->image_supported(layers, &why));
    ImageDesc samples = mips;
    samples.mip_levels = 1;
    samples.samples = VK_SAMPLE_COUNT_64_BIT;
    samples.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    CHECK(!dev->image_supported(samples, &why));
    ImageDesc storage_astc = astc;
    storage_astc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    CHECK(!dev->image_supported(storage_astc, &why));
    ImageDesc zero = mips;
    zero.width = 0;
    CHECK(!dev->image_supported(zero, &why));
    ImageDesc good = mips;
    good.mip_levels = 1;
    CHECK(dev->image_supported(good, &why));

    // the answers are cached: the same question again costs no driver call
    const FormatCheckStats s0 = dev->format_check_stats();
    for (int i = 0; i < 10; ++i) CHECK(dev->image_supported(good, &why));
    const FormatCheckStats s1 = dev->format_check_stats();
    CHECK_EQ(s1.checks, s0.checks);
    CHECK_EQ(s1.cache_hits, s0.cache_hits + 10);
    CHECK_EQ(dev->validation_stats().errors, 0ull);
}

NXTEST(validation_layer_really_reports_errors) {
    // The "zero validation errors" oracle is only worth something when the layer is live: provoke one error and see it counted.
    auto dev = make_device();
    REQUIRE(dev != nullptr);
    if (!dev->validation_enabled()) {
        CHECK_MSG(false, "the Khronos validation layer is not installed: the validation oracle cannot run (install the Vulkan SDK)");
        return;
    }
    dev->validation_reset();
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = 0;   // VUID-VkBufferCreateInfo-size-00912
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkBuffer b = VK_NULL_HANDLE;
    vkCreateBuffer(dev->device(), &bi, nullptr, &b);
    if (b) vkDestroyBuffer(dev->device(), b, nullptr);
    const ValidationStats v = dev->validation_stats();
    std::printf("    provoked: %llu error(s): %s\n", static_cast<unsigned long long>(v.errors), v.first_error.substr(0, 90).c_str());
    CHECK(v.errors >= 1);
    dev->validation_reset();
    CHECK_EQ(dev->validation_stats().errors, 0ull);
}

NXTEST(buffers_and_one_shot_submit_round_trip) {
    auto dev = make_device();
    REQUIRE(dev != nullptr);
    Buffer up, down, gpu;
    constexpr VkDeviceSize kSize = 64 * 1024;
    REQUIRE(dev->create_buffer(kSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryUse::Upload, &up, "up") == VK_SUCCESS);
    REQUIRE(dev->create_buffer(kSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUse::Readback, &down, "down") == VK_SUCCESS);
    REQUIRE(dev->create_buffer(kSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUse::GpuOnly, &gpu, "gpu") == VK_SUCCESS);
    CHECK(up.mapped != nullptr);
    CHECK(down.mapped != nullptr);
    CHECK(gpu.mapped == nullptr);
    auto* p = static_cast<uint8_t*>(up.mapped);
    for (VkDeviceSize i = 0; i < kSize; ++i) p[i] = static_cast<uint8_t>(i * 7 + (i >> 8));
    dev->flush_buffer(up);
    CHECK(dev->submit_once([&](VkCommandBuffer cmd) {
        VkBufferCopy c = {0, 0, kSize};
        vkCmdCopyBuffer(cmd, up.buffer, gpu.buffer, 1, &c);
        VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
        vkCmdCopyBuffer(cmd, gpu.buffer, down.buffer, 1, &c);
    }));
    dev->invalidate_buffer(down);
    CHECK_EQ(std::memcmp(up.mapped, down.mapped, kSize), 0);
    dev->destroy_buffer(up);
    dev->destroy_buffer(down);
    dev->destroy_buffer(gpu);
    CHECK(!up);
    CHECK_EQ(dev->memory_stats().allocations, 0ull);
    CHECK_EQ(dev->validation_stats().errors, 0ull);
}

NXTEST(create_and_destroy_many_images_does_not_leak) {
    auto dev = make_device();
    REQUIRE(dev != nullptr);
    const Device::MemoryStats before = dev->memory_stats();
    for (int i = 0; i < 300; ++i) {
        ImageDesc d;
        d.format = (i % 3 == 0) ? VK_FORMAT_R8G8B8A8_UNORM : (i % 3 == 1 ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R16G16B16A16_SFLOAT);
        d.width = 32 + static_cast<uint32_t>(i % 64);
        d.height = 16 + static_cast<uint32_t>(i % 32);
        d.mip_levels = 1 + static_cast<uint32_t>(i % 4);
        d.array_layers = 1 + static_cast<uint32_t>(i % 3);
        d.name = "leak probe";
        Image img;
        std::string why;
        REQUIRE(dev->create_image(d, &img, &why) == VK_SUCCESS);
        dev->destroy_image(img);
    }
    const Device::MemoryStats after = dev->memory_stats();
    CHECK_EQ(after.allocations, before.allocations);
    CHECK_EQ(after.bytes, before.bytes);
    CHECK_EQ(dev->validation_stats().errors, 0ull);
}

// Measurement (no checks beyond success): the time of each image creation for 600 BC7 textures of 2048x2048 with mips (3.3 GB): new memory
// blocks show as the slow ones. Prints the distribution and the slowest creations.
NXTEST(image_creation_times_bc7_area_load) {
    auto dev = make_device("", 0);
    REQUIRE(dev != nullptr);
    std::vector<Image> imgs;
    std::vector<double> ms;
    for (int i = 0; i < 600; ++i) {
        ImageDesc d;
        d.format = VK_FORMAT_BC7_SRGB_BLOCK;
        d.width = 2048;
        d.height = 2048;
        d.mip_levels = 12;
        d.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        d.name = "timing probe";
        Image img;
        std::string why;
        const auto t0 = std::chrono::steady_clock::now();
        const VkResult r = dev->create_image(d, &img, &why);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        if (r != VK_SUCCESS) break;
        imgs.push_back(img);
    }
    std::vector<double> s = ms;
    std::sort(s.begin(), s.end());
    double tot = 0;
    for (double v : s) tot += v;
    std::printf("    %zu images: total %.1f ms, median %.3f, p90 %.3f, p99 %.3f, max %.1f ms; slow (>1 ms):", ms.size(), tot, s[s.size() / 2],
                s[s.size() * 9 / 10], s[s.size() * 99 / 100], s.back());
    for (size_t i = 0; i < ms.size(); ++i)
        if (ms[i] > 1.0) std::printf(" #%zu %.1f", i, ms[i]);
    std::printf("\n");
    for (Image& img : imgs) dev->destroy_image(img);
}

NXTEST(repeated_device_create_and_destroy) {
    for (int i = 0; i < 5; ++i) {
        auto dev = make_device();
        REQUIRE(dev != nullptr);
        CHECK_EQ(dev->validation_stats().errors, 0ull);
    }
}
