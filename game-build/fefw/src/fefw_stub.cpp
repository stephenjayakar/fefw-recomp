// SPDX-License-Identifier: GPL-3.0-or-later
// fefw.exe of the release package: built on the player's PC (tools/build/fefw_build.py, bundled Zig) from this file and the six
// libraries lifted out of their own dump. All it does is hand the lifted module tables to the runtime in fefw_rt.dll (fefw_rt_api.h).
#include <stdint.h>
#include "fefw_rt_api.h"

// The processor features this build of the game code needs: fefw_build.py passes its variant's set (-DFEFW_CPU_NEEDS=...).
#ifndef FEFW_CPU_NEEDS
#define FEFW_CPU_NEEDS FEFW_CPU_NEEDS_FMA_BUILD
#endif

extern "C" {
extern const struct NxModule nxmod_main, nxmod_sdk, nxmod_subsdk0, nxmod_subsdk1, nxmod_subsdk2, nxmod_rtld;
extern const uint32_t nxmod_main_abi, nxmod_sdk_abi, nxmod_subsdk0_abi, nxmod_subsdk1_abi, nxmod_subsdk2_abi, nxmod_rtld_abi;
}

#ifdef FEFW_PGO_GEN
// An instrumented build for a profile-guided one (fefw_build.py --pgo-gen): the game ends with TerminateProcess, so the profile
// runtime's exit hook never runs; write the counters to LLVM_PROFILE_FILE every 15 seconds instead (each write replaces the last).
#include <windows.h>
extern "C" int __llvm_profile_write_file(void);
static DWORD WINAPI pgo_writer(void*) {
    for (;;) {
        Sleep(15000);
        __llvm_profile_write_file();
    }
}
#endif

// The manifest (res/fefw.manifest) sets the UTF-8 code page, so argv is UTF-8 as the runtime expects.
int main(int argc, char** argv) {
#ifdef FEFW_PGO_GEN
    CloseHandle(CreateThread(nullptr, 0, pgo_writer, nullptr, 0, nullptr));
#endif
    static const FefwRtModules mods = {
        sizeof(FefwRtModules), FEFW_RT_API,
        &nxmod_rtld, &nxmod_main, &nxmod_sdk, &nxmod_subsdk0, &nxmod_subsdk1, &nxmod_subsdk2,
        &nxmod_rtld_abi, &nxmod_main_abi, &nxmod_sdk_abi, &nxmod_subsdk0_abi, &nxmod_subsdk1_abi, &nxmod_subsdk2_abi,
        FEFW_CPU_NEEDS,
    };
    return fefw_rt_main(argc, argv, &mods);
}
