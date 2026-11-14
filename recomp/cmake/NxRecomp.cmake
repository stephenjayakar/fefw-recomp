# SPDX-License-Identifier: GPL-3.0-or-later
#
# nxrecomp_add_module(<target>
#     INPUT <module file>          NSO or static ELF (see --kind)
#     NAME <module name>           default: main        (the generated descriptor is `nxmod_<NAME>`)
#     [BASE <0x...>]               guest address the module is mapped at (default 0x7100000000)
#     [INDEX <n>]                  NxModule.index / import-id module number (default: position in the game config's module list)
#     [GAME <name>]                recomp/games/<name>/config.py (default: generic)
#     [KIND nso|nx_elf|static_elf] default: guessed from the file's magic
#     [SHARDS <n>]                 exactly n translation units c_000.cpp ... (default 16)
#     [JOBS <n>]                   lifter processes (default: all cores)
#     [JOB_POOL <pool>]            Ninja job pool of the lift step (declared by the caller in the JOB_POOLS global property)
#     [PATCHES <file>]             instruction patch list
#     [SITES <file>]               site list: patches, inline callouts, function hooks (nxrecomp/program.py load_sites)
#     [OUTDIR <dir>])              where the generated sources go (default: <build>/gen/<target>)
#
# Lifts the module with `python -m nxrecomp lift` (a build step, re-run when the input or the lifter changes) and builds
# the result as the static library <target>. Link it into the runtime: it provides `nxmod_<NAME>` (ctx.h: NxModule) and
# expects the runtime to provide the functions ctx.h declares as runtime-provided. Public include directories: recomp/include
# (nxrecomp/ctx.h) and the generated folder (funcs.h).
#
# Flags: clang-cl, C++20, no exceptions/RTTI, -fno-strict-aliasing and -ffp-contract=off (smo-recomp's: lifted FP must not be
# fused or reordered), -mcx16 (128-bit exclusives are cmpxchg16b). Functions longer than the game config's optnone_insns are
# emitted with NX_OPTNONE by the lifter itself (bounded compile time); cmake/nx_cc_launcher.py adds the time-limit fallback.

set(NXRECOMP_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH "recomp/ folder")
set(NX_PYTHON "${NXRECOMP_ROOT}/../tools/.venv/Scripts/python.exe" CACHE FILEPATH "Python with the nxrecomp requirements")
set(NX_OPT "/O1" CACHE STRING "optimisation flag of the lifted code (clang-cl syntax)")
set(NX_EXTRA_FLAGS "" CACHE STRING "more compile flags of the lifted code, space-separated (a profile-guided build: -fprofile-instr-generate, then -fprofile-instr-use=<file>)")
option(NX_USE_LAUNCHER "wrap lifted-code compiles in cmake/nx_cc_launcher.py (time-limit fallback to optnone / -O0)" OFF)

file(GLOB NX_LIFTER_SOURCES CONFIGURE_DEPENDS
     "${NXRECOMP_ROOT}/nxrecomp/*.py" "${NXRECOMP_ROOT}/nxrecomp/lift/*.py" "${NXRECOMP_ROOT}/nxrecomp/lift/tier2/*.py"
     "${NXRECOMP_ROOT}/include/nxrecomp/*.h")

function(nxrecomp_add_module target)
  cmake_parse_arguments(A "" "INPUT;NAME;BASE;INDEX;GAME;KIND;SHARDS;JOBS;JOB_POOL;OUTDIR;PATCHES;SITES" "" ${ARGN})
  if(NOT A_NAME)
    set(A_NAME main)
  endif()
  if(NOT A_BASE)
    set(A_BASE 0x7100000000)
  endif()
  if(NOT A_SHARDS)
    set(A_SHARDS 16)
  endif()
  if(NOT A_JOBS)
    cmake_host_system_information(RESULT A_JOBS QUERY NUMBER_OF_LOGICAL_CORES)
  endif()
  if(NOT A_OUTDIR)
    set(A_OUTDIR "${CMAKE_CURRENT_BINARY_DIR}/gen/${target}")
  endif()
  set(extra "")
  set(game_deps "")
  if(A_GAME)
    list(APPEND extra --game ${A_GAME})
    file(GLOB game_deps CONFIGURE_DEPENDS "${NXRECOMP_ROOT}/games/${A_GAME}/*.py")
  endif()
  if(A_KIND)
    list(APPEND extra --kind ${A_KIND})
  endif()
  if(NOT "${A_INDEX}" STREQUAL "")
    list(APPEND extra --index ${A_INDEX})
  endif()
  if(A_PATCHES)
    list(APPEND extra --patches ${A_PATCHES})
  endif()
  if(A_SITES)
    list(APPEND extra --sites ${A_SITES})
  endif()
  set(pool "")
  if(A_JOB_POOL)
    set(pool JOB_POOL ${A_JOB_POOL})
  endif()
  set(outs "${A_OUTDIR}/funcs.h" "${A_OUTDIR}/module_${A_NAME}.cpp")
  set(shard_srcs "")
  math(EXPR last "${A_SHARDS} - 1")
  foreach(i RANGE ${last})
    string(LENGTH "${i}" len)
    if(len EQUAL 1)
      set(n "00${i}")
    elseif(len EQUAL 2)
      set(n "0${i}")
    else()
      set(n "${i}")
    endif()
    list(APPEND shard_srcs "${A_OUTDIR}/c_${n}.cpp")
  endforeach()
  add_custom_command(
    OUTPUT ${outs} ${shard_srcs}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${A_OUTDIR}"
    COMMAND "${NX_PYTHON}" -m nxrecomp lift "${A_INPUT}" "${A_OUTDIR}" --name ${A_NAME} --base ${A_BASE}
            --shards ${A_SHARDS} --jobs ${A_JOBS} ${extra}
    WORKING_DIRECTORY "${NXRECOMP_ROOT}"
    DEPENDS "${A_INPUT}" ${A_SITES} ${NX_LIFTER_SOURCES} ${game_deps}
    COMMENT "nxrecomp: lifting ${A_NAME} (${A_INPUT})"
    ${pool}
    VERBATIM)
  add_library(${target} STATIC ${shard_srcs} "${A_OUTDIR}/module_${A_NAME}.cpp")
  target_include_directories(${target} PUBLIC "${NXRECOMP_ROOT}/include" "${A_OUTDIR}")
  target_compile_features(${target} PUBLIC cxx_std_20)
  if(MSVC)   # clang-cl
    # clang-cl ignores -ffp-contract=off ("unknown argument"; the compiler then contracts): it takes the clang spelling through /clang:.
    # -fno-math-errno: sqrtf and friends become instructions (the guest's errno is guest memory, the host's is never read). /GS-: no stack
    # cookie in every lifted function that has a local array.
    target_compile_options(${target} PRIVATE ${NX_OPT} /w /EHs-c- /GR- /GS- -fno-strict-aliasing /clang:-ffp-contract=off -fno-math-errno -mcx16)
    # the context is noescape at every call (include/nxrecomp/ctx.h NX_NOESC); LLVM must follow its thousands of uses to use that
    target_compile_options(${target} PRIVATE -DNX_NOESCAPE_CTX "SHELL:-mllvm -capture-tracking-max-uses-to-explore=100000")
    separate_arguments(extra_flags NATIVE_COMMAND "${NX_EXTRA_FLAGS}")
    target_compile_options(${target} PRIVATE ${extra_flags})
  else()
    target_compile_options(${target} PRIVATE -O1 -w -fno-exceptions -fno-rtti -fno-strict-aliasing -ffp-contract=off -mcx16)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      target_compile_options(${target} PRIVATE -DNX_NOESCAPE_CTX "SHELL:-mllvm -capture-tracking-max-uses-to-explore=100000")
    endif()
  endif()
  if(NX_USE_LAUNCHER)
    set_target_properties(${target} PROPERTIES CXX_COMPILER_LAUNCHER "${NX_PYTHON};${NXRECOMP_ROOT}/cmake/nx_cc_launcher.py")
  endif()
endfunction()
