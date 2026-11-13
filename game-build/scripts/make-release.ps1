# SPDX-License-Identifier: GPL-3.0-or-later
# Packs the player zip: no game code or data in it. The player's PC lifts and links fefw.exe from their own dump (tools/build/fefw_build.py,
# the bundled Zig and Python), so players install nothing else.
#
#   pwsh game-build\scripts\make-release.ps1 -Version 0.1.0
#
# Inputs, built beforehand from this checkout:
#   fefw_rt.dll + fefw_rt.lib   pwsh game-build\scripts\build_fefw.ps1 -Target fefw_rt -BuildDir build\rtdll -CMakeArgs "-DFEFW_RT_DLL=ON"
#   fefw-launcher.exe           built here (npm, Rust) with the build machine's paths remapped, unless -Launcher names one
#   the shader translator       staged Ryujinx sources (tools\shaders\scripts\setup_ryujinx.ps1); this script runs dotnet publish
# Third-party payload, downloaded into the cache folder from the official sites: Zig (checked against ziglang.org's sha256), the Python
# embeddable package, and the wheels of capstone, lz4, zbic and zstandard (pip, for CPython 3.13 win_amd64).
param(
    [Parameter(Mandatory)][string]$Version,
    [string]$Out = "",
    [string]$RtDir = "",
    [string]$Launcher = "",
    [string]$ZigVersion = "0.17.0",
    [string]$ZigSha256 = "b5663f69581dcf391293fbf16c06cb80d81d806545ce618b4d0bab7f0eb8c428",
    [string]$PythonVersion = "3.13.17",
    [switch]$NoZip
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Out) { $Out = Join-Path $root 'work\release' }
if (-not $RtDir) { $RtDir = Join-Path $root 'game-build\fefw\build\rtdll' }
$buildLauncher = -not $Launcher
if (-not $Launcher) { $Launcher = Join-Path $root 'launcher\src-tauri\target\release\fefw-launcher.exe' }
$cache = Join-Path $Out 'cache'
$name = "fefw-recomp-$Version"
$stage = Join-Path $Out 'stage'
$pkg = Join-Path $stage $name
New-Item -ItemType Directory -Force $cache | Out-Null
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $pkg | Out-Null

function Need([string]$p, [string]$what) { if (-not (Test-Path $p)) { throw "missing $what`: $p" } }
function Put([string]$from, [string]$to) {
    $dst = Join-Path $pkg $to
    New-Item -ItemType Directory -Force (Split-Path -Parent $dst) | Out-Null
    Copy-Item -Force $from $dst
}
function Mirror([string]$from, [string]$to, [string[]]$exclude = @()) {
    $dst = Join-Path $pkg $to
    New-Item -ItemType Directory -Force $dst | Out-Null
    $src = (Resolve-Path $from).Path
    Get-ChildItem $src -Recurse -File | Where-Object {
        $rel = $_.FullName.Substring($src.Length + 1)
        -not ($exclude | Where-Object { $rel -like $_ })
    } | ForEach-Object {
        $t = Join-Path $dst $_.FullName.Substring($src.Length + 1)
        New-Item -ItemType Directory -Force (Split-Path -Parent $t) | Out-Null
        Copy-Item $_.FullName $t
    }
}
function Fetch([string]$url, [string]$file, [string]$sha = "") {
    $p = Join-Path $cache $file
    if (-not (Test-Path $p)) {
        Write-Host "downloading $url"
        Invoke-WebRequest -UseBasicParsing $url -OutFile "$p.part"
        Move-Item -Force "$p.part" $p
    }
    if ($sha -and (Get-FileHash -Algorithm SHA256 $p).Hash.ToLower() -ne $sha.ToLower()) { Remove-Item $p; throw "sha256 mismatch: $url" }
    return $p
}

Need (Join-Path $RtDir 'fefw_rt.dll') 'fefw_rt.dll (build it with -DFEFW_RT_DLL=ON)'
Need (Join-Path $RtDir 'fefw_rt.lib') 'fefw_rt.lib'
Need "$root\runtime\third_party\renhdrum\renhdrum.dll" 'renhdrum.dll (runtime\third_party\renhdrum, built from the RenHDrum v0.3.0 source with a static CRT)'
# Desktop GPUs have no ASTC: a runtime built with NXRT_ASTC2BC7=OFF renders most textures black (issue #1). Refuse to package one.
if (Select-String -Path (Join-Path $RtDir 'fefw_rt.dll') -SimpleMatch 'NXRT_ASTC2BC7=OFF' -Quiet) {
    throw "fefw_rt.dll was built without the ASTC transcoder (NXRT_ASTC2BC7=OFF): reconfigure with -DNXRT_ASTC2BC7=ON"
}
if ($buildLauncher) {
    # Rust keeps source paths for panic messages: map the user profile (cargo registry, rustup) and the checkout to neutral names
    $env:RUSTFLAGS = "--remap-path-prefix=$($env:USERPROFILE)=~ --remap-path-prefix=$root=."
    if ($env:CARGO_HOME) { $env:RUSTFLAGS += " --remap-path-prefix=$($env:CARGO_HOME)=cargo" }
    Push-Location (Join-Path $root 'launcher')
    try {
        npm install --no-audit --no-fund; if ($LASTEXITCODE) { throw 'npm install failed' }
        npm run tauri build -- --no-bundle; if ($LASTEXITCODE) { throw 'launcher build failed' }
    } finally { Pop-Location; Remove-Item Env:RUSTFLAGS }
}
Need $Launcher 'fefw-launcher.exe'

# --- built from this checkout
# --- the haptics engine (MIT OR Apache-2.0, so it can ship, unlike the Streamline and FidelityFX DLLs). src/gfx/haptic.cpp
# loads it at run time with LoadLibraryExW from the exe folder and falls back to the standard two-motor rumble when it is not
# there. It is built from the RenHDrum source with a static CRT, so it does not need the Visual C++ redistributable.
Put "$root\runtime\third_party\renhdrum\renhdrum.dll" 'renhdrum.dll'
Put "$root\runtime\third_party\renhdrum\LICENSE-MIT" 'licenses\LICENSE.RenHDrum.MIT.txt'
Put "$root\runtime\third_party\renhdrum\NOTICE" 'licenses\NOTICE.RenHDrum.txt'
Put "$root\runtime\third_party\renhdrum\SHA256SUMS" 'licenses\RenHDrum.SHA256SUMS.txt'
Put (Join-Path $RtDir 'fefw_rt.dll') 'fefw_rt.dll'
Put (Join-Path $RtDir 'fonts\DejaVuSans.ttf') 'fonts\DejaVuSans.ttf'
Get-ChildItem $RtDir -Filter 'astcdec-*-shared.dll' -ErrorAction SilentlyContinue | ForEach-Object { Put $_.FullName $_.Name }
Put $Launcher 'fefw-launcher.exe'
Put (Join-Path $RtDir 'fefw_rt.lib') 'tools\build\sdk\fefw_rt.lib'
foreach ($f in 'fefw_rt_api.h', 'fefw_stub.cpp', 'fefw_fma.c') { Put "$root\game-build\fefw\src\$f" "tools\build\sdk\$f" }
foreach ($f in 'fefw.rc', 'fefw.manifest', 'fefw.ico') { Put "$root\game-build\fefw\res\$f" "tools\build\sdk\$f" }
foreach ($f in 'fefw_build.py', 'dump_hashes.json') { Put "$root\game-build\release\$f" "tools\build\$f" }
Put "$root\tools\nso\nso_unpack.py" 'tools\build\nso_unpack.py'
Put "$root\recomp\LICENSE" 'tools\build\recomp\LICENSE'
Mirror "$root\recomp\nxrecomp" 'tools\build\recomp\nxrecomp' @('*__pycache__*', '*.pyc')
Mirror "$root\recomp\include\nxrecomp" 'tools\build\recomp\include\nxrecomp'
foreach ($f in 'CMakeLists.txt', 'config.py', 'main.sites', 'hot.txt') { Put "$root\recomp\games\fefw\$f" "tools\build\recomp\games\fefw\$f" }
Put "$root\tools\shaders\ampere\extract_corpus.py" 'tools\shaders\extract_corpus.py'
(Get-Content -Raw "$root\game-build\release\README.txt").Replace('@VERSION@', $Version) | Set-Content -NoNewline (Join-Path $pkg 'README.txt')
Put "$root\LICENSE" 'LICENSE.txt'
Put "$root\LEGAL.md" 'LEGAL.txt'
Put "$root\game-build\release\BUNDLED_TOOLS.txt" 'licenses\BUNDLED_TOOLS.txt'
Put "$root\runtime\THIRD_PARTY_NOTICES.md" 'licenses\THIRD_PARTY_NOTICES.md'
Put "$root\tools\shaders\licenses\LICENSE.Ryubing.MIT" 'licenses\LICENSE.Ryujinx.MIT.txt'

# --- the language tables fefw_rt.dll reads (locales\<lang>.ini next to the DLL). The launcher copies them into the run folder
# before the first start and does not replace one that is already there, so a player's own translation of a shipped language wins.
$tables = Get-ChildItem "$root\locales" -Filter '*.ini' -File -ErrorAction SilentlyContinue
if ($tables) { foreach ($t in $tables) { Put $t.FullName "locales\$($t.Name)" } }
Put "$root\locales\README.md" 'locales\README.md'

# --- the shader translator, self-contained (no .NET install needed on the player's PC)
$harness = Join-Path $Out 'harness'
# Always rebuild: reusing Out/harness can silently ship an older translator after
# the launcher invalidates the player's shader cache for a new translator version.
Need "$root\tools\shaders\third_party\ryubing" 'staged Ryujinx sources (tools\shaders\scripts\setup_ryujinx.ps1)'
dotnet publish "$root\tools\shaders\harness\SpirvHarness.csproj" -c Release -r win-x64 --self-contained true -o $harness `
    -p:DebugType=none -p:DebugSymbols=false -p:Deterministic=true "-p:PathMap=$root=."
if ($LASTEXITCODE) { throw 'dotnet publish failed' }
Mirror $harness 'tools\spirv_harness' @('*.pdb')

# --- Zig (the compiler for the lifted code on the player's PC)
$zipZig = Fetch "https://ziglang.org/download/$ZigVersion/zig-x86_64-windows-$ZigVersion.zip" "zig-$ZigVersion.zip" $ZigSha256
$zx = Join-Path $cache "zig-$ZigVersion"
if (-not (Test-Path $zx)) { [IO.Compression.ZipFile]::ExtractToDirectory($zipZig, $zx) }
$ztop = Get-ChildItem $zx -Directory | Select-Object -First 1
Mirror $ztop.FullName 'tools\build\zig' @('doc\*')
Put (Join-Path $ztop.FullName 'LICENSE') 'licenses\LICENSE.zig.MIT.txt'

# --- Python embeddable package + the lifter's and tools' packages
$zipPy = Fetch "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip" "python-$PythonVersion-embed-amd64.zip"
$py = Join-Path $pkg 'tools\python'
[IO.Compression.ZipFile]::ExtractToDirectory($zipPy, $py)
$pth = Get-ChildItem $py -Filter 'python*._pth' | Select-Object -First 1
Add-Content $pth.FullName "Lib\site-packages"
$whl = Join-Path $cache "wheels-$PythonVersion"
if (-not (Test-Path $whl)) {
    python -m pip download --only-binary=:all: --python-version 3.13 --platform win_amd64 -d $whl capstone==5.0.9 lz4 zbic zstandard
    if ($LASTEXITCODE) { throw 'pip download failed' }
}
$site = Join-Path $py 'Lib\site-packages'
New-Item -ItemType Directory -Force $site | Out-Null
Get-ChildItem $whl -Filter *.whl | ForEach-Object { [IO.Compression.ZipFile]::ExtractToDirectory($_.FullName, $site, $true) }
Put (Join-Path $py 'LICENSE.txt') 'licenses\LICENSE.Python.PSF.txt'

# --- never ship anything from a game dump, a shader cache built from the game, saves or keys
function Magic([IO.FileInfo]$f) {
    if ($f.Length -lt 4) { return '' }
    $s = [IO.File]::OpenRead($f.FullName); $b = New-Object byte[] 4; [void]$s.Read($b, 0, 4); $s.Close()
    return [Text.Encoding]::ASCII.GetString($b)
}
$bad = Get-ChildItem $pkg -Recurse -Force | Where-Object {
    $rel = $_.FullName.Substring($pkg.Length + 1)
    $_.Name -match '\.(nso|nro|nca|nsp|xci|nup|nsz|xcz|spv|keys|sav)$' -or
    $_.Name -in 'exefs', 'romfs', 'shader_cache', 'save', 'saves', 'fefw.exe', 'fefw.ini', 'controls.ini' -or
    $rel -match '(^|\\)(updates|work|nup_dump)(\\|$)' -or
    (-not $_.PSIsContainer -and (Magic $_) -in 'NSO0', 'NRO0', 'NCA3', 'PFS0', 'HFS0', 'BNSH', 'SARC', 'NUP')
}
if ($bad) { $bad | ForEach-Object { Write-Error "game-derived or private file in the package: $($_.FullName)" -ErrorAction Continue }; throw 'package check failed' }

# --- and nothing that names the machine it was built on: the builder's profile path, in ASCII or UTF-16
$needles = @($env:USERPROFILE, $env:USERPROFILE.Replace('\', '/')) | Where-Object { $_ -and $_.Length -ge 8 }
$leaks = Get-ChildItem $pkg -Recurse -File | Where-Object {
    $bytes = [IO.File]::ReadAllBytes($_.FullName)
    $hit = $false
    foreach ($n in $needles) {
        foreach ($enc in [Text.Encoding]::ASCII, [Text.Encoding]::Unicode) {
            $text = $enc.GetString($bytes)
            if ($text.IndexOf($n, [StringComparison]::OrdinalIgnoreCase) -ge 0) { $hit = $true }
        }
    }
    $hit
}
if ($leaks) { $leaks | ForEach-Object { Write-Error "build-machine path in the package: $($_.FullName)" -ErrorAction Continue }; throw 'package check failed' }

if ($NoZip) { Write-Host "staged $pkg"; return }
$zip = Join-Path $Out "$name-win64.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
[IO.Compression.ZipFile]::CreateFromDirectory($pkg, $zip, [IO.Compression.CompressionLevel]::Optimal, $true)
Write-Host "wrote $zip ($([math]::Round((Get-Item $zip).Length / 1MB)) MiB)"
