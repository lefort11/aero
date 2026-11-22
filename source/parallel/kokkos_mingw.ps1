# Совместимость установленного CUDA Toolkit с MinGW Clang и статической CRT.
param([string]$Cuda, [string]$Directory, [switch]$Static)
$ErrorActionPreference = 'Stop'
$header = Join-Path $Cuda 'include/crt/math_functions.hpp'
$source = [IO.File]::ReadAllText($header).Replace("`r`n", "`n")
$guard = "#elif defined(_WIN32)`n`n#if (!defined(_MSC_VER) || _MSC_VER < 1800)"
$entries = @()
if ($source.Contains($guard)) {
    $replacement = Join-Path $Directory 'math_functions.hpp'
    $updated = $guard.Replace('defined(_WIN32)', 'defined(_WIN32) && !defined(__MINGW32__)')
    [IO.File]::WriteAllText($replacement, $source.Replace($guard, $updated))
    $entries += @{ type = 'file'; name = $header.Replace('\','/'); 'external-contents' = $replacement.Replace('\','/') }
}
if ($Static) {
    $toolset = $env:VCToolsInstallDir
    if (!$toolset) {
        $locator = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
        $vs = & $locator -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        $version = (Get-Content -Raw (Join-Path $vs 'VC/Auxiliary/Build/Microsoft.VCToolsVersion.default.txt')).Trim()
        $toolset = Join-Path $vs "VC/Tools/MSVC/$version"
    }
    foreach ($pair in @(@('libcmt','LIBCMT'), @('oldnames','OLDNAMES'), @('libvcruntime','vcruntime'))) {
        Copy-Item -LiteralPath (Join-Path $toolset "lib/x64/$($pair[0]).lib") -Destination (Join-Path $Directory "../lib$($pair[1]).a")
    }
}
$overlay = @{version = 0; roots = $entries} | ConvertTo-Json -Depth 4
[IO.File]::WriteAllText((Join-Path $Directory 'cuda-headers.json'), $overlay)
