# Visual Studio передаёт выбранную конфигурацию корневому GNU Make.
param(
    [Parameter(Mandatory)][string]$Goal,
    [ValidateSet('Debug','Release')][string]$Config = 'Release',
    [ValidateSet('cpu','cuda')][string]$Backend = 'cpu',
    [ValidatePattern('^[0-9]+$')][string]$Arch = '86',
    [string]$Mpi = 'true', [string]$OpenMp = 'true', [string]$Static = 'true',
    [string]$Bin = 'bin', [switch]$Rebuild, [switch]$Clean
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $root
$msys = if ($env:MSYS2_ROOT) { $env:MSYS2_ROOT } else { 'C:\msys64' }
$makeExe = Join-Path $msys 'usr\bin\make.exe'
$env:PATH = "$msys\mingw64\bin;$env:PATH;$msys\usr\bin"
# Преобразует флаг Visual Studio true/false в целочисленный параметр GNU Make.
function MakeFlag([string]$value) {
    if ($value -in @('true','1')) { return 1 }
    if ($value -in @('false','0')) { return 0 }
    throw "Expected true, false, 1 or 0. Received '$value'."
}
$arguments = @("-j$([Environment]::ProcessorCount)", "BUILD_TYPE=$Config", "KOKKOS_BACKEND=$Backend",
    "CUDA_ARCH=$Arch", "WITH_MPI=$(MakeFlag $Mpi)", "WITH_OPENMP=$(MakeFlag $OpenMp)",
    "STATIC_LINK=$(MakeFlag $Static)")
# GNU Make запускается в MSYS2, поэтому путь выходного каталога переводится в его Unix-форму.
$unixBin = & (Join-Path $msys 'usr\bin\cygpath.exe') -au $Bin
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$arguments += "BIN=$unixBin"
if ($Rebuild) { $arguments += '-B' }
$arguments += if ($Clean) { 'clean' } else { $Goal }

# IDE может одновременно начать сборку нескольких проектов решения. GNU Make
# сам распараллеливает исходники. Два Make не должны писать один архив одновременно.
$hash = [Security.Cryptography.SHA256]::Create()
$id = [BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($root.ToLowerInvariant())))
$hash.Dispose()
$mutex = [Threading.Mutex]::new($false, "Local\aero-make-$id")
try {
    try { [void]$mutex.WaitOne() } catch [Threading.AbandonedMutexException] {}
    & $makeExe @arguments
    $code = $LASTEXITCODE
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
exit $code
