param(
    [ValidateSet('MinGW','MSVC')][string]$Toolchain = 'MinGW',
    [string]$CMake = 'cmake',
    [string]$CCompiler = 'gcc',
    [string]$Make = 'mingw32-make',
    [string]$Generator = 'Visual Studio 18 2026',
    [ValidateSet('Debug','Release')][string[]]$Configurations = @('Debug','Release')
)
$ErrorActionPreference = 'Stop'
# CMake принимает число рабочих процессов в виде строки для обоих генераторов.
$parallelJobs = [Environment]::ProcessorCount.ToString()
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$deps = Join-Path $projectRoot 'build/deps'
New-Item -ItemType Directory -Force -Path $deps | Out-Null
# Запускает CMake и останавливает подготовку зависимостей при ненулевом коде завершения.
function Run([string[]]$Arguments) {
    & $CMake @Arguments
    if ($LASTEXITCODE -ne 0) { throw "CMake failed with exit code $LASTEXITCODE" }
}
# Проверяем контрольную сумму локального архива до распаковки, в том числе при повторном запуске.
function Source([string]$Archive, [string]$Hash, [string]$Directory) {
    $zip = Join-Path (Join-Path $projectRoot 'third_party') $Archive
    if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -ne $Hash) { throw "Archive checksum mismatch: $zip" }
    $source = Join-Path $deps $Directory
    if (!(Test-Path -LiteralPath $source)) { Expand-Archive -LiteralPath $zip -DestinationPath $deps }
    return $source
}
$cgSource = Source 'CGNS-4.4.0.zip' '72AD499936089102C1D77E3CDD78A623D61D1C6AD94C4DF3B23F36DB9DDFAD6A' 'CGNS-4.4.0'
$hdfSource = Source 'HDF5-1.14.6.zip' 'E78B518DA9FF8BCEC3A095A3DDBB9C906DFC99024593CF3D16DB200D97A10B09' 'hdf5-hdf5_1.14.6'
# Собираем только статические библиотеки, нужные CGNS; примеры, тесты и дополнительные кодеки не требуются.
$hdfOptions = @('-DBUILD_SHARED_LIBS=OFF','-DBUILD_TESTING=OFF','-DHDF5_BUILD_TOOLS=OFF','-DHDF5_BUILD_EXAMPLES=OFF','-DHDF5_BUILD_HL_LIB=OFF','-DHDF5_ENABLE_Z_LIB_SUPPORT=OFF','-DHDF5_ENABLE_SZIP_SUPPORT=OFF','-DHDF5_ENABLE_NONSTANDARD_FEATURE_FLOAT16=OFF')
$cgOptions = @('-DCGNS_BUILD_SHARED=OFF','-DCGNS_BUILD_CGNSTOOLS=OFF','-DCGNS_ENABLE_FORTRAN=OFF','-DCGNS_ENABLE_HDF5=ON','-DCGNS_ENABLE_TESTS=OFF')
if ($Toolchain -eq 'MinGW') {
    $compiler = (Get-Command $CCompiler).Source
    $makePath = (Get-Command $Make).Source
    $prefix = Join-Path $deps 'install'
    $hdfBuild = Join-Path $deps 'hdf5-build'
    $cgBuild = Join-Path $deps 'cgns-build'
    $toolOptions = @('-G','MinGW Makefiles',"-DCMAKE_C_COMPILER=$compiler","-DCMAKE_MAKE_PROGRAM=$makePath",'-DCMAKE_BUILD_TYPE=Release',"-DCMAKE_INSTALL_PREFIX=$prefix")
    Run -Arguments (@('-S',$hdfSource,'-B',$hdfBuild) + $toolOptions + $hdfOptions)
    Run -Arguments @('--build',$hdfBuild,'--target','install','--parallel',$parallelJobs)
    Run -Arguments (@('-S',$cgSource,'-B',$cgBuild,"-DCMAKE_PREFIX_PATH=$prefix") + $toolOptions + $cgOptions)
    Run -Arguments @('--build',$cgBuild,'--target','install','--parallel',$parallelJobs)
} else {
    # Все конфигурации MSVC используют статическую CRT; Debug выбирает её отладочный вариант.
    $crt = @('-DCMAKE_POLICY_DEFAULT_CMP0091=NEW','-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>')
    $hdfBuild = Join-Path $deps 'hdf5-msvc-mt'
    Run -Arguments (@('-S',$hdfSource,'-B',$hdfBuild,'-G',$Generator,'-A','x64') + $crt + $hdfOptions)
    # У каждой конфигурации свои бинарные библиотеки HDF5 и CGNS и свой префикс установки.
    foreach ($config in $Configurations) {
        $prefix = Join-Path $deps "install-msvc-mt/$config"
        $cgBuild = Join-Path $deps "cgns-msvc-mt-$config"
        Run -Arguments @('--build',$hdfBuild,'--config',$config,'--parallel',$parallelJobs)
        Run -Arguments @('--install',$hdfBuild,'--config',$config,'--prefix',$prefix)
        Run -Arguments (@('-S',$cgSource,'-B',$cgBuild,'-G',$Generator,'-A','x64',"-DCMAKE_INSTALL_PREFIX=$prefix","-DCMAKE_PREFIX_PATH=$prefix") + $crt + $cgOptions)
        Run -Arguments @('--build',$cgBuild,'--config',$config,'--target','INSTALL','--parallel',$parallelJobs)
    }
}
Write-Output "Dependencies installed inside $deps"
