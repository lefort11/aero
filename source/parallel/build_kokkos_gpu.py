"""Собирает NVIDIA-часть Kokkos на C++11 для статической или динамической линковки."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]

def windows_static_libraries(destination):
    """Находит библиотеки CRT, с которыми NVIDIA собрала cudart_static.lib для Windows."""
    configured = os.environ.get("VCToolsInstallDir")
    if configured:
        library = Path(configured) / "lib/x64"
    else:
        locator = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
        installation = subprocess.check_output([str(locator), "-latest", "-products", "*", "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
            text=True).strip()
        versions = Path(installation) / "VC/Auxiliary/Build/Microsoft.VCToolsVersion.default.txt"
        version = versions.read_text(encoding="utf-8-sig").strip()
        library = Path(installation) / "VC/Tools/MSVC" / version / "lib/x64"
    # В секции .drectve CUDA запросила LIBCMT и OLDNAMES по именам MSVC.
    # В режиме MinGW LLD ищет те же архивы под именем lib<имя>.a.
    for source, archive in (("libcmt", "libLIBCMT"), ("oldnames", "libOLDNAMES"),
                            ("libvcruntime", "libvcruntime")):
        shutil.copy2(library / (source + ".lib"), destination / (archive + ".a"))


def windows_cuda_headers(cuda, work):
    """Даёт Clang корректные объявления математических функций для MinGW, сохраняя установленный SDK."""
    header = cuda / "include/crt/math_functions.hpp"
    data = header.read_text(encoding="utf-8")
    marker = "#elif defined(_WIN32)\n\n#if (!defined(_MSC_VER) || _MSC_VER < 1800)"
    if marker not in data:
        return []
    replacement = marker.replace("defined(_WIN32)", "defined(_WIN32) && !defined(__MINGW32__)")
    local = work / header.name
    local.write_text(data.replace(marker, replacement, 1), encoding="utf-8")
    # Подмена действует только в этом запуске компилятора и только для одного заголовка.
    overlay = work / "cuda-headers.json"
    overlay.write_text(json.dumps({"version": 0, "roots": [
        {"type": "file", "name": header.as_posix(), "external-contents": local.as_posix()}]}), encoding="utf-8")
    return ["-ivfsoverlay", str(overlay)]


def build(args):
    """Компилирует библиотеку и обходы отдельно от OpenMP-части, чтобы CPU-расчёт обходился без GPU."""
    windows = os.name == "nt"
    work, source, cuda = (p.resolve() for p in (args.work, args.source, args.cuda))
    work.mkdir(parents=True, exist_ok=True)
    # Обе половины модуля используют одно описание типов Kokkos и одну версию стандарта.
    settings = {"KOKKOS_VERSION": " 30201"}
    for feature in ("CXX11", "SERIAL", "CUDA", "CUDA_LAMBDA", "COMPLEX_ALIGN"):
        settings["KOKKOS_ENABLE_" + feature] = ""
    template = (source / "cmake/KokkosCore_config.h.in").read_text(encoding="utf-8")

    def definition(match):
        name = match[1]
        return "#define " + name + settings[name] if name in settings else "/* #undef " + name + " */"

    config = re.sub(r"(?m)^#cmakedefine (\w+)[^\n]*", definition, template)
    (work / "KokkosCore_config.h").write_text(config, encoding="utf-8")
    os.environ["PATH"] = str(cuda / "bin") + os.pathsep + os.environ["PATH"]
    includes = ["-I" + str(p) for p in (work, source / "core/src", source / "containers/src")]
    if args.static:
        # У CPU-сборки Kokkos свой DefaultExecutionSpace. При включении двух реализаций
        # в один EXE разделяем их C++-символы, а вызывающая сторона использует общий API Aero.
        includes += ["-DKokkos=AeroDeviceKokkos", "-DAERO_STATIC_DEVICE_WORK"]
    # В Linux компилятор nvcc поддерживает C++11 на обеих сторонах. В Windows применяем
    # Clang из среды MinGW, чтобы и код хоста действительно компилировался с -std=c++11.
    if windows:
        gpu = [args.compiler, "-std=c++11", "-O3", "-x", "cuda", "--cuda-path=" + str(cuda),
               "--cuda-gpu-arch=sm_" + args.arch, "-D__CRT__NO_INLINE", "-Wno-ignored-attributes",
               *windows_cuda_headers(cuda, work)]
    else:
        gpu = [args.compiler, "-std=c++11", "-O3", "-x", "cu", "-arch=sm_" + args.arch,
               "--extended-lambda", "--expt-relaxed-constexpr", "-Xcompiler=-fPIC,-fvisibility=hidden"]
    host = [args.host_compiler, "-std=c++11", "-O3", "-I" + str(cuda / "include")]
    core = source / "core/src"
    bridge = ROOT / "source/parallel/device_work.cu"
    units = sorted(core.glob("impl/*.cpp")) + sorted(core.glob("Cuda/*.cpp")) + [bridge]

    def compile_unit(unit):
        """Ошибку параллельной сборки сопровождаем выводом именно того файла, на котором она произошла."""
        command = gpu
        if windows and unit.stem in ("Kokkos_CPUDiscovery", "Kokkos_Stacktrace"):
            command = host
        elif windows and unit.parent.name == "impl":
            command = gpu + ["--cuda-host-only"]
        obj = work / (unit.stem + ".o")
        result = subprocess.run(command + includes + ["-c", str(unit), "-o", str(obj)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(result.stdout.decode(errors="replace"))
        return str(obj)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        objects = list(pool.map(compile_unit, units))
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.static:
        if windows:
            windows_static_libraries(target.parent)
        # Собираем новый архив целиком. В нём не останутся старые объекты при смене конфигурации.
        temporary = target.with_suffix(".new.a")
        if temporary.exists():
            temporary.unlink()
        subprocess.run([args.archiver, "rcs", str(temporary), *objects], check=True)
        temporary.replace(target)
    elif windows:
        output = target.parent / "aero_kokkos_cuda.dll"
        subprocess.run([args.host_compiler, "-shared", "-static",
                        "-Wl,--exclude-all-symbols", "-Wl,--out-implib," + str(target), *objects,
                        str(cuda / "lib/x64/cudart.lib"), "-o", str(output)], check=True)
        # Windows ищет этот модуль рядом с программой, в том числе при запуске тестов без настроенного PATH.
        for runtime in (cuda / "bin").glob("cudart64_*.dll"):
            shutil.copy2(runtime, target.parent / runtime.name)
    else:
        # Скрываем внутренние символы GPU-версии Kokkos от независимо собранной библиотеки CPU.
        subprocess.run([args.host_compiler, "-shared", "-Wl,-Bsymbolic", "-Wl,-soname," + target.name, *objects,
                        "-L" + str(cuda / "lib64"), "-lcudart", "-ldl", "-pthread", "-o", str(target)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("work", "source", "cuda", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--host-compiler", required=True)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--archiver", default="ar")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    build(parser.parse_args())
