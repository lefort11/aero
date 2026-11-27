#include "../../source/io/cli_help.h"
#include "cli.h"
#include "../../source/io/visualization/vtk.h"
#include "../../source/io/scheduled_output.h"
#include <fstream>
#include <cctype>
#include <algorithm>
#include "../../source/io/visualization/tecplot.h"
#include "../../source/io/submesh.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#endif
#include <limits>
#include <ostream>

namespace aero {
namespace visfield {
using tecplot::ExportError;
namespace {
// Проверяем идентификатор файла, чтобы распознать также относительный путь и ссылки.
bool same_file(const char* input, const char* output) {
#ifdef _WIN32
    // Определяет файловую идентичность путей для защиты исходных данных от перезаписи.
    const auto identify = [](const char* path, BY_HANDLE_FILE_INFORMATION& info) {
        HANDLE h = CreateFileA(path, FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               nullptr, OPEN_EXISTING, 0, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            const auto error = GetLastError();
            if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) return false;
            throw ExportError(std::string("Cannot inspect file: ") + path);
        }
        const bool ok = GetFileInformationByHandle(h, &info) != 0;
        CloseHandle(h);
        if (!ok) throw ExportError(std::string("Cannot identify file: ") + path);
        return true;
    };
    BY_HANDLE_FILE_INFORMATION a{}, b{};
    return identify(input, a) && identify(output, b) && a.dwVolumeSerialNumber == b.dwVolumeSerialNumber &&
           a.nFileIndexHigh == b.nFileIndexHigh && a.nFileIndexLow == b.nFileIndexLow;
#else
    struct stat a{}, b{};
    return stat(input, &a) == 0 && stat(output, &b) == 0 && a.st_dev == b.st_dev && a.st_ino == b.st_ino;
#endif
}
// Печатает синтаксис запуска и доступные параметры утилиты.
void usage(std::ostream& out) {
    out << "Usage: visfield MESH [FIELDS] OUTPUT.{plt,vtk} [--threads N]\n\n"
           "  MESH        Native AEROMESH v2/v3 or a .submesh plane slice.\n"
           "  FIELDS      Optional AEROFIELDS v1: text, binary or LZMA.\n"
           "  OUTPUT      .vtk writes binary VTK Legacy. .plt writes binary Tecplot 360.\n"
           "  --threads N Positive thread count (default: OpenMP runtime settings).\n"
           "  -h, --help  Show this help.\n\n"
           "Mesh SHA-256 must match. Coordinates use the original mesh units.\n"
           "A successful export replaces an existing output file.\n"
           "OpenMP: "
        << (tecplot::openmp_available() ? "enabled" : "disabled (one thread)") << "\n";
}
// Сравнивает расширение VTK без учёта регистра на всех платформах.
bool vtk_path(const std::string& path) {
    if (path.size() < 4) return false;
    auto suffix = path.substr(path.size() - 4);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return suffix == ".vtk";
}
// Проверяет привязку внешнего снимка и публикует готовый VTK через временный файл.
void export_vtk(const char* path, const Mesh& mesh, const FieldSnapshot* fields) {
    if (fields) fields->check_mesh(mesh);
    StagedFieldFile staged(path);
    std::ofstream stream(staged.temporary(), std::ios::binary);
    vtk::write(stream, &mesh, fields);
    stream.close();
    if (!stream) throw ExportError("Cannot finish VTK output");
    staged.publish(true);
}
// Разбирает положительное число потоков из аргумента командной строки.
int positive(const std::string& value) {
    if (value.empty())
        throw ExportError("Missing thread count");
    int result = 0;
    for (char c : value) {
        if (c < '0' || c > '9' || result > (std::numeric_limits<int>::max() - (c - '0')) / 10)
            throw ExportError("Thread count must be a positive integer");
        result = result * 10 + c - '0';
    }
    if (!result)
        throw ExportError("Thread count must be positive");
    return result;
}
}
// Читает пути к сетке и снимку, выбирает параметры OpenMP и запускает выбранный писатель.
int run_cli(int argc, const char* const* argv, std::ostream& out, std::ostream& err) {
    try {
        if (aero::help_requested(argc, argv)) {
            usage(out);
            return 0;
        }
        int end_paths = 1;
        while (end_paths < argc && std::string(argv[end_paths]).compare(0, 2, "--") != 0) {
            if (!argv[end_paths][0]) {
                usage(err);
                return 1;
            }
            ++end_paths;
        }
        if (end_paths != 3 && end_paths != 4) {
            usage(err);
            return 1;
        }
        tecplot::Options options;
        if (end_paths < argc) {
            if (std::string(argv[end_paths]) != "--threads" || argc - end_paths != 2) {
                usage(err);
                return 1;
            }
            options.threads = positive(argv[end_paths + 1]);
        }
        if (!tecplot::openmp_available() && options.threads > 1)
            throw ExportError("This build has no OpenMP; use --threads 1");
        const std::string input=argv[1];
        const bool slice=input.size()>=8 && input.compare(input.size()-8,8,".submesh")==0;
        const Mesh mesh = slice ? Submesh::read(input).mesh : Mesh::read(input);
        const char* path = argv[end_paths - 1];
        const bool vtk_output = vtk_path(path);
        for (int i = 1; i < end_paths - 1; ++i)
            if (same_file(argv[i], path)) throw ExportError("Output path refers to an input file");
        if (end_paths == 4) {
            const FieldSnapshot fields = FieldSnapshot::read(argv[2]);
            if (vtk_output) export_vtk(path, mesh, &fields);
            else tecplot::write(path, mesh, fields, options);
            out << "Written: " << path << "; dimension=" << fields.dimension << ", cells=" << fields.counts[3]
                << ", fields=" << fields.fields.size() << ", step=" << fields.step << "\n";
        } else {
            if (vtk_output) export_vtk(path, mesh, nullptr);
            else tecplot::write(path, mesh, options);
            out << "Written: " << path << "; dimension=" << mesh.view().dimension
                << ", cells=" << mesh.view().cell_count << "\n";
        }
        return 0;
    } catch (const std::exception& error) {
        err << "visfield export failed: " << error.what() << "\n";
        return 1;
    }
}
}
}
