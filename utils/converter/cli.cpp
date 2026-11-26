#include "../../source/io/cli_help.h"
#include "../../source/io/utf8_label.h"
#include "cli.h"
#include "converter.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <locale>
#include <sstream>
namespace {
// Печатает синтаксис запуска и доступные параметры утилиты.
void usage(std::ostream& out) {
    out << "Usage: mesh_converter INPUT OUTPUT [OPTIONS]\n"
           "\n"
           "Arguments:\n"
           "  INPUT   Input mesh: Gmsh (.msh), CGNS (.cgns), another directory/mesh.txt (text .msh arrays).\n"
           "  OUTPUT  Output mesh path: AEROMESH v3.\n"
           "\n"
           "Options:\n"
           "  --from gmsh|cgns|another  Input format (default: infer from path).\n"
           "  --format text|binary|lzma  Output encoding (default: binary).\n"
           "  --base N               CGNS base index, starting at 1.\n"
           "  --zone N               CGNS zone index, starting at 1.\n"
           "  --unmarked NAME        Name for unmarked boundaries (default: unmarked).\n"
           "  -h, --help             Show this help.\n"
           "\n"
           "Example: mesh_converter input.msh output.mesh --format text\n"
           "Existing output files are never overwritten.\n"
           "CGNS support: "
        << (aero::converter::cgns_available() ? "enabled" : "disabled") << "\n";
}
// Разбирает номер базы или зоны из аргумента командной строки.
int index(const std::string& text) {
    std::istringstream in(text);
    int value = 0;
    if (!(in >> value) || value < 1 || in.peek() != std::char_traits<char>::eof())
        throw aero::MeshError("Expected a positive index");
    return value;
}
// Определяет формат сетки по параметру или имени входного файла.
std::string detect_input_format(const std::string& input) {
    const auto slash = input.find_last_of("/\\");
    const auto name = aero::lower_keyword(input.substr(slash == std::string::npos ? 0 : slash + 1));
    std::ifstream descriptor((input + "/mesh.txt").c_str());
    if (descriptor || name == "mesh.txt")
        return "another";
    const auto dot = input.find_last_of('.');
    const auto extension = dot == std::string::npos ? "" : aero::lower_keyword(input.substr(dot));
    if (extension == ".msh")
        return "gmsh";
    if (extension == ".cgns")
        return "cgns";
    throw aero::MeshError("Cannot infer input format; use --from gmsh|cgns|another");
}
// После разбора параметров каждый импортёр отдаёт один и тот же тип Mesh для записи AEROMESH.
aero::Mesh read_input_mesh(const std::string& format, const std::string& path,
                           const aero::converter::Options& options) {
    if (format == "gmsh")
        return aero::converter::read_gmsh(path, options);
    if (format == "cgns")
        return aero::converter::read_cgns(path, options);
    return aero::converter::read_another(path, options);
}
}
// Разбирает аргументы, выполняет операцию утилиты и возвращает код завершения.
int aero::converter::run_cli(int argc, const char* const* argv, std::ostream& out, std::ostream& err) {
    try {
        if (aero::help_requested(argc, argv)) {
            usage(out);
            return 0;
        }
        if (argc < 3 || std::string(argv[1]).empty() || std::string(argv[2]).empty() ||
            std::string(argv[1]).compare(0, 2, "--") == 0 || std::string(argv[2]).compare(0, 2, "--") == 0) {
            usage(err);
            return 1;
        }
        std::string input = argv[1], output = argv[2], from;
        aero::MeshFormat format = aero::MeshFormat::Binary;
        aero::converter::Options options;
        for (int i = 3; i < argc; ++i) {
            const std::string flag = argv[i];
            if (i + 1 == argc)
                throw aero::MeshError("Missing value for " + flag);
            const std::string value = argv[++i];
            if (flag == "--from")
                from = value;
            else if (flag == "--format") {
                if (value == "text")
                    format = aero::MeshFormat::Text;
                else if (value == "binary")
                    format = aero::MeshFormat::Binary;
                else if (value == "lzma")
                    format = aero::MeshFormat::Lzma;
                else
                    throw aero::MeshError("Output format must be text, binary or lzma");
            } else if (flag == "--base")
                options.base = index(value);
            else if (flag == "--zone")
                options.zone = index(value);
            else if (flag == "--unmarked")
                options.unmarked_name = value;
            else
                throw aero::MeshError("Unknown option: " + flag);
        }
        if (from.empty())
            from = detect_input_format(input);
        if (from != "gmsh" && from != "cgns" && from != "another")
            throw aero::MeshError("Input format must be gmsh, cgns or another");
        if (from != "cgns" && (options.base || options.zone))
            throw aero::MeshError("--base/--zone apply only to CGNS");
        {
            std::ifstream existing(output.c_str(), std::ios::binary);
            if (existing)
                throw aero::MeshError("Output already exists: " + output);
        }
        const aero::Mesh mesh = read_input_mesh(from, input, options);
        mesh.write(output, format);
        const aero::MeshView v = mesh.view();
        out << "Converted: dimension=" << v.dimension << ", nodes=" << v.node_count
            << ", cells=" << v.cell_count << ", faces=" << v.face_count << ", patches=" << v.patch_count
            << "\n";
        for (const std::string& name : mesh.data().patch_names)
            out << "  boundary: " << name << "\n";
        return 0;
    } catch (const std::exception& error) {
        err << "Conversion failed: " << error.what() << "\n";
        return 1;
    }
}
