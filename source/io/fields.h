#ifndef AERO_IO_FIELDS_H
#define AERO_IO_FIELDS_H

#include <array>
#include <cstdint>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>

namespace aero {
class Mesh;
enum class FieldFormat { Text, Binary, Lzma };
enum class FieldLocation : std::uint8_t { Node = 0, Edge = 1, Face = 2, Cell = 3 };

class FieldError : public std::runtime_error {
public:
    explicit FieldError(const std::string& message) : std::runtime_error(message) {}
};

struct Field {
    std::string name;
    FieldLocation location;
    std::vector<std::string> components;
    std::string units;
    // Component-major: values[component * entity_count + entity_id].
    // Covers every entity of location, using the mesh's zero-based IDs.
    std::vector<double> values;
    Field() : location(FieldLocation::Cell) {}
};

struct FieldReadLimits {
    // Stored vectors/strings, excluding allocator overhead and codec workspace.
    std::uint64_t max_payload_bytes;
    // Complete uncompressed file size, including metadata and text whitespace.
    std::uint64_t max_decoded_bytes;
    std::uint64_t max_string_bytes, max_fields, max_components;
    FieldReadLimits() : max_payload_bytes(1024ULL*1024*1024),
        max_decoded_bytes(2ULL*1024*1024*1024), max_string_bytes(4096),
        max_fields(1024), max_components(1024) {}
};

// Host-only C++11 I/O. No geometry or connectivity is stored in this object.
// All formats are detected by signature, regardless of file extension.
// See FIELDS.txt for AEROFIELDS v1 and its binary wire layout.
class FieldSnapshot {
public:
    unsigned dimension;
    // NODE, EDGE, FACE, CELL, in that order.
    std::array<std::uint64_t, 4> counts;
    // Lowercase hex SHA-256 of Mesh::write(..., MeshFormat::Binary), v2.
    std::string mesh_sha256;
    std::uint64_t step;
    double time;
    std::vector<Field> fields;

    // Unbound snapshot; populate metadata or construct from a mesh before use.
    FieldSnapshot() : dimension(0), counts{{0,0,0,0}}, step(0), time(0) {}
    explicit FieldSnapshot(const Mesh& mesh, std::uint64_t step_number = 0, double time_seconds = 0);
    std::uint64_t entity_count(FieldLocation location) const;
    void validate() const;
    void check_mesh(const Mesh& mesh) const;
    static std::string mesh_fingerprint(const Mesh& mesh);

    static FieldSnapshot read(const std::string& path, const FieldReadLimits& limits = FieldReadLimits());
    static FieldSnapshot read(std::istream& input, const FieldReadLimits& limits = FieldReadLimits());
    // Validates before opening/truncating a file. I/O failure may leave partial output.
    void write(const std::string& path, FieldFormat format = FieldFormat::Binary) const;
    void write(std::ostream& output, FieldFormat format = FieldFormat::Binary) const;
};
} // namespace aero
#endif
