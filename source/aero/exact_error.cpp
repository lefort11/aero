#include "exact_error.h"
#include "startup.h"
#include "../io/flow_values.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace aero {
ExactError::ExactError(const RunSettings& run, const Processes& world, const MeshPartition& part,
                       const PeriodicPartition* seam, const double* volumes, const std::string& directory)
    : dimension_(part.mesh.view().dimension), run_(run), world_(world) {
    names_.push_back("rho");
    const char* velocity[] = {"ux","uy","uz"};
    for (unsigned j = 0; j < dimension_; ++j) names_.push_back(velocity[j]);
    names_.push_back("p");
    const auto expressions = read_exact_formulas(run.directories, names_, &run.input_warnings);
    for (unsigned k = 0; k < expressions.size(); ++k) {
        try { formulas_.emplace_back(expressions[k]); }
        catch (const std::exception& e) { throw std::invalid_argument("exact.txt: " + names_[k] + ": " + e.what()); }
    }
    names_.push_back("E");
    const auto mesh = part.mesh.view();
    for (MeshIndex i = 0; i < mesh.node_count; ++i) {
        if (part.node_owners[i] != world.rank) continue;
        const auto slot = seam ? seam->node_dof[i] : i;
        // Объём периодических копий уже суммирован у представителя.
        if (seam && seam->ids[slot] != part.nodes[i]) continue;
        nodes_.push_back({mesh.coordinate(i), slot, std::sqrt(volumes[slot])});
    }
    if (!world.rank) {
        create_output_directory(directory);
        file_.open(directory + "/sol_error.txt");
        if (!file_) throw std::runtime_error("cannot create " + directory + "/sol_error.txt");
    }
}

void ExactError::write(MeshIndex step, double time, const std::vector<double>& state) {
    const unsigned width = dimension_ + 2, fields = width + 1;
    std::vector<double> local(2 * fields, 0.), gathered(world_.size * local.size());
    world_.local([&] {
        for (const auto& node : nodes_) {
            double exact[6] = {};
            for (unsigned k = 0; k < width; ++k)
                exact[k] = formulas_[k].evaluate(node.point[0], node.point[1], node.point[2], time);
            // Последняя колонка — энергия на единицу объёма, в Linear её первая вариация.
            exact[width] = run_.background.active() ? run_.background.energy(dimension_, exact, run_.gamma)
                : exact[width-1]/(run_.gamma-1.) + .5*exact[0]*euler_detail::dot(dimension_, exact+1, exact+1);
            const auto computed = output_values(dimension_, state.data()+node.slot*width,
                                                run_.gamma, node.slot, run_.background);
            for (unsigned k = 0; k < fields; ++k) {
                const unsigned column = k < width-1 ? k : 4 + k - (width-1);
                const double error = std::abs(computed[column] - exact[k]);
                if (!std::isfinite(error)) throw std::domain_error("nonfinite exact error in " + names_[k]);
                local[k] = std::hypot(local[k], node.root_volume * error);
                local[fields+k] = std::max(local[fields+k], error);
            }
        }
    });
    world_.allgather(local.data(), static_cast<int>(local.size()), gathered.data());
    world_.local([&] {
        if (world_.rank) return;
        std::ostringstream out;
        out << std::setprecision(17) << "Solution error, step " << step << ", time " << time
            << "\nVariable L_2 L_inf\n";
        for (unsigned k = 0; k < fields; ++k) {
            double l2 = 0., linf = 0.;
            for (size_t offset = 0; offset < gathered.size(); offset += local.size()) {
                l2 = std::hypot(l2, gathered[offset+k]);
                linf = std::max(linf, gathered[offset+fields+k]);
            }
            out << names_[k] << ' ' << l2 << ' ' << linf << '\n';
        }
        const auto table = out.str();
        std::cout << table;
        file_ << table << std::flush;
        if (!file_) throw std::runtime_error("cannot write sol_error.txt");
    });
}
}
