// Copyright (c) 2024, ETH Zurich

#ifndef _FAST_WAVE_CONFIG_PARSING_HPP
#define _FAST_WAVE_CONFIG_PARSING_HPP

#include <filesystem>
#include <yaml-cpp/yaml.h>

#include "simulation.hpp"

namespace fs = std::filesystem;

using DeltabetaTable = std::vector<std::pair<Material, Complex<double>>>;

// Per-element multi-element Chantler coefficients (方案 B'), precomputed in
// Python and stored in subconfig.yaml under `plasma_optics_table`.
struct PlasmaOptics {
    double f1bar, f2eff_bar, zbar, z2bar;
};

[[nodiscard]] Complex<double> db_table_lookup(const Material &mat, const DeltabetaTable &db_table);

[[nodiscard]] Grating parse_grating(const YAML::Node &node, const DeltabetaTable &db_table);

[[nodiscard]] EnvGrating parse_envgrating(const YAML::Node &node, const DeltabetaTable &db_table);

[[nodiscard]] Sample parse_sample(const YAML::Node &node, const DeltabetaTable &db_table,
                                  const fs::path &config_path);

//precise_sample_update_begin
[[nodiscard]] PreciseSample parse_precise_sample(const YAML::Node &node, const DeltabetaTable &db_table,
                                                const fs::path &sim_dir);
//precise_sample_update_end

//plasma_sample_begin
[[nodiscard]] PlasmaSample parse_plasma_sample(const YAML::Node &node,
                                               const fs::path &sim_dir,
                                               const PlasmaOptics &optics);
//plasma_sample_end

[[nodiscard]] std::unique_ptr<OpticalElement> parse_optical_element(const YAML::Node &node,
                                                                    const DeltabetaTable &db_table,
                                                                    const fs::path &sim_dir,
                                                                    const std::vector<PlasmaOptics> &plasma_optics,
                                                                    std::size_t &plasma_idx);

[[nodiscard]] std::vector<std::unique_ptr<OpticalElement>>
parse_optical_elements(const YAML::Node &node, const DeltabetaTable &db_table,
                       const fs::path &sim_dir,
                       const std::vector<PlasmaOptics> &plasma_optics);

[[nodiscard]] std::vector<PlasmaOptics> parse_plasma_optics_table(const YAML::Node &node);

[[nodiscard]] DeltabetaTable parse_deltabeta_table(const YAML::Node &node);

[[nodiscard]] SimParams parse_sim_params(const YAML::Node &node, double wl);

[[nodiscard]] PointSource parse_point_source(const YAML::Node &node);

[[nodiscard]] DType parse_dtype(const YAML::Node &node);

[[nodiscard]] std::vector<double> parse_cutoff_angles(const YAML::Node &node);

[[nodiscard]] fs::path get_subdir(const fs::path &sim_dir, int source_idx);

[[nodiscard]] Config parse_config(const fs::path &sim_dir, int source_idx);

std::string zeropad(int number, std::size_t length);

#endif // _FAST_WAVE_CONFIG_PARSING_HPP
