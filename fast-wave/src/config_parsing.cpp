// Copyright (c) 2024, ETH Zurich

#include <Npy++.h>
#include <spdlog/spdlog.h>
#include <config_parsing.hpp>
#include <plasma_physics.hpp>

#include <algorithm>
#include <fstream>
#include <numeric>

std::string stringify_key(const char *key) { return std::string(key); }

std::string stringify_key(int key) { return std::to_string(key); }

// Load a float32/float64 .npy grid and return it as float32, honouring the
// dtype declared in the npy header. npypp::LoadFull<T> ignores the header's
// word size and reads sizeof(T) per element, so float64 grids were silently
// misread as float32 (each double split into two bogus floats) — which turned
// plasma_sample deltabeta grids into NaN/garbage and produced all-NaN 2D
// detector outputs. This function is the dtype-aware replacement.
[[nodiscard]] std::vector<float> load_float_grid(const fs::path &path,
                                                 std::vector<size_t> &shape_out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open npy grid: " + path.string());

    char magic[6];
    f.read(magic, 6);
    if (std::string(magic, 6) != std::string("\x93NUMPY", 6))
        throw std::runtime_error("not an npy file: " + path.string());

    uint8_t major = 0;
    uint8_t minor = 0;
    f.read(reinterpret_cast<char *>(&major), 1);
    f.read(reinterpret_cast<char *>(&minor), 1);  // version minor byte
    uint64_t header_len = 0;
    if (major == 1) {
        uint16_t hl = 0;
        f.read(reinterpret_cast<char *>(&hl), 2);
        header_len = hl;
    } else if (major == 2) {
        uint32_t hl = 0;
        f.read(reinterpret_cast<char *>(&hl), 4);
        header_len = hl;
    } else {
        throw std::runtime_error("unsupported npy major version " + std::to_string(major) +
                                 " in " + path.string());
    }
    std::string header(header_len, '\0');
    f.read(header.data(), static_cast<std::streamsize>(header_len));

    size_t word_size = 0;
    char endianness = 0;
    bool fortran_order = false;
    std::vector<size_t> shape;
    npypp::detail::ParseNpyHeader(header, word_size, shape, fortran_order, endianness);
    if (fortran_order)
        throw std::runtime_error("fortran-order npy grids are not supported: " + path.string());

    const size_t n = std::accumulate(shape.begin(), shape.end(), size_t(1),
                                     std::multiplies<size_t>());
    std::vector<float> out(n);
    if (word_size == 4) {
        f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(n * 4));
    } else if (word_size == 8) {
        std::vector<double> tmp(n);
        f.read(reinterpret_cast<char *>(tmp.data()), static_cast<std::streamsize>(n * 8));
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(tmp[i]);
    } else {
        throw std::runtime_error("unsupported npy word size " + std::to_string(word_size) +
                                 " in " + path.string());
    }
    shape_out = shape;
    return out;
}

template <typename Key> [[nodiscard]] double get_scalar(YAML::Node const &node, const Key &key) {
    if (!node[key].IsDefined()) {
        throw std::runtime_error("Value " + stringify_key(key) + " is not defined");
    }
    if (node[key].IsScalar()) {
        return node[key].template as<double>();
    } else {
        throw std::runtime_error("Value " + stringify_key(key) + " is not a scalar");
    }
}

template <typename Key>
[[nodiscard]] Material get_material(YAML::Node const &node, const Key &key) {
    if (!node[key].IsDefined()) {
        throw std::runtime_error("Value " + stringify_key(key) + " is not defined");
    }
    if (!node[key].IsSequence()) {
        throw std::runtime_error("Value " + stringify_key(key) + " is not a sequence");
    }
    if (!node[key][0].IsDefined()) {
        throw std::runtime_error("First component (material name) of material " +
                                 stringify_key(key) + " is missing");
    }
    if (!node[key][1].IsScalar()) {
        throw std::runtime_error("Second component (density) of material " + stringify_key(key) +
                                 " is missing");
    }

    return Material{node[key][0].template as<std::string>(), get_scalar(node[key], 1)};
}

[[nodiscard]] Complex<double> db_table_lookup(const Material &mat, const DeltabetaTable &db_table) {
    using SCPair = std::pair<double, Complex<double>>;

    std::vector<SCPair> candidates;
    candidates.reserve(db_table.size());

    for (const auto &pair : db_table) {
        if (pair.first.name == mat.name) {
            candidates.emplace_back(std::abs(pair.first.density - mat.density), pair.second);
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const SCPair &a, const SCPair &b) { return a.first < b.first; });

    if (candidates.empty()) {
        throw std::runtime_error("Material " + mat.name + " not found in deltabeta_table");
    }
    if (candidates[0].first > 1e-5) {
        throw std::runtime_error("Material " + mat.name +
                                 " has no matching density in deltabeta_table");
    }
    return candidates[0].second;
}

template <typename Key>
[[nodiscard]] Complex<double> parse_optional_material(const YAML::Node &node, const Key &key,
                                                      const DeltabetaTable &db_table) {
    if (!node[key].IsDefined()) {
        throw std::runtime_error("Material " + stringify_key(key) + " is not defined");
    }
    if (node[key].size() == 0) {
        return Complex<double>{};
    }
    return db_table_lookup(get_material(node, key), db_table);
}

[[nodiscard]] Grating parse_grating(const YAML::Node &node, const DeltabetaTable &db_table) {
    const auto z_start = get_scalar(node, "z_start");
    const auto dc = node["dc"].as<std::array<double, 2>>();
    const auto thickness = get_scalar(node, "thickness");
    const auto pitch = get_scalar(node, "pitch");
    const auto nr_steps = node["nr_steps"].as<int>();
    auto x_positions = node["x_positions"].as<std::vector<double>>();
    const auto substrate_thickness = get_scalar(node, "substrate_thickness");

    const auto deltabeta_a = parse_optional_material(node, "mat_a", db_table);
    const auto deltabeta_b = parse_optional_material(node, "mat_b", db_table);
    const auto deltabeta_substrate = parse_optional_material(node, "mat_substrate", db_table);

    Grating g;
    g.z_start = z_start;
    g.x_positions = std::move(x_positions);
    g.type = OpticalElementType::Grating;
    g.thickness = thickness;
    g.pitch = pitch;
    g.dc = dc;
    g.nr_steps = nr_steps;
    g.substrate_thickness = substrate_thickness;
    g.deltabeta_a = deltabeta_a;
    g.deltabeta_b = deltabeta_b;
    g.deltabeta_substrate = deltabeta_substrate;

    return g;
}

[[nodiscard]] EnvGrating parse_envgrating(const YAML::Node &node, const DeltabetaTable &db_table) {
    const auto z_start = get_scalar(node, "z_start");
    const auto dc0 = node["dc0"].as<std::array<double, 2>>();
    const auto dc1 = node["dc1"].as<std::array<double, 2>>();
    const auto thickness = get_scalar(node, "thickness");
    const auto pitch0 = get_scalar(node, "pitch0");
    const auto pitch1 = get_scalar(node, "pitch1");
    const auto nr_steps = node["nr_steps"].as<int>();
    auto x_positions = node["x_positions"].as<std::vector<double>>();
    const auto substrate_thickness = get_scalar(node, "substrate_thickness");

    const auto deltabeta_a = parse_optional_material(node, "mat_a", db_table);
    const auto deltabeta_b = parse_optional_material(node, "mat_b", db_table);
    const auto deltabeta_substrate = parse_optional_material(node, "mat_substrate", db_table);

    EnvGrating e;
    e.z_start = z_start;
    e.x_positions = std::move(x_positions);
    e.type = OpticalElementType::EnvGrating;
    e.thickness = thickness;
    e.pitch0 = pitch0;
    e.pitch1 = pitch1;
    e.dc0 = dc0;
    e.dc1 = dc1;
    e.nr_steps = nr_steps;
    e.substrate_thickness = substrate_thickness;
    e.deltabeta_a = deltabeta_a;
    e.deltabeta_b = deltabeta_b;
    e.deltabeta_substrate = deltabeta_substrate;

    return e;
}

[[nodiscard]] Sample parse_sample(const YAML::Node &node, const DeltabetaTable &db_table,
                                  const fs::path &sim_dir) {
    const auto z_start = get_scalar(node, "z_start");
    const auto pixel_size_x = get_scalar(node, "pixel_size_x");
    const auto pixel_size_z = get_scalar(node, "pixel_size_z");
    double pixel_size_y = node["pixel_size_y"] ? get_scalar(node, "pixel_size_y") : pixel_size_x;
    auto x_positions = node["x_positions"].as<std::vector<double>>();

    std::vector<Complex<double>> deltabetas = {{0., 0.}};
    const auto materials_node = node["materials"];
    for (std::size_t i = 0; i < materials_node.size(); ++i) {
        const auto mat = get_material(materials_node, i);
        deltabetas.emplace_back(db_table_lookup(mat, db_table));
    }

    const std::string grid_path = sim_dir / node["grid_path"].as<std::string>();
    npypp::MultiDimensionalArray<uint32_t> arr = npypp::LoadFull<uint32_t>(grid_path, false);

    Sample s;
    s.z_start = z_start;
    s.x_positions = std::move(x_positions);
    std::vector<double> y_positions;
    if (node["y_positions"]) {
        y_positions = node["y_positions"].as<std::vector<double>>();
    } else {
        // 默认所有相位步的 y 位置为 0
        y_positions.resize(x_positions.size(), 0.0);
    }
    s.y_positions = std::move(y_positions);
    s.type = OpticalElementType::Sample;
    s.pixel_size_x = pixel_size_x;
    s.pixel_size_y = pixel_size_y;  // 设置 y 方向像素尺寸
    s.pixel_size_z = pixel_size_z;
    s.deltabetas = std::move(deltabetas);
    s.grid = std::move(arr.data);
    // 根据网格维度确定尺寸
    if (arr.shape.size() == 2) {
        // 2D 网格 (z, x) - 1D 样品
        s.z_len = arr.shape[0];
        s.x_len = arr.shape[1];
        s.y_len = 1;  // y 方向尺寸为 1
    } else if (arr.shape.size() == 3) {
        // 3D 网格 (z, y, x) - 2D 样品
        s.z_len = arr.shape[0];
        s.y_len = arr.shape[1];
        s.x_len = arr.shape[2];
    } else {
        throw std::runtime_error("Sample grid must be 2D (z, x) or 3D (z, y, x)");
    }
    return s;
}

//precise_sample_update_begin
[[nodiscard]] Complex<double> lookup_material_density(const std::string& name, double density, const DeltabetaTable &db_table) {
    Complex<double> best_match{0., 0.};
    double best_diff = std::numeric_limits<double>::max();
    bool found = false;
    
    for (const auto &pair : db_table) {
        if (pair.first.name == name) {
            found = true;
            double density_diff = std::abs(pair.first.density - density);
            if (density_diff < best_diff) {
                best_diff = density_diff;
                best_match = pair.second;
            }
        }
    }
    
    if (!found) {
        throw std::runtime_error("Material " + name + " not found in deltabeta_table");
    }
    
    if (best_diff > 1e-5) {
        spdlog::warn("No exact density match for {}: requested {}, closest match diff {}", 
                    name, density, best_diff);
    }
    
    return best_match;
}
[[nodiscard]] PreciseSample parse_precise_sample(const YAML::Node &node, const DeltabetaTable &db_table,
                                                const fs::path &sim_dir) {
    const auto z_start = get_scalar(node, "z_start");
    const auto pixel_size_x = get_scalar(node, "pixel_size_x");
    const auto pixel_size_z = get_scalar(node, "pixel_size_z");
    auto x_positions = node["x_positions"].as<std::vector<double>>();

    // 解析材料列表
    std::vector<Material> materials;
    const auto materials_node = node["materials"];
    for (std::size_t i = 0; i < materials_node.size(); ++i) {
        materials.push_back(get_material(materials_node, i));
    }

    // 加载密度网格（dtype-aware：float64 网格自动转 float32）
    const std::string density_grid_path = sim_dir / node["density_grid_path"].as<std::string>();
    std::vector<size_t> density_shape;
    std::vector<float> density_data = load_float_grid(density_grid_path, density_shape);

    // 加载材料网格
    const std::string material_grid_path = sim_dir / node["material_grid_path"].as<std::string>();
    npypp::MultiDimensionalArray<uint32_t> material_arr = npypp::LoadFull<uint32_t>(material_grid_path, false);

    // 验证网格尺寸一致性
    if (density_shape[0] != material_arr.shape[0] || density_shape[1] != material_arr.shape[1]) {
        throw std::runtime_error("Density grid and material grid must have same dimensions");
    }

    // 关键修改：预先计算deltabeta网格
    // 参考Python版本：为每个像素计算实际的deltabeta值
    std::vector<Complex<double>> deltabeta_grid;
    deltabeta_grid.reserve(density_data.size());
    
    const std::size_t z_len = density_shape[0];
    const std::size_t x_len = density_shape[1];
    
    spdlog::info("Precomputing deltabeta grid for precise_sample: {} x {}", z_len, x_len);
    
    for (std::size_t z = 0; z < z_len; ++z) {
        for (std::size_t x = 0; x < x_len; ++x) {
            const std::size_t idx = z * x_len + x;
            const uint32_t material_idx = material_arr.data[idx];
            const float density = density_data[idx];
            
            if (material_idx == 0) {
                // 真空
                deltabeta_grid.push_back(Complex<double>{0., 0.});
            } else if (material_idx - 1 < uint32_t(materials.size())) {
                // 根据实际密度查找或计算deltabeta
                 const Material& material = materials[material_idx - 1];
                
                // // 查找匹配密度的deltabeta
                 Complex<double> best_match{0., 0.};
                // double best_diff = std::numeric_limits<double>::max();
                
                // for (const auto &pair : db_table) {
                //     if (pair.first.name == material.name) {
                //         double density_diff = std::abs(pair.first.density - density);
                //         if (density_diff < best_diff) {
                //             best_diff = density_diff;
                //             best_match = pair.second;
                //         }
                //     }
                // }
                
                // // if (best_diff > 1e-5) {
                // //     spdlog::warn("No exact density match for {}: requested {}, best match diff {}", 
                // //                 material.name, density, best_diff);
                // // }
                best_match=lookup_material_density(material.name,density,db_table);
                deltabeta_grid.push_back(best_match);
                
                // // 调试输出前几个像素
                // if (z == 0 && x < 5) {
                //     spdlog::info("  Pixel ({},{}): material={}, density={}, deltabeta=({}, {})",
                //                 z, x, material.name, density, best_match.real(), best_match.imag());
                // }
            } else {
                throw std::runtime_error("Material index " + std::to_string(material_idx) + 
                                       " out of range in precise_sample");
            }
        }
    }

    PreciseSample ps;
    ps.z_start = z_start;
    ps.x_positions = std::move(x_positions);
    ps.type = OpticalElementType::PreciseSample;
    ps.pixel_size_x = pixel_size_x;
    ps.pixel_size_z = pixel_size_z;
    ps.density_grid = std::move(density_data);
    ps.material_grid = std::move(material_arr.data);
    ps.deltabeta_grid = std::move(deltabeta_grid);  // 新增：存储预先计算的deltabeta网格
    ps.z_len = z_len;
    ps.x_len = x_len;
    
    spdlog::info("PreciseSample deltabeta grid computed: {} elements", ps.deltabeta_grid.size());
    
    return ps;
}
//precise_sample_update_end

//plasma_sample_begin
[[nodiscard]] PlasmaSample parse_plasma_sample(const YAML::Node &node,
                                               const fs::path &sim_dir) {
    const auto z_start = get_scalar(node, "z_start");
    const auto pixel_size_x = get_scalar(node, "pixel_size_x");
    const auto pixel_size_y = node["pixel_size_y"] ? get_scalar(node, "pixel_size_y") : pixel_size_x;
    const auto pixel_size_z = get_scalar(node, "pixel_size_z");
    auto x_positions = node["x_positions"].as<std::vector<double>>();
    std::vector<double> y_positions;
    if (node["y_positions"]) {
        y_positions = node["y_positions"].as<std::vector<double>>();
    } else {
        y_positions.resize(x_positions.size(), 0.0);
    }
    const int Z = node["Z"].as<int>();

    // dtype-aware loading: npypp::LoadFull<float> would misread float64 grids.
    std::vector<size_t> ne_shape, ni_shape, te_shape, zs_shape;
    auto ne_data = load_float_grid(sim_dir / node["ne_grid_path"].as<std::string>(), ne_shape);
    auto ni_data = load_float_grid(sim_dir / node["ni_grid_path"].as<std::string>(), ni_shape);
    auto te_data = load_float_grid(sim_dir / node["te_grid_path"].as<std::string>(), te_shape);
    auto zs_data = load_float_grid(sim_dir / node["zstar_grid_path"].as<std::string>(), zs_shape);

    if (ne_shape.size() != ni_shape.size() ||
        ne_shape.size() != te_shape.size() ||
        ne_shape.size() != zs_shape.size()) {
        throw std::runtime_error("plasma_sample grid shape ranks differ");
    }
    for (std::size_t d = 0; d < ne_shape.size(); ++d) {
        if (ne_shape[d] != ni_shape[d] ||
            ne_shape[d] != te_shape[d] ||
            ne_shape[d] != zs_shape[d]) {
            throw std::runtime_error("plasma_sample grid shape mismatch at dim " + std::to_string(d));
        }
    }

    std::size_t z_len, y_len, x_len;
    if (ne_shape.size() == 2) {
        z_len = ne_shape[0]; y_len = 1; x_len = ne_shape[1];
    } else if (ne_shape.size() == 3) {
        z_len = ne_shape[0]; y_len = ne_shape[1]; x_len = ne_shape[2];
    } else {
        throw std::runtime_error("plasma_sample grids must be 2D (z,x) or 3D (z,y,x)");
    }

    PlasmaSample ps;
    ps.z_start = z_start;
    ps.x_positions = std::move(x_positions);
    ps.y_positions = std::move(y_positions);
    ps.type = OpticalElementType::PlasmaSample;
    ps.pixel_size_x = pixel_size_x;
    ps.pixel_size_y = pixel_size_y;
    ps.pixel_size_z = pixel_size_z;
    ps.ne_grid = std::move(ne_data);
    ps.ni_grid = std::move(ni_data);
    ps.te_grid = std::move(te_data);
    ps.zstar_grid = std::move(zs_data);
    ps.Z = Z;
    ps.x_len = x_len;
    ps.y_len = y_len;
    ps.z_len = z_len;
    ps.deltabeta_grid.resize(z_len * y_len * x_len, Complex<double>{0., 0.});

    spdlog::info("Parsed PlasmaSample: {} x {} x {}, Z={}", z_len, y_len, x_len, Z);
    return ps;
}

void fill_plasma_deltabeta_grid(PlasmaSample &ps, double energy) {
    const std::size_t total = ps.z_len * ps.y_len * ps.x_len;
    ps.deltabeta_grid.resize(total);
    for (std::size_t i = 0; i < total; ++i) {
        const float ne = ps.ne_grid[i];
        if (ne <= 0.0f) {
            ps.deltabeta_grid[i] = Complex<double>{0., 0.};
            continue;
        }
        const float ni = ps.ni_grid[i];
        const float te = ps.te_grid[i];
        const float zs = ps.zstar_grid[i];
        double d, b, atlen;
        plasma_physics::plasma_delta_beta_host(
            static_cast<double>(ne), static_cast<double>(ni),
            static_cast<double>(te), static_cast<double>(zs),
            ps.Z, energy, d, b, atlen);
        ps.deltabeta_grid[i] = Complex<double>{d, b};
    }
    spdlog::info("Filled PlasmaSample deltabeta_grid ({} elements, energy={:.1f} eV)", total, energy);
}
//plasma_sample_end

[[nodiscard]] std::unique_ptr<OpticalElement> parse_optical_element(const YAML::Node &node,
                                                                    const DeltabetaTable &db_table,
                                                                    const fs::path &sim_dir) {
    const auto type = node["type"].as<std::string>();
    if (type == "grating") {
        return std::make_unique<Grating>(parse_grating(node, db_table));
    } else if (type == "env_grating") {
        return std::make_unique<EnvGrating>(parse_envgrating(node, db_table));
    } else if (type == "sample") {
        return std::make_unique<Sample>(parse_sample(node, db_table, sim_dir));
    } else if (type == "precise_sample") {
        return std::make_unique<PreciseSample>(parse_precise_sample(node, db_table, sim_dir));
    } else if (type == "plasma_sample") {
        return std::make_unique<PlasmaSample>(parse_plasma_sample(node, sim_dir));
    } else {
        throw std::runtime_error("Unknown optical element type: " + type);
    }
}

[[nodiscard]] std::vector<std::unique_ptr<OpticalElement>>
parse_optical_elements(const YAML::Node &node, const DeltabetaTable &db_table,
                       const fs::path &sim_dir) {
    std::vector<std::unique_ptr<OpticalElement>> elements;
    for (std::size_t i = 0; i < node.size(); ++i) {
        elements.push_back(parse_optical_element(node[i], db_table, sim_dir));
    }
    return elements;
}

[[nodiscard]] DeltabetaTable parse_deltabeta_table(const YAML::Node &node) {
    DeltabetaTable db_table;
    for (const auto &entry : node) {
        const auto mat = get_material(entry, 0);
        const Complex<double> db(get_scalar(entry[1], 0), get_scalar(entry[1], 1));
        //spdlog::info("entry10: {}", double(get_scalar(entry[1], 0)));

        db_table.emplace_back(mat, db);
    }
    return db_table;
}

// 在 config_parsing.cpp 中
// [[nodiscard]] SimParams parse_sim_params(const YAML::Node &node, double wl) {
//     // 检查是否存在2D参数
//     bool has_2d_params = node["nx"] && node["ny"];
//     SimParams params;
//     params.wl = wl;
//     params.is2d = node["is2d"].as<bool>();
    
//     if (has_2d_params) {
//         // 2D配置：解析2D参数
//         int nx = node["nx"].as<int>();
//         int ny = node["ny"].as<int>();
//         double dx = get_scalar(node, "dx");
//         double dy = node["dy"] ? get_scalar(node, "dy") : dx;  // 如果dy未提供，使用dx
        
//         // 设置2D参数
//         params.nx = nx;
//         params.ny = ny;
//         params.dx = dx;
//         params.dy = dy;
//         params.N = nx * ny;  // 计算总点数
        
//         // 解析探测器参数
//         params.z_detector = get_scalar(node, "z_detector");
        
//         // 探测器尺寸（优先使用2D参数）
//         if (node["detector_size_x"]) {
//             params.detector_size_x = get_scalar(node, "detector_size_x");
//             params.detector_size = *params.detector_size_x;  // 同步到1D参数
//         } else if (node["detector_size"]) {
//             params.detector_size = get_scalar(node, "detector_size");
//             params.detector_size_x = params.detector_size;
//         } else {
//             // 默认值：整个模拟区域的x尺寸
//             params.detector_size = nx * dx;
//             params.detector_size_x = nx * dx;
//         }
        
//         if (node["detector_size_y"]) {
//             params.detector_size_y = get_scalar(node, "detector_size_y");
//         } else {
//             // 默认值：整个模拟区域的y尺寸
//             params.detector_size_y = ny * dy;
//         }
        
//         // 像素尺寸
//         if (node["detector_pixel_size_x"]) {
//             params.detector_pixel_size_x = get_scalar(node, "detector_pixel_size_x");
//         } else {
//             params.detector_pixel_size_x = dx;
//         }
        
//         if (node["detector_pixel_size_y"]) {
//             params.detector_pixel_size_y = get_scalar(node, "detector_pixel_size_y");
//         } else {
//             params.detector_pixel_size_y = dy;
//         }
        
//     } else {
//         // 1D配置（向后兼容）
//         params.N = node["N"].as<int>();
//         params.dx = get_scalar(node, "dx");
//         params.z_detector = get_scalar(node, "z_detector");
        
//         // 探测器尺寸
//         if (node["detector_size"]) {
//             params.detector_size = get_scalar(node, "detector_size");
//         } else {
//             params.detector_size = params.N * params.dx;  // 默认值
//         }
        
//         // 像素尺寸
//         if (node["detector_pixel_size_x"]) {
//             params.detector_pixel_size_x = get_scalar(node, "detector_pixel_size_x");
//         } else {
//             params.detector_pixel_size_x = params.dx;  // 默认等于dx
//         }
        
//         if (node["detector_pixel_size_y"]) {
//             params.detector_pixel_size_y = get_scalar(node, "detector_pixel_size_y");
//         } else {
//             params.detector_pixel_size_y = 1.0;  // 1D默认值，保持现有行为
//         }
        
//         // 对于1D配置，也设置2D参数为nx=N, ny=1
//         params.nx = params.N;
//         params.ny = 1;
//         params.dy = params.dx;  // dy等于dx
//         params.detector_size_x = params.detector_size;
//         params.detector_size_y = params.detector_pixel_size_y;  // 使用y方向像素尺寸
//     }
    
//     return params;
// }

[[nodiscard]] SimParams parse_sim_params(const YAML::Node &node, double wl) {
    SimParams params;
    params.wl = wl;

    bool has_2d = node["nx"] && node["ny"];
    params.is2d = has_2d;   // ⬅ 设置标志

    // Fresnel scaling: 需要显式启用
    if (has_2d && node["use_fresnel_scaling"]) {
        const auto &fs_node = node["use_fresnel_scaling"];
        if (fs_node.IsScalar()) {
            // 支持 YAML 布尔值 (true) 和字符串 ('true')
            params.use_fresnel_scaling = fs_node.as<bool>() ||
                (fs_node.as<std::string>("") == "true");
        }
    }

    if (has_2d) {
        params.nx = node["nx"].as<int>();
        params.ny = node["ny"].as<int>();
        params.dx = get_scalar(node, "dx");
        params.dy = node["dy"] ? get_scalar(node, "dy") : params.dx;
        params.N = params.nx * params.ny;
        params.z_detector = get_scalar(node, "z_detector");

        // 探测器尺寸
        if (node["detector_size_x"]) {
            params.detector_size_x = get_scalar(node, "detector_size_x");
            params.detector_size = params.detector_size_x;
        } else if (node["detector_size"]) {
            params.detector_size = get_scalar(node, "detector_size");
            params.detector_size_x = params.detector_size;
        } else {
            params.detector_size = params.nx * params.dx;
            params.detector_size_x = params.detector_size;
        }

        if (node["detector_size_y"]) {
            params.detector_size_y = get_scalar(node, "detector_size_y");
        } else {
            params.detector_size_y = params.ny * params.dy;
        }

        if (node["detector_pixel_size_x"])
            params.detector_pixel_size_x = get_scalar(node, "detector_pixel_size_x");
        else
            params.detector_pixel_size_x = params.dx;

        if (node["detector_pixel_size_y"])
            params.detector_pixel_size_y = get_scalar(node, "detector_pixel_size_y");
        else
            params.detector_pixel_size_y = params.dy;
    } else {
        // 1D 配置
        params.N = node["N"].as<int>();
        params.dx = get_scalar(node, "dx");
        params.z_detector = get_scalar(node, "z_detector");

        if (node["detector_size"]) {
            params.detector_size = get_scalar(node, "detector_size");
        } else {
            params.detector_size = params.N * params.dx;
        }

        if (node["detector_pixel_size_x"])
            params.detector_pixel_size_x = get_scalar(node, "detector_pixel_size_x");
        else
            params.detector_pixel_size_x = params.dx;

        if (node["detector_pixel_size_y"])
            params.detector_pixel_size_y = get_scalar(node, "detector_pixel_size_y");
        else
            params.detector_pixel_size_y = 1.0;

        // 为兼容旧代码，显式设置 nx, ny, dy 的默认值（即使 1D 不会用到）
        params.nx = params.N;      // 或者保持 0，但 get_nx() 会返回 N，此处不关键
        params.ny = 1;             // ⬅ 明确设为 1
        params.dy = params.dx;
        params.detector_size_x = params.detector_size;
        params.detector_size_y = params.detector_pixel_size_y;
    }

    return params;
}

[[nodiscard]] PointSource parse_point_source(const YAML::Node &node) {
    const auto x = get_scalar(node, "x");
    const auto z = get_scalar(node, "z");
    PointSource s;
    s.type = SourceType::Point;
    s.x = x;
    s.z = z;
    if (node["y"]) {
        s.y = get_scalar(node, "y");
    } else {
        s.y = std::nullopt;  // 1D仿真中没有y坐标
    }
    return s;
}

[[nodiscard]] VectorSource parse_vector_source(const YAML::Node &node) {
    const auto input_path = node["input_path"].as<std::string>();
    const auto z = get_scalar(node, "z");
    VectorSource s;
    s.type = SourceType::Vector;
    s.input_path = std::move(input_path);
    s.z = z;
    return s;
}

[[nodiscard]] std::unique_ptr<Source> parse_source(const YAML::Node &node) {
    const auto type = node["type"].as<std::string>();

    if (type == "point") {
        return std::make_unique<PointSource>(parse_point_source(node));
    } else if (type == "vector") {
        return std::make_unique<VectorSource>(parse_vector_source(node));
    } else {
        throw std::runtime_error("unknown source type: " + type);
    }
}

[[nodiscard]] DType parse_dtype(const YAML::Node &node) {
    if (!node.IsDefined()) {
        throw std::runtime_error("dtype is not defined in config");
    }

    const auto dtype = node.as<std::string>();
    if (dtype == "c8") {
        return DType::C8;
    } else if (dtype == "c16") {
        return DType::C16;
    } else {
        throw std::runtime_error("Unknown dtype: " + dtype);
    }
}

[[nodiscard]] std::vector<double> parse_cutoff_angles(const YAML::Node &node) {
    std::vector<double> angles;
    for (std::size_t i = 0; i < node.size(); ++i) {
        angles.push_back(get_scalar(node, i));
    }
    return angles;
}

std::string zeropad(int number, std::size_t length) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(length) << number;
    return ss.str();
}

[[nodiscard]] fs::path get_subdir(const fs::path &sim_dir, int source_idx) {
    return sim_dir / zeropad(source_idx, 8);
}

[[nodiscard]] Config parse_config(const fs::path &sim_dir, int source_idx) {
    const auto config_path = sim_dir / "config.yaml";
    const auto subconfig_path = get_subdir(sim_dir, source_idx) / "subconfig.yaml";
    const auto computed_path = sim_dir / "computed.yaml";

    YAML::Node config_node = YAML::LoadFile(config_path.string());
    YAML::Node subconfig_node = YAML::LoadFile(subconfig_path.string());
    YAML::Node computed_node = YAML::LoadFile(computed_path.string());

    const DType dtype = parse_dtype(config_node["dtype"]);
    const double energy = get_scalar(subconfig_node, "energy");

    const SimParams sim_params =
        parse_sim_params(config_node["sim_params"], convert_energy_wavelength(energy));
    const auto db_table = parse_deltabeta_table(subconfig_node["deltabeta_table"]);
    //test
    auto optical_elements = parse_optical_elements(config_node["elements"], db_table, sim_dir);
    auto source = parse_source(subconfig_node["source"]);

    auto cutoff_angles = parse_cutoff_angles(computed_node["cutoff_angles"]);

    const bool save_final_u_vectors = config_node["save_final_u_vectors"].as<bool>();
    const bool save_debug_wavefields = config_node["save_debug_wavefields"] ? config_node["save_debug_wavefields"].as<bool>() : false;

    return Config{
        sim_params, std::move(optical_elements), std::move(source),
        dtype,      save_final_u_vectors,        save_debug_wavefields,
        std::move(cutoff_angles),
    };
}
