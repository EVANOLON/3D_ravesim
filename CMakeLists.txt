cmake_minimum_required(VERSION 3.24)

project(rave_sim LANGUAGES CXX CUDA)

# Keep the repository-root build command working while maintaining a single
# authoritative CUDA implementation under fast-wave/.
add_subdirectory(fast-wave)
