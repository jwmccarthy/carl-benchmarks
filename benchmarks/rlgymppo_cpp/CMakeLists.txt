cmake_minimum_required(VERSION 3.18)
project(carl_benchmark_rlgymppo_cpp LANGUAGES CXX)

set(RLGYMPPO_CPP_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../rlgym-ppo-cpp/RLGymPPO_CPP" CACHE PATH
    "Unmodified rlgym-ppo-cpp library source")
set(RLGYM_COLLISION_MESHES_DIR "" CACHE PATH "RLGym collision_meshes directory")
if(NOT EXISTS "${RLGYMPPO_CPP_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Set RLGYMPPO_CPP_SOURCE_DIR to rlgym-ppo-cpp/RLGymPPO_CPP")
endif()
if(NOT EXISTS "${RLGYM_COLLISION_MESHES_DIR}/soccar")
    message(FATAL_ERROR "Set RLGYM_COLLISION_MESHES_DIR to RLGym's collision_meshes directory")
endif()

set(RLGYMPPO_CPP_COMPAT_DIR "${CMAKE_BINARY_DIR}/rlgymppo_cpp_compat")
file(COPY "${RLGYMPPO_CPP_SOURCE_DIR}/" DESTINATION "${RLGYMPPO_CPP_COMPAT_DIR}")

function(replace_in_file path before after)
    file(READ "${path}" contents)
    string(REPLACE "${before}" "${after}" contents "${contents}")
    file(WRITE "${path}" "${contents}")
endfunction()

replace_in_file(
    "${RLGYMPPO_CPP_COMPAT_DIR}/src/public/RLGymPPO_CPP/Util/Timer.h"
    "std::chrono::high_resolution_clock::now()"
    "std::chrono::steady_clock::now()")
replace_in_file(
    "${RLGYMPPO_CPP_COMPAT_DIR}/src/private/RLGymPPO_CPP/Util/gradscaler.hpp"
    "assert(found_infs.size() > 0, \"No inf checks were recorded prior to update.\");"
    "TORCH_CHECK(found_infs.size() > 0, \"No inf checks were recorded prior to update.\");")
replace_in_file(
    "${RLGYMPPO_CPP_COMPAT_DIR}/src/private/RLGymPPO_CPP/Util/gradscaler.hpp"
    "inline auto sum(PerDeviceTensors const& per_device)"
    "inline Type sum(PerDeviceTensors const& per_device)")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory("${RLGYMPPO_CPP_COMPAT_DIR}" rlgymppo_cpp_dependency)
add_executable(rlgymppo_cpp_benchmark runner.cpp)
target_compile_features(rlgymppo_cpp_benchmark PRIVATE cxx_std_20)
target_link_libraries(rlgymppo_cpp_benchmark PRIVATE RLGymPPO_CPP)
target_compile_definitions(rlgymppo_cpp_benchmark PRIVATE
    COLLISION_MESHES_DIR="${RLGYM_COLLISION_MESHES_DIR}")
