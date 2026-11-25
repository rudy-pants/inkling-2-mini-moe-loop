cmake_minimum_required(VERSION 3.20)
project(moe-loop LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# The prebuilt llama.cpp we link against, and the exact source tree it was built from (commit 189f431bf).
get_filename_component(INKLING_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(LLAMA_SRC_DIR "${INKLING_ROOT}/llama.cpp-src" CACHE PATH "llama.cpp source tree matching the prebuilt libs")
set(LLAMA_LIB_DIR "${INKLING_ROOT}/llama-rocm"    CACHE PATH "directory with the prebuilt libllama.so / libggml*.so")
set(MOE_DEFAULT_MODEL "${INKLING_ROOT}/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf"
    CACHE FILEPATH "model used when -m is not given")

foreach(f "${LLAMA_SRC_DIR}/include/llama.h" "${LLAMA_LIB_DIR}/libllama.so")
    if(NOT EXISTS "${f}")
        message(FATAL_ERROR "missing ${f} (set LLAMA_SRC_DIR / LLAMA_LIB_DIR)")
    endif()
endforeach()

add_library(llama_prebuilt INTERFACE)
target_include_directories(llama_prebuilt SYSTEM INTERFACE "${LLAMA_SRC_DIR}/include" "${LLAMA_SRC_DIR}/ggml/include")
target_link_libraries(llama_prebuilt INTERFACE
    "${LLAMA_LIB_DIR}/libllama.so" "${LLAMA_LIB_DIR}/libggml.so" "${LLAMA_LIB_DIR}/libggml-base.so")
# Runtime lookup of the prebuilt libs (and the ROCm libs bundled next to them).
target_link_options(llama_prebuilt INTERFACE "LINKER:-rpath,${LLAMA_LIB_DIR}" "LINKER:--disable-new-dtags")

set(MOE_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)

# Code that needs no model or GPU: unit-testable on its own.
add_library(moe_core STATIC src/trace.cpp src/permutation.cpp src/cachesim.cpp src/ordering.cpp)
target_include_directories(moe_core PUBLIC include)
target_compile_options(moe_core PRIVATE ${MOE_WARNINGS})

# GGUF metadata access (libggml-base only, no GPU).
add_library(moe_gguf STATIC src/gguf_layout.cpp src/gguf_permute.cpp)
target_include_directories(moe_gguf PUBLIC include)
target_include_directories(moe_gguf SYSTEM PUBLIC "${LLAMA_SRC_DIR}/ggml/include")
target_link_libraries(moe_gguf PUBLIC moe_core "${LLAMA_LIB_DIR}/libggml-base.so")
target_link_options(moe_gguf PUBLIC "LINKER:-rpath,${LLAMA_LIB_DIR}" "LINKER:--disable-new-dtags")
target_compile_options(moe_gguf PRIVATE ${MOE_WARNINGS})

# Code that talks to libllama.
add_library(moe_llama STATIC src/model_info.cpp)
target_include_directories(moe_llama PUBLIC include)
target_link_libraries(moe_llama PUBLIC moe_core moe_gguf llama_prebuilt)
target_compile_options(moe_llama PRIVATE ${MOE_WARNINGS})

function(moe_tool name)
    add_executable(${name} tools/${name}.cpp)
    target_link_libraries(${name} PRIVATE ${ARGN})
    target_compile_definitions(${name} PRIVATE MOE_DEFAULT_MODEL="${MOE_DEFAULT_MODEL}" MOE_BACKEND_DIR="${LLAMA_LIB_DIR}")
    target_compile_options(${name} PRIVATE ${MOE_WARNINGS})
endfunction()

moe_tool(moe-smoke moe_llama)
moe_tool(moe-trace moe_llama)
moe_tool(moe-dump  moe_core)
moe_tool(moe-analyze moe_gguf)
moe_tool(moe-permute moe_gguf)

# Tests
enable_testing()
add_library(doctest INTERFACE)
target_include_directories(doctest SYSTEM INTERFACE third_party)

add_executable(moe-tests tests/test_main.cpp tests/test_layout.cpp tests/test_trace.cpp tests/test_stats.cpp tests/test_permute.cpp)
target_link_libraries(moe-tests PRIVATE moe_core moe_gguf doctest)
target_compile_options(moe-tests PRIVATE ${MOE_WARNINGS})
add_test(NAME unit COMMAND moe-tests)

# Needs the GPU and the model: opt in with -DMOE_MODEL_TESTS=ON (ctest -L model).
option(MOE_MODEL_TESTS "register tests that load the real model" OFF)
if(MOE_MODEL_TESTS)
    add_test(NAME smoke_metadata COMMAND moe-smoke --info-only)
    set_tests_properties(smoke_metadata PROPERTIES LABELS model PASS_REGULAR_EXPRESSION "metadata check  OK")
endif()
