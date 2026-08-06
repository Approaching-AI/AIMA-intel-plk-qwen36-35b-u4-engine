cmake_minimum_required(VERSION 3.24)

project(intel_qwen36_engine LANGUAGES CXX)

find_package(Threads REQUIRED)
find_package(OpenCL REQUIRED)
find_library(IQ36_LEVEL_ZERO_LOADER NAMES ze_loader REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(iq36_core
  src/gguf_loader.cpp
  src/gpu_q4_cpu_order_matvec.cpp
  src/gpu_q4x8_matvec.cpp
  src/grouped_s8_u4_prefill_runtime.cpp
  src/packed_token_schedule.cpp
  src/packed_token_level_zero_backend.cpp
  src/resident_harness.cpp
  src/layer.cpp
  src/loop.cpp
)

target_include_directories(iq36_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(iq36_core
  PUBLIC
    Threads::Threads
    OpenCL::OpenCL
    ${IQ36_LEVEL_ZERO_LOADER}
    ${CMAKE_DL_LIBS}
)

# Offline provider dispatch recorder. It is injected only into source-capture
# workers and is never part of the promoted inference runtime.
add_library(iq36-opencl-dispatch-trace SHARED
  tools/opencl_dispatch_trace.cpp
)
target_link_libraries(iq36-opencl-dispatch-trace
  PRIVATE
    Threads::Threads
    ${CMAKE_DL_LIBS}
)
target_include_directories(iq36-opencl-dispatch-trace
  PRIVATE
    ${OpenCL_INCLUDE_DIRS}
)
set_target_properties(iq36-opencl-dispatch-trace PROPERTIES PREFIX "")

if(MSVC)
  target_compile_options(iq36_core PRIVATE /W4)
else()
  target_compile_options(iq36_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()

# ---------------------------------------------------------------------------
# Test/compare targets are generated from the single boundary registry,
# engine/boundaries.json, instead of one hand-written stanza per boundary.
# Add a boundary there (and its tests/<name>_compare.cpp); do not edit this
# file per boundary.  See AGENTS.md "Read order" and tools/iq36-ladder.py.
# ---------------------------------------------------------------------------
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/boundaries.json IQ36_BOUNDARIES_JSON)

function(iq36_add_target target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE iq36_core)
endfunction()

function(iq36_apply_json_compile_definitions target section index)
  string(JSON _definition_count ERROR_VARIABLE _definition_error
    LENGTH "${IQ36_BOUNDARIES_JSON}" ${section} ${index} compile_definitions)
  if(NOT _definition_error)
    math(EXPR _definition_last "${_definition_count} - 1")
    foreach(j RANGE ${_definition_last})
      string(JSON _definition GET "${IQ36_BOUNDARIES_JSON}"
        ${section} ${index} compile_definitions ${j})
      target_compile_definitions(${target} PRIVATE ${_definition})
    endforeach()
  endif()
endfunction()

# Infrastructure targets (loader/inspect/harness checks).
string(JSON IQ36_INFRA_COUNT LENGTH "${IQ36_BOUNDARIES_JSON}" infra_targets)
math(EXPR IQ36_INFRA_LAST "${IQ36_INFRA_COUNT} - 1")
foreach(i RANGE ${IQ36_INFRA_LAST})
  string(JSON _tgt GET "${IQ36_BOUNDARIES_JSON}" infra_targets ${i} target)
  string(JSON _src GET "${IQ36_BOUNDARIES_JSON}" infra_targets ${i} source)
  iq36_add_target(${_tgt} ${_src})
  iq36_apply_json_compile_definitions(${_tgt} infra_targets ${i})
endforeach()

# The self-test runs standalone (no model/oracle args) and is the ctest gate.
add_test(NAME iq36_self_test COMMAND iq36-self-test)

# Boundary compare targets.  These require the locked model + oracle payloads as
# arguments and run on the Intel target, so they are built here but not
# registered with ctest; batch status comes from tools/iq36-ladder.py.
string(JSON IQ36_BOUNDARY_COUNT LENGTH "${IQ36_BOUNDARIES_JSON}" boundaries)
math(EXPR IQ36_BOUNDARY_LAST "${IQ36_BOUNDARY_COUNT} - 1")
foreach(i RANGE ${IQ36_BOUNDARY_LAST})
  string(JSON _tgt GET "${IQ36_BOUNDARIES_JSON}" boundaries ${i} target)
  string(JSON _src GET "${IQ36_BOUNDARIES_JSON}" boundaries ${i} source)
  iq36_add_target(${_tgt} ${_src})
  iq36_apply_json_compile_definitions(${_tgt} boundaries ${i})
endforeach()
