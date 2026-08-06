// Copyright (C) 2021-2025 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Minimal ABI prefix from Intel's level-zero-vpu-extensions ze_graph_ext.h.
// Source revision: c7cb5d218ca14f6a81b3ef0bb89e718e9fcdba8e.
// Upstream: https://github.com/intel/level-zero-vpu-extensions
//
// Only the version-1.0 graph ABI used by the model-specific native-blob
// preflight is declared here.  The driver returns a larger, backward-
// compatible DDI table; this prefix preserves the upstream field order.

#pragma once

#include <level_zero/ze_api.h>

#include <cstddef>
#include <cstdint>

#define IQ36_ZE_GRAPH_EXT_NAME "ZE_extension_graph"
#define IQ36_ZE_MAX_GRAPH_ARGUMENT_NAME 256
#define IQ36_ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE 5

extern "C" {

typedef struct _ze_graph_handle_t* iq36_ze_graph_handle_t;
typedef struct _ze_graph_profiling_query_handle_t*
    iq36_ze_graph_profiling_query_handle_t;

enum iq36_ze_graph_ext_version_t : std::uint32_t {
  IQ36_ZE_GRAPH_EXT_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),
};

enum iq36_ze_graph_format_t : std::uint32_t {
  IQ36_ZE_GRAPH_FORMAT_NATIVE = 0x1,
  IQ36_ZE_GRAPH_FORMAT_NGRAPH_LITE = 0x2,
};

struct iq36_ze_graph_compiler_version_info_t {
  std::uint16_t major;
  std::uint16_t minor;
};

enum iq36_ze_structure_type_graph_ext_t : std::uint32_t {
  IQ36_ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES = 0x1,
  IQ36_ZE_STRUCTURE_TYPE_GRAPH_DESC = 0x2,
  IQ36_ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES = 0x3,
  IQ36_ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES = 0x4,
};

struct iq36_ze_device_graph_properties_t {
  iq36_ze_structure_type_graph_ext_t stype;
  void* pNext;
  iq36_ze_graph_ext_version_t graphExtensionVersion;
  iq36_ze_graph_compiler_version_info_t compilerVersion;
  iq36_ze_graph_format_t graphFormatsSupported;
  std::uint32_t maxOVOpsetVersionSupported;
};

enum iq36_ze_graph_argument_precision_t : std::uint32_t {
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN = 0x00,
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_FP32 = 0x01,
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_FP16 = 0x02,
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_UINT16 = 0x03,
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_UINT8 = 0x04,
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_INT32 = 0x05,
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_INT16 = 0x06,
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_INT8 = 0x07,
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_BIN = 0x08,
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_BF16 = 0x09,
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_UINT32 = 0x0A,
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_UINT4 = 0x0B,
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_INT4 = 0x0C,
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_DYNAMIC = 0x0D,
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_BOOLEAN = 0x0E,
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_FP64 = 0x0F,
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_UINT64 = 0x10,
  IQ36_ZE_GRAPH_ARGUMENT_PRECISION_INT64 = 0x11,
};

enum iq36_ze_graph_argument_layout_t : std::uint32_t {
  IQ36_ZE_GRAPH_ARGUMENT_LAYOUT_ANY = 0x00,
};

struct iq36_ze_graph_desc_t {
  iq36_ze_structure_type_graph_ext_t stype;
  void* pNext;
  iq36_ze_graph_format_t format;
  std::size_t inputSize;
  const std::uint8_t* pInput;
  const char* pBuildFlags;
};

struct iq36_ze_graph_properties_t {
  iq36_ze_structure_type_graph_ext_t stype;
  void* pNext;
  std::uint32_t numGraphArgs;
};

enum iq36_ze_graph_argument_type_t : std::uint32_t {
  IQ36_ZE_GRAPH_ARGUMENT_TYPE_INPUT = 0,
  IQ36_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT = 1,
};

struct iq36_ze_graph_argument_properties_t {
  iq36_ze_structure_type_graph_ext_t stype;
  void* pNext;
  char name[IQ36_ZE_MAX_GRAPH_ARGUMENT_NAME];
  iq36_ze_graph_argument_type_t type;
  std::uint32_t dims[IQ36_ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE];
  iq36_ze_graph_argument_precision_t networkPrecision;
  iq36_ze_graph_argument_layout_t networkLayout;
  iq36_ze_graph_argument_precision_t devicePrecision;
  iq36_ze_graph_argument_layout_t deviceLayout;
};

using iq36_ze_pfn_device_get_graph_properties_t = ze_result_t(ZE_APICALL*)(
    ze_device_handle_t, iq36_ze_device_graph_properties_t*);
using iq36_ze_pfn_graph_create_t = ze_result_t(ZE_APICALL*)(
    ze_context_handle_t, ze_device_handle_t, const iq36_ze_graph_desc_t*,
    iq36_ze_graph_handle_t*);
using iq36_ze_pfn_graph_destroy_t =
    ze_result_t(ZE_APICALL*)(iq36_ze_graph_handle_t);
using iq36_ze_pfn_graph_get_properties_t = ze_result_t(ZE_APICALL*)(
    iq36_ze_graph_handle_t, iq36_ze_graph_properties_t*);
using iq36_ze_pfn_graph_get_argument_properties_t = ze_result_t(ZE_APICALL*)(
    iq36_ze_graph_handle_t, std::uint32_t,
    iq36_ze_graph_argument_properties_t*);
using iq36_ze_pfn_graph_set_argument_value_t = ze_result_t(ZE_APICALL*)(
    iq36_ze_graph_handle_t, std::uint32_t, const void*);
using iq36_ze_pfn_append_graph_initialize_t = ze_result_t(ZE_APICALL*)(
    ze_command_list_handle_t, iq36_ze_graph_handle_t, ze_event_handle_t,
    std::uint32_t, ze_event_handle_t*);
using iq36_ze_pfn_append_graph_execute_t = ze_result_t(ZE_APICALL*)(
    ze_command_list_handle_t, iq36_ze_graph_handle_t,
    iq36_ze_graph_profiling_query_handle_t, ze_event_handle_t, std::uint32_t,
    ze_event_handle_t*);
using iq36_ze_pfn_graph_get_native_binary_t = ze_result_t(ZE_APICALL*)(
    iq36_ze_graph_handle_t, std::size_t*, std::uint8_t*);

struct iq36_ze_graph_dditable_ext_t {
  iq36_ze_pfn_graph_create_t pfnCreate;
  iq36_ze_pfn_graph_destroy_t pfnDestroy;
  iq36_ze_pfn_graph_get_properties_t pfnGetProperties;
  iq36_ze_pfn_graph_get_argument_properties_t pfnGetArgumentProperties;
  iq36_ze_pfn_graph_set_argument_value_t pfnSetArgumentValue;
  iq36_ze_pfn_append_graph_initialize_t pfnAppendGraphInitialize;
  iq36_ze_pfn_append_graph_execute_t pfnAppendGraphExecute;
  iq36_ze_pfn_graph_get_native_binary_t pfnGetNativeBinary;
  iq36_ze_pfn_device_get_graph_properties_t pfnDeviceGetGraphProperties;
};

}  // extern "C"
