/*
 * Copyright 2025 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef TEST_TORX_HELPERS_H
#define TEST_TORX_HELPERS_H

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

/**
 * Detect the device ID of the local NPU hardware.
 *
 * Probes /dev/accel via the pipe loader and returns the device id
 * string (e.g. "ethosu-65-256-98304"), or empty string on failure.
 */
std::string
detect_device_id();

/**
 * Metadata loaded from build-time artifacts (meta.json).
 */
struct TestMeta {
   bool ok = false;
   std::string error;
   std::vector<float> output_scales;
   std::vector<int> output_zero_points;
   std::vector<float> max_quant_errors;
};

/**
 * Discover all per-op test cases by scanning for device-specific .pte files.
 *
 * Returns paths relative to the work dir,
 * e.g. "mobilenet_v2/000".
 */
std::vector<std::string>
get_model_files(const std::string &device_id);

/**
 * Discover whole-model test cases by scanning for device-specific .pte files
 * directly inside a model directory (no op subdirectory).
 *
 * Returns model names, e.g. "mobilenet_v2".
 */
std::vector<std::string>
get_whole_model_files(const std::string &device_id);

/**
 * Load build-time metadata from a device-specific .json file.
 */
TestMeta
load_meta(const std::filesystem::path &meta_path);

/**
 * Run inference on the local NPU.
 *
 * Loads the .pte model from pte_path and device-specific float input from
 * work_dir, runs ExecuTorch inference, and writes output to
 * work_dir/output_0.bin.
 *
 * Returns 0 on success.
 */
int
run_on_npu(const std::filesystem::path &work_dir,
           const std::filesystem::path &pte_path,
           const std::string &device_id);

/**
 * Load a raw binary file as a vector of float32 values.
 */
std::vector<float>
load_tensor_float(const std::filesystem::path &path);

/**
 * Get the test data directory (TORX_TEST_DATA or torx_tests/).
 */
std::filesystem::path
get_work_dir();

#endif /* TEST_TORX_HELPERS_H */
