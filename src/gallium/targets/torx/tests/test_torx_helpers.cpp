/*
 * Copyright 2025 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 *
 * SPDX-License-Identifier: MIT
 */

#include "test_torx_helpers.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <unistd.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include "util/parson.h"
#include "torx_backend.h"

/* ------------------------------------------------------------------ */
/* detect_device_id — probe local NPU hardware                         */
/* ------------------------------------------------------------------ */

std::string
detect_device_id()
{
   struct torx_backend *backend = torx_backend_create();
   if (!backend)
      return {};

   std::string id;
   if (backend->ml_dev && backend->ml_dev->id)
      id = backend->ml_dev->id;

   torx_backend_destroy(backend);
   return id;
}

/* ------------------------------------------------------------------ */
/* get_model_files — scan build artifacts for test cases               */
/* ------------------------------------------------------------------ */

std::vector<std::string>
get_model_files(const std::string &device_id)
{
   std::vector<std::string> tests;

   auto work_dir = get_work_dir();
   if (!std::filesystem::exists(work_dir)) {
      fprintf(stderr, "Test artifacts directory not found: %s\n"
              "Run 'meson compile -C build' to generate test data.\n",
              work_dir.c_str());
      return tests;
   }

   /* Scan for <device>.pte files in <work_dir>/<model>/<op>/<device>.pte */
   std::string pte_name = device_id + ".pte";
   for (auto &entry : std::filesystem::recursive_directory_iterator(work_dir)) {
      if (entry.path().filename() != pte_name)
         continue;

      auto rel = std::filesystem::relative(entry.path().parent_path(),
                                           work_dir);
      /* Only include two-level paths (model/op), skip single-level (whole-model) */
      if (rel.string().find('/') == std::string::npos)
         continue;
      tests.push_back(rel.string());
   }

   std::sort(tests.begin(), tests.end());
   return tests;
}

/* ------------------------------------------------------------------ */
/* get_whole_model_files — scan for whole-model test cases             */
/* ------------------------------------------------------------------ */

std::vector<std::string>
get_whole_model_files(const std::string &device_id)
{
   std::vector<std::string> models;

   auto work_dir = get_work_dir();
   if (!std::filesystem::exists(work_dir))
      return models;

   /* Scan for <device>.pte directly in <work_dir>/<model>/<device>.pte */
   std::string pte_name = device_id + ".pte";
   for (auto &entry : std::filesystem::directory_iterator(work_dir)) {
      if (!entry.is_directory())
         continue;
      auto pte_path = entry.path() / pte_name;
      if (std::filesystem::exists(pte_path))
         models.push_back(entry.path().filename().string());
   }

   std::sort(models.begin(), models.end());
   return models;
}

/* ------------------------------------------------------------------ */
/* load_meta — read build-time metadata from meta.json                 */
/* ------------------------------------------------------------------ */

TestMeta
load_meta(const std::filesystem::path &meta_path)
{
   TestMeta meta{};

   if (!std::filesystem::exists(meta_path)) {
      meta.ok = false;
      meta.error = "not found: " + meta_path.string();
      return meta;
   }

   JSON_Value *root = json_parse_file(meta_path.c_str());
   if (!root) {
      meta.ok = false;
      meta.error = "Failed to parse " + meta_path.string();
      return meta;
   }

    JSON_Object *obj = json_value_get_object(root);

    size_t num_outputs = (size_t)json_object_get_number(obj, "num_outputs");
    JSON_Array *scales_arr = json_object_get_array(obj, "output_scales");
    JSON_Array *zps_arr = json_object_get_array(obj, "output_zero_points");
    JSON_Array *errors_arr = json_object_get_array(obj, "max_quant_errors");

    if (num_outputs > 0 && scales_arr && zps_arr && errors_arr) {
        for (size_t i = 0; i < num_outputs; i++) {
            meta.output_scales.push_back((float)json_array_get_number(scales_arr, i));
            meta.output_zero_points.push_back((int)json_array_get_number(zps_arr, i));
            meta.max_quant_errors.push_back((float)json_array_get_number(errors_arr, i));
        }
        meta.ok = true;
    } else {
        meta.ok = false;
        meta.error = "Missing required fields in " + meta_path.string();
    }

    json_value_free(root);
   return meta;
}

/* ------------------------------------------------------------------ */
/* run_on_npu — local ExecuTorch inference                             */
/* ------------------------------------------------------------------ */

int
run_on_npu(const std::filesystem::path &work_dir,
           const std::filesystem::path &pte_path,
           const std::string &device_id)
{
   using namespace ::executorch::extension;

   /* Load the model. */
   Module module(pte_path.string());

   auto meta = module.method_meta("forward");
   if (!meta.ok()) {
      fprintf(stderr, "method_meta(\"forward\") failed\n");
      return 1;
   }

   size_t num_inputs = meta->num_inputs();
   std::vector<std::vector<float>> input_datas(num_inputs);
   std::vector<::executorch::extension::TensorPtr> inputs;

   for (size_t i = 0; i < num_inputs; i++) {
      auto input_meta = meta->input_tensor_meta(i);
      if (!input_meta.ok()) {
         fprintf(stderr, "input_tensor_meta(%zu) failed\n", i);
         return 1;
      }

      std::vector<int> shape;
      int64_t numel = 1;
      for (size_t d = 0; d < input_meta->sizes().size(); d++) {
         shape.push_back((int)input_meta->sizes()[d]);
         numel *= input_meta->sizes()[d];
      }

      auto input_path = work_dir / (device_id + "-input-" + std::to_string(i) + ".data");
      input_datas[i].resize(numel);
      {
         std::ifstream in(input_path, std::ios::binary);
         if (!in.is_open()) {
            fprintf(stderr, "Cannot open %s\n", input_path.c_str());
            return 1;
         }
         in.read(reinterpret_cast<char *>(input_datas[i].data()),
                 (std::streamsize)(numel * sizeof(float)));
      }

      inputs.push_back(from_blob(input_datas[i].data(), shape,
                                 exec_aten::ScalarType::Float));
   }

   /* Run inference. */
   std::vector<::executorch::runtime::EValue> evalues;
   for (auto &t : inputs)
      evalues.push_back(*t);
   auto result = module.forward(evalues);
   if (!result.ok()) {
      fprintf(stderr, "forward() failed\n");
      return 1;
   }

   /* Write outputs. */
   for (size_t i = 0; i < result->size(); i++) {
      const auto &out_t = result->at(i).toTensor();
      const void *out_ptr = out_t.const_data_ptr();
      auto out_path = work_dir / ("output_" + std::to_string(i) + ".bin");
      std::ofstream out(out_path, std::ios::binary);
      if (!out.is_open()) {
         fprintf(stderr, "Cannot open %s\n", out_path.c_str());
         return 1;
      }
      out.write(static_cast<const char *>(out_ptr),
                out_t.numel() * out_t.element_size());
   }

   return 0;
}

/* ------------------------------------------------------------------ */
/* Tensor I/O helpers                                                  */
/* ------------------------------------------------------------------ */

std::vector<float>
load_tensor_float(const std::filesystem::path &path)
{
   std::ifstream f(path, std::ios::binary | std::ios::ate);
   if (!f.is_open()) {
      fprintf(stderr, "Cannot open %s\n", path.c_str());
      return {};
   }
   size_t bytes = (size_t)f.tellg();
   f.seekg(0);
   std::vector<float> data(bytes / sizeof(float));
   f.read(reinterpret_cast<char *>(data.data()), (std::streamsize)bytes);
   return data;
}

/* ------------------------------------------------------------------ */
/* Utility                                                             */
/* ------------------------------------------------------------------ */

std::filesystem::path
get_work_dir()
{
   const char *cache_dir = getenv("TORX_TEST_DATA");
   if (cache_dir)
      return std::filesystem::path(cache_dir);

   return std::filesystem::current_path() / "src/gallium/targets/torx/tests";
}
