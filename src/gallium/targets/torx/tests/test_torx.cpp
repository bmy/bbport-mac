/*
 * Copyright 2025 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 *
 * SPDX-License-Identifier: MIT
 *
 * GTest-based end-to-end test suite for the Torx ExecuTorch backend.
 *
 * This binary runs directly on the NPU board.  It discovers pre-compiled
 * test artifacts (.pte files, CPU references, metadata) generated at
 * build time by compile_torx_tests.py, runs NPU inference locally, and
 * compares the result against the CPU reference.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "test_torx_helpers.h"

/* ------------------------------------------------------------------ */
/* Tensor dump helpers                                                 */
/* ------------------------------------------------------------------ */

static void
dump_tensor_float(const std::vector<float> &data,
                  const std::filesystem::path &path)
{
   std::ofstream out(path, std::ios::binary);
   if (!out.is_open()) {
      fprintf(stderr, "Cannot dump to %s\n", path.c_str());
      return;
   }
   out.write(reinterpret_cast<const char *>(data.data()),
             (std::streamsize)(data.size() * sizeof(float)));
}

/* ------------------------------------------------------------------ */
/* Output comparison                                                   */
/* ------------------------------------------------------------------ */

static void
compare_tensors(const std::vector<float> &actual,
                const std::vector<float> &expected,
                float tolerance,
                const std::string &test_name)
{
   ASSERT_EQ(actual.size(), expected.size())
      << test_name << ": output size mismatch";

   int mismatches = 0;
   float max_diff = 0;

   for (size_t i = 0; i < actual.size(); i++) {
      float diff = std::abs(actual[i] - expected[i]);
      if (diff > tolerance) {
         if (mismatches < 24) {
            ADD_FAILURE()
               << test_name << "[" << i << "]: "
               << "expected " << expected[i]
               << " got " << actual[i]
               << " (diff=" << diff << ")";
         }
         mismatches++;
      }
      max_diff = std::max(max_diff, diff);
   }

   EXPECT_EQ(mismatches, 0)
      << test_name << ": " << mismatches << "/"
      << actual.size() << " elements differ"
      << " (max_diff=" << max_diff
      << ", tolerance=" << tolerance << ")";
}

/* ------------------------------------------------------------------ */
/* Test fixture — shared by per-op and whole-model tests               */
/* ------------------------------------------------------------------ */

class TorxTest : public testing::Test {
 public:
   std::string test_name_;   /* e.g. "mobilenet_v2/000" or "mobilenet_v2" */
   std::string device_id_;   /* e.g. "ethosu-65-256-98304" */

   TorxTest(const std::string &test_name, const std::string &device_id)
      : test_name_(test_name), device_id_(device_id) {}

   void TestBody() override
   {
      auto work_dir = get_work_dir() / test_name_;
      auto pte_path = work_dir / (device_id_ + ".pte");
      auto meta_path = work_dir / (device_id_ + ".json");

      ASSERT_TRUE(std::filesystem::exists(pte_path))
         << "No .pte for device " << device_id_
         << " in " << work_dir.string();

      /* 1. Load build-time metadata */
      auto meta = load_meta(meta_path);
      ASSERT_TRUE(meta.ok)
         << "Failed to load metadata for " << test_name_
         << ": " << meta.error;

      /* 2. Run inference on the local NPU */
      int rc = run_on_npu(work_dir, pte_path, device_id_);
      ASSERT_EQ(rc, 0)
         << "NPU inference failed for " << test_name_;

      /* 3. Compare each output against float CPU reference. */
      for (size_t i = 0; ; i++) {
         auto ref_path = work_dir / ("output-" + std::to_string(i) + ".data");
         if (!std::filesystem::exists(ref_path))
            break;
         ASSERT_LT(i, meta.output_scales.size())
            << "Metadata has only " << meta.output_scales.size()
            << " output(s) but output-" << i << ".data exists"
            << " — rebuild with fixed run_on_cpu.py / compile_torx_tests.py";
         auto npu_float = load_tensor_float(
            work_dir / ("output_" + std::to_string(i) + ".bin"));
         auto expected = load_tensor_float(ref_path);

          /* Dump for offline analysis */
          auto dump_name = test_name_ + "-output-" + std::to_string(i);
          std::replace(dump_name.begin(), dump_name.end(), '/', '.');
          dump_tensor_float(expected, dump_name + "-expected.bin");
          dump_tensor_float(npu_float,  dump_name + "-actual.bin");

         float tolerance = meta.max_quant_errors[i] + meta.output_scales[i];
         compare_tensors(npu_float, expected, tolerance, test_name_);
      }
   }
};

/* ------------------------------------------------------------------ */
/* Dynamic test registration                                           */
/* ------------------------------------------------------------------ */

static void
register_model_tests()
{
   std::string device_id = detect_device_id();
   if (device_id.empty()) {
      fprintf(stderr, "No NPU hardware detected, skipping all tests\n");
      return;
   }

   /* Per-op tests: <model>/<op>/<device>.pte */
   auto tests = get_model_files(device_id);
   for (const auto &test : tests) {
      /* "mobilenet_v2/000" → suite "mobilenet_v2", name "000" */
      auto slash = test.find('/');
      std::string suite = test.substr(0, slash);
      std::string name  = test.substr(slash + 1);

      testing::RegisterTest(
         suite.c_str(), name.c_str(),
         nullptr, nullptr,
         __FILE__, __LINE__,
         [test, device_id]() -> TorxTest * {
            return new TorxTest(test, device_id);
         });
   }

   /* Whole-model tests: <model>/<device>.pte (no op subdirectory) */
   auto models = get_whole_model_files(device_id);
   for (const auto &model : models) {
      testing::RegisterTest(
         "model", model.c_str(),
         nullptr, nullptr,
         __FILE__, __LINE__,
         [model, device_id]() -> TorxTest * {
            return new TorxTest(model, device_id);
         });
   }
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int
main(int argc, char **argv)
{
   testing::InitGoogleTest(&argc, argv);

   /* Discover and register tests from pre-built artifacts */
   register_model_tests();

   return RUN_ALL_TESTS();
}
