// 本测试文件由 Claude (Anthropic) 编写。
// 测试内容：C++ SSD io_uring 后端：普通 file IO、O_DIRECT、io_uring 分块、io_uring+O_DIRECT 的写入/清零/读回 payload 校验。

#include <torch/torch.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include "ssd_io_uring.h"

namespace {

auto make_cpu_tensor(int64_t layer_num, int64_t kv_dim, int64_t cpu_num_blocks)
    -> torch::Tensor {
  auto tensor = torch::empty(
      {layer_num, kv_dim, cpu_num_blocks, 3, 2, 5},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  for (int64_t layer_id = 0; layer_id < layer_num; ++layer_id) {
    for (int64_t kv_id = 0; kv_id < kv_dim; ++kv_id) {
      for (int64_t block_id = 0; block_id < cpu_num_blocks; ++block_id) {
        const float value =
            static_cast<float>(layer_id * 1000 + kv_id * 100 + block_id);
        tensor.index_put_({layer_id, kv_id, block_id}, value);
      }
    }
  }
  return tensor.contiguous();
}

// 创建 4096 对齐的 CPU tensor，用于覆盖 O_DIRECT 路径。
auto make_aligned_cpu_tensor(int64_t layer_num, int64_t kv_dim, int64_t cpu_num_blocks)
    -> torch::Tensor {
  const int64_t tokens_per_block = 1024;
  const int64_t num_heads = 1;
  const int64_t head_size = 1;
  const int64_t numel =
      layer_num * kv_dim * cpu_num_blocks * tokens_per_block * num_heads * head_size;
  const int64_t bytes = numel * static_cast<int64_t>(sizeof(float));

  void* raw = nullptr;
  if (::posix_memalign(&raw, 4096, static_cast<size_t>(bytes)) != 0) {
    throw std::runtime_error("posix_memalign failed");
  }

  auto tensor = torch::from_blob(
      raw,
      {layer_num, kv_dim, cpu_num_blocks, tokens_per_block, num_heads, head_size},
      [](void* ptr) { std::free(ptr); },
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

  for (int64_t layer_id = 0; layer_id < layer_num; ++layer_id) {
    for (int64_t kv_id = 0; kv_id < kv_dim; ++kv_id) {
      for (int64_t block_id = 0; block_id < cpu_num_blocks; ++block_id) {
        const float value =
            static_cast<float>(layer_id * 1000 + kv_id * 100 + block_id);
        tensor.index_put_({layer_id, kv_id, block_id}, value);
      }
    }
  }
  return tensor;
}

void create_cache_file(const std::filesystem::path& path, int64_t bytes) {
  FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) {
    throw std::runtime_error("failed to create cache file");
  }
  if (::ftruncate(::fileno(file), bytes) != 0) {
    std::fclose(file);
    throw std::runtime_error("failed to size cache file");
  }
  std::fclose(file);
}

void expect_block_equal(const torch::Tensor& actual,
                        int64_t actual_block,
                        const torch::Tensor& expected,
                        int64_t expected_block,
                        const std::string& message) {
  if (!torch::equal(
          actual.index({torch::indexing::Slice(), torch::indexing::Slice(), actual_block}),
          expected.index({torch::indexing::Slice(), torch::indexing::Slice(), expected_block}))) {
    throw std::runtime_error(message);
  }
}

void expect_block_zero(const torch::Tensor& tensor,
                       int64_t block_id,
                       const std::string& message) {
  const auto block = tensor.index(
      {torch::indexing::Slice(), torch::indexing::Slice(), block_id});
  if (!torch::equal(block, torch::zeros_like(block))) {
    throw std::runtime_error(message);
  }
}

void run_roundtrip_case(torch::Tensor cpu_tensor,
                        bool use_direct_io,
                        const std::string& case_name) {
  const int64_t layer_num = 2;
  const int64_t kv_dim = 2;
  const int64_t cpu_num_blocks = 4;
  const int64_t blocks_per_file = 4;

  auto original = cpu_tensor.clone();
  const int64_t slice_bytes =
      cpu_tensor.index({0, 0, 0}).numel() * cpu_tensor.element_size();
  const int64_t block_bytes = slice_bytes * layer_num * kv_dim;

  const auto temp_dir = std::filesystem::temp_directory_path() /
                        ("miniflex_ssd_io_uring_test_" +
                         std::to_string(::getpid()));
  std::filesystem::create_directories(temp_dir);
  const auto file_path = temp_dir / "ssd_cache_0.bin";
  create_cache_file(file_path, blocks_per_file * block_bytes);

  miniflex::SSDIOCTX ctx(
      0,
      blocks_per_file,
      cpu_tensor,
      layer_num,
      kv_dim,
      cpu_num_blocks,
      slice_bytes,
      std::vector<std::string>{file_path.string()},
      use_direct_io);

  auto cpu_src_base = torch::tensor({99, 3, 99, 1, 99}, torch::kInt64);
  auto ssd_dst_base = torch::tensor({99, 0, 99, 2, 99}, torch::kInt64);
  auto cpu_src_blocks = cpu_src_base.index({torch::indexing::Slice(1, torch::indexing::None, 2)});
  auto ssd_dst_blocks = ssd_dst_base.index({torch::indexing::Slice(1, torch::indexing::None, 2)});
  if (!ctx.transfer_blocks(cpu_src_blocks, ssd_dst_blocks, false)) {
    throw std::runtime_error(case_name + ": SSD write transfer failed");
  }

  cpu_tensor.zero_();

  auto ssd_src_base = torch::tensor({99, 0, 99, 2, 99}, torch::kInt64);
  auto cpu_dst_base = torch::tensor({99, 0, 99, 2, 99}, torch::kInt64);
  auto ssd_src_blocks = ssd_src_base.index({torch::indexing::Slice(1, torch::indexing::None, 2)});
  auto cpu_dst_blocks = cpu_dst_base.index({torch::indexing::Slice(1, torch::indexing::None, 2)});
  if (!ctx.transfer_blocks(ssd_src_blocks, cpu_dst_blocks, true)) {
    throw std::runtime_error(case_name + ": SSD read transfer failed");
  }

  if (!torch::equal(cpu_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}),
                    original.index({torch::indexing::Slice(), torch::indexing::Slice(), 3}))) {
    throw std::runtime_error("CPU block 0 does not match original block 3");
  }
  if (!torch::equal(cpu_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(), 2}),
                    original.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}))) {
    throw std::runtime_error("CPU block 2 does not match original block 1");
  }
  if (!torch::equal(cpu_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}),
                    torch::zeros_like(cpu_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(), 1})))) {
    throw std::runtime_error("CPU block 1 should remain zero");
  }
  if (!torch::equal(cpu_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(), 3}),
                    torch::zeros_like(cpu_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(), 3})))) {
    throw std::runtime_error("CPU block 3 should remain zero");
  }

  std::filesystem::remove_all(temp_dir);
}

void run_io_uring_chunk_roundtrip_case(torch::Tensor cpu_tensor,
                                       bool use_direct_io,
                                       const std::string& case_name) {
  const int64_t layer_num = 2;
  const int64_t kv_dim = 2;
  const int64_t cpu_num_blocks = 6;
  const int64_t blocks_per_file = 3;
  const int queue_depth = 2;

  auto original = cpu_tensor.clone();
  const int64_t slice_bytes =
      cpu_tensor.index({0, 0, 0}).numel() * cpu_tensor.element_size();
  const int64_t block_bytes = slice_bytes * layer_num * kv_dim;

  const auto temp_dir = std::filesystem::temp_directory_path() /
                        ("miniflex_ssd_io_uring_chunk_test_" +
                         std::to_string(::getpid()));
  std::filesystem::create_directories(temp_dir);
  std::vector<std::string> file_paths;
  for (int file_idx = 0; file_idx < 2; ++file_idx) {
    const auto file_path = temp_dir /
                           ("ssd_cache_" + std::to_string(file_idx) + ".bin");
    create_cache_file(file_path, blocks_per_file * block_bytes);
    file_paths.push_back(file_path.string());
  }

  miniflex::SSDIOCTX ctx(
      queue_depth,
      blocks_per_file,
      cpu_tensor,
      layer_num,
      kv_dim,
      cpu_num_blocks,
      slice_bytes,
      file_paths,
      use_direct_io);
  if (!ctx.is_using_io_uring()) {
    std::filesystem::remove_all(temp_dir);
    throw std::runtime_error(case_name + ": io_uring is not available");
  }

  auto cpu_src_base = torch::tensor({99, 5, 99, 2, 99, 4, 99, 1, 99, 3, 99}, torch::kInt64);
  auto ssd_dst_base = torch::tensor({99, 4, 99, 0, 99, 5, 99, 1, 99, 3, 99}, torch::kInt64);
  auto cpu_src_blocks = cpu_src_base.index({torch::indexing::Slice(1, torch::indexing::None, 2)});
  auto ssd_dst_blocks = ssd_dst_base.index({torch::indexing::Slice(1, torch::indexing::None, 2)});
  if (!ctx.transfer_blocks(cpu_src_blocks, ssd_dst_blocks, false)) {
    std::filesystem::remove_all(temp_dir);
    throw std::runtime_error(case_name + ": io_uring SSD write transfer failed");
  }

  cpu_tensor.zero_();

  auto ssd_src_base = torch::tensor({99, 4, 99, 0, 99, 5, 99, 1, 99, 3, 99}, torch::kInt64);
  auto cpu_dst_base = torch::tensor({99, 0, 99, 1, 99, 2, 99, 3, 99, 4, 99}, torch::kInt64);
  auto ssd_src_blocks = ssd_src_base.index({torch::indexing::Slice(1, torch::indexing::None, 2)});
  auto cpu_dst_blocks = cpu_dst_base.index({torch::indexing::Slice(1, torch::indexing::None, 2)});
  if (!ctx.transfer_blocks(ssd_src_blocks, cpu_dst_blocks, true)) {
    std::filesystem::remove_all(temp_dir);
    throw std::runtime_error(case_name + ": io_uring SSD read transfer failed");
  }

  expect_block_equal(cpu_tensor, 0, original, 5, "CPU block 0 does not match original block 5");
  expect_block_equal(cpu_tensor, 1, original, 2, "CPU block 1 does not match original block 2");
  expect_block_equal(cpu_tensor, 2, original, 4, "CPU block 2 does not match original block 4");
  expect_block_equal(cpu_tensor, 3, original, 1, "CPU block 3 does not match original block 1");
  expect_block_equal(cpu_tensor, 4, original, 3, "CPU block 4 does not match original block 3");
  expect_block_zero(cpu_tensor, 5, "CPU block 5 should remain zero");

  std::filesystem::remove_all(temp_dir);
}

// 验证普通 file IO 路径能正确写入并读回真实数据。
void test_file_io_roundtrip_validates_transfer_data() {
  run_roundtrip_case(make_cpu_tensor(2, 2, 4), false, "file_io");
}

// 验证 O_DIRECT 路径能在 4096 对齐 tensor 和 4096 字节 slice 下正确传输。
void test_direct_io_roundtrip_validates_transfer_data() {
  run_roundtrip_case(make_aligned_cpu_tensor(2, 2, 4), true, "direct_io");
}

// 验证 io_uring 路径能跨 queue_depth 分 chunk，且跨多个 SSD 文件读写真数据。
void test_io_uring_chunk_roundtrip_validates_transfer_data() {
  run_io_uring_chunk_roundtrip_case(make_cpu_tensor(2, 2, 6), false, "io_uring_chunk");
}

// 验证 io_uring + O_DIRECT 在 4096 对齐 tensor 下也能跨 chunk 正确传输。
void test_io_uring_direct_chunk_roundtrip_validates_transfer_data() {
  run_io_uring_chunk_roundtrip_case(
      make_aligned_cpu_tensor(2, 2, 6), true, "io_uring_direct_chunk");
}

}  // namespace

int main() {
  std::cout << "[开始] SSDIOCTX file IO 写入后读回校验真实传输数据\n";
  try {
    test_file_io_roundtrip_validates_transfer_data();
  } catch (const std::exception& exc) {
    std::cerr << "[失败] SSDIOCTX file IO 写入后读回校验真实传输数据: "
              << exc.what() << "\n";
    return 1;
  }
  std::cout << "[通过] SSDIOCTX file IO 写入后读回校验真实传输数据\n";

  std::cout << "[开始] SSDIOCTX O_DIRECT 写入后读回校验真实传输数据\n";
  try {
    test_direct_io_roundtrip_validates_transfer_data();
  } catch (const std::exception& exc) {
    std::cerr << "[失败] SSDIOCTX O_DIRECT 写入后读回校验真实传输数据: "
              << exc.what() << "\n";
    return 1;
  }
  std::cout << "[通过] SSDIOCTX O_DIRECT 写入后读回校验真实传输数据\n";

  std::cout << "[开始] SSDIOCTX io_uring chunk 写入后读回校验真实传输数据\n";
  try {
    test_io_uring_chunk_roundtrip_validates_transfer_data();
  } catch (const std::exception& exc) {
    std::cerr << "[失败] SSDIOCTX io_uring chunk 写入后读回校验真实传输数据: "
              << exc.what() << "\n";
    return 1;
  }
  std::cout << "[通过] SSDIOCTX io_uring chunk 写入后读回校验真实传输数据\n";

  std::cout << "[开始] SSDIOCTX io_uring O_DIRECT chunk 写入后读回校验真实传输数据\n";
  try {
    test_io_uring_direct_chunk_roundtrip_validates_transfer_data();
  } catch (const std::exception& exc) {
    std::cerr << "[失败] SSDIOCTX io_uring O_DIRECT chunk 写入后读回校验真实传输数据: "
              << exc.what() << "\n";
    return 1;
  }
  std::cout << "[通过] SSDIOCTX io_uring O_DIRECT chunk 写入后读回校验真实传输数据\n";
  return 0;
}
