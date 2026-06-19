#ifndef MINIFLEX_GPUCPU_TRANSFER_CTX_H
#define MINIFLEX_GPUCPU_TRANSFER_CTX_H
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <torch/torch.h>

namespace miniflex {
class GPUCPUTransferCTX{
 private:
  char* cpu_tensor_ptr_;
  std::vector<char*> gpu_tensor_ptrs_;
  int64_t num_layers_;
  int64_t kv_dim_;
  int64_t cpu_num_blocks_;
  int64_t gpu_num_blocks_;
  int64_t slice_bytes_;
  cudaStream_t stream_;

 public:
  GPUCPUTransferCTX(char* cpu_tensor_ptr,
                    const std::vector<char*>& gpu_tensor_ptrs,
                    int64_t num_layers,
                    int64_t kv_dim,
                    int64_t cpu_num_blocks,
                    int64_t gpu_num_blocks,
                    int64_t slice_bytes);
  ~GPUCPUTransferCTX();
  auto transfer_blocks(const torch::Tensor& src_block_ids,
                       const torch::Tensor& dst_block_ids,
                       bool is_h2d) -> bool;
 private:
  auto transfer_blocks_h2d(const torch::Tensor& src_block_ids,
                           const torch::Tensor& dst_block_ids) -> bool;
  auto transfer_blocks_d2h(const torch::Tensor& src_block_ids,
                           const torch::Tensor& dst_block_ids) -> bool;
};
}
#endif