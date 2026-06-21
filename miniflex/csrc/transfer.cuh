#ifndef MINIFLEX_GPUCPU_TRANSFER_CTX_H
#define MINIFLEX_GPUCPU_TRANSFER_CTX_H
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <torch/torch.h>

namespace miniflex {
class GPUCPUTransferCTX{
 private:
  std::vector<char*> cpu_tensor_ptrs_;
  std::vector<char*> gpu_tensor_ptrs_;
  int64_t num_layers_;
  int64_t kv_dim_;
  int64_t slice_bytes_;     // 一块一层一份(K 或 V)的字节数 = 2D 拷贝的 width
  int64_t cpu_block_step_;  // CPU 层内相邻 block 的字节跨距
  int64_t cpu_kv_pitch_;    // CPU 层内 K->V 的字节跨距
  int64_t gpu_block_step_;  // GPU 层内相邻 block 的字节跨距
  int64_t gpu_kv_pitch_;    // GPU 层内 K->V 的字节跨距
  cudaStream_t stream_;

 public:
  GPUCPUTransferCTX(const std::vector<char*>& cpu_tensor_ptrs,
                    const std::vector<char*>& gpu_tensor_ptrs,
                    int64_t num_layers,
                    int64_t kv_dim,
                    int64_t slice_bytes,
                    int64_t cpu_block_step,
                    int64_t cpu_kv_pitch,
                    int64_t gpu_block_step,
                    int64_t gpu_kv_pitch);
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
