#include "transfer.cuh"

namespace miniflex {
GPUCPUTransferCTX::GPUCPUTransferCTX(char* cpu_tensor_ptr,
                                     const std::vector<char*>& gpu_tensor_ptrs,
                                     int64_t num_layers,
                                     int64_t kv_dim,
                                     int64_t cpu_num_blocks,
                                     int64_t gpu_num_blocks,
                                     int64_t slice_bytes)
    : cpu_tensor_ptr_(cpu_tensor_ptr),
      gpu_tensor_ptrs_(gpu_tensor_ptrs),
      num_layers_(num_layers),
      kv_dim_(kv_dim),
      cpu_num_blocks_(cpu_num_blocks),
      gpu_num_blocks_(gpu_num_blocks),
      slice_bytes_(slice_bytes) {
  cudaStreamCreate(&stream_);
}

GPUCPUTransferCTX::~GPUCPUTransferCTX() {
  cudaStreamDestroy(stream_);
}

auto GPUCPUTransferCTX::transfer_blocks(const torch::Tensor& src_block_ids,
                                        const torch::Tensor& dst_block_ids,
                                        bool is_h2d) -> bool {
  if (is_h2d) {
    return transfer_blocks_h2d(src_block_ids, dst_block_ids);
  } else {
    return transfer_blocks_d2h(src_block_ids, dst_block_ids);
  }
}

auto GPUCPUTransferCTX::transfer_blocks_h2d(const torch::Tensor& src_block_ids,
                                            const torch::Tensor& dst_block_ids)
    -> bool {
  const int64_t num_transfer_blocks = src_block_ids.numel();
  if (num_transfer_blocks == 0) {
    return true;
  }
  if (dst_block_ids.numel() != num_transfer_blocks) {
    return false;
  }
  auto cpu_ids = src_block_ids.to(torch::kCPU).to(torch::kLong).contiguous();
  auto gpu_ids = dst_block_ids.to(torch::kCPU).to(torch::kLong).contiguous();
  const int64_t* cids = cpu_ids.data_ptr<int64_t>();
  const int64_t* gids = gpu_ids.data_ptr<int64_t>();

  const int64_t cpu_pitch = cpu_num_blocks_ * slice_bytes_;
  const int64_t gpu_pitch = gpu_num_blocks_ * slice_bytes_;

  for (int64_t layer = 0; layer < num_layers_; ++layer) {
    char* cpu_layer =
        cpu_tensor_ptr_ + layer * kv_dim_ * cpu_num_blocks_ * slice_bytes_;
    char* gpu_layer = gpu_tensor_ptrs_[layer];
    for (int64_t i = 0; i < num_transfer_blocks; ++i) {
      char* cpu_blk = cpu_layer + cids[i] * slice_bytes_;
      char* gpu_blk = gpu_layer + gids[i] * slice_bytes_;
      cudaError_t err = cudaMemcpy2DAsync(
          gpu_blk, gpu_pitch,
          cpu_blk, cpu_pitch,
          slice_bytes_, kv_dim_,
          cudaMemcpyHostToDevice, stream_);
      if (err != cudaSuccess) {
        return false;
      }
    }
  }
  return cudaStreamSynchronize(stream_) == cudaSuccess;
}

auto GPUCPUTransferCTX::transfer_blocks_d2h(const torch::Tensor& src_block_ids,
                                            const torch::Tensor& dst_block_ids)
    -> bool {
  const int64_t num_transfer_blocks = src_block_ids.numel();
  if (num_transfer_blocks == 0) {
    return true;
  }
  if (dst_block_ids.numel() != num_transfer_blocks) {
    return false;
  }
  auto gpu_ids = src_block_ids.to(torch::kCPU).to(torch::kLong).contiguous();
  auto cpu_ids = dst_block_ids.to(torch::kCPU).to(torch::kLong).contiguous();
  const int64_t* gids = gpu_ids.data_ptr<int64_t>();
  const int64_t* cids = cpu_ids.data_ptr<int64_t>();

  const int64_t cpu_pitch = cpu_num_blocks_ * slice_bytes_;
  const int64_t gpu_pitch = gpu_num_blocks_ * slice_bytes_;

  for (int64_t layer = 0; layer < num_layers_; ++layer) {
    char* cpu_layer =
        cpu_tensor_ptr_ + layer * kv_dim_ * cpu_num_blocks_ * slice_bytes_;
    char* gpu_layer = gpu_tensor_ptrs_[layer];
    for (int64_t i = 0; i < num_transfer_blocks; ++i) {
      char* gpu_blk = gpu_layer + gids[i] * slice_bytes_;
      char* cpu_blk = cpu_layer + cids[i] * slice_bytes_;
      cudaError_t err = cudaMemcpy2DAsync(
          cpu_blk, cpu_pitch,
          gpu_blk, gpu_pitch,
          slice_bytes_, kv_dim_,
          cudaMemcpyDeviceToHost, stream_);
      if (err != cudaSuccess) {
        return false;
      }
    }
  }
  return cudaStreamSynchronize(stream_) == cudaSuccess;
}

}
