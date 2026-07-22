#ifndef MINIFLEX_GDS_IO_H
#define MINIFLEX_GDS_IO_H

// GPU Direct Storage (GDS) transfer context.
//
// GDS moves KV cache blocks directly between SSD files and GPU memory through
// cuFile. The public API intentionally matches SSDIOCTX: callers provide
// source/destination IDs and a direction flag, while the implementation keeps
// distinct read and write paths internally.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

#ifdef MINIFLEX_WITH_CUFILE
#include <cufile.h>
#endif

namespace miniflex {

class GDSIOCTX {
 public:
  GDSIOCTX(int64_t blocks_per_file,
           std::vector<torch::Tensor> gpu_tensors,
           int64_t layer_num,
           int64_t kv_dim,
           int64_t gpu_num_blocks,
           int64_t slice_bytes,
           int64_t gpu_block_step,
           int64_t gpu_kv_pitch,
           std::vector<std::string> file_paths);
  ~GDSIOCTX();

  GDSIOCTX(const GDSIOCTX&) = delete;
  GDSIOCTX& operator=(const GDSIOCTX&) = delete;

  // is_read=true  -> SSD -> GPU (DISK2D)
  // is_read=false -> GPU -> SSD (D2DISK)
  auto transfer_blocks(const torch::Tensor& src_block_ids,
                       const torch::Tensor& dst_block_ids,
                       bool is_read) -> bool;

  // True only after cuFile, all file handles, and all GPU buffer
  // registrations have succeeded. It does not imply the platform offers
  // peer-to-peer GDS DMA rather than cuFile compatibility mode.
  auto is_available() const -> bool { return available_; }

 private:
  struct GDSBuffer {
    char* base = nullptr;
    size_t bytes = 0;
    bool registered = false;
  };

  struct GDSFile {
    int fd = -1;
#ifdef MINIFLEX_WITH_CUFILE
    CUfileHandle_t handle = nullptr;
#endif
    bool handle_registered = false;
  };

  void init();
  void cleanup() noexcept;
  auto transfer_blocks_read(const torch::Tensor& gpu_block_ids,
                            const torch::Tensor& ssd_block_ids) -> bool;
  auto transfer_blocks_write(const torch::Tensor& gpu_block_ids,
                             const torch::Tensor& ssd_block_ids) -> bool;

  int64_t blocks_per_file_;
  // Holding Tensor objects, rather than only raw pointers, keeps the CUDA
  // allocations alive for the entire lifetime of their cuFile registrations.
  std::vector<torch::Tensor> gpu_tensors_;
  int64_t layer_num_;
  int64_t kv_dim_;
  int64_t gpu_num_blocks_;
  int64_t slice_bytes_;
  int64_t gpu_block_step_;
  int64_t gpu_kv_pitch_;
  std::vector<std::string> file_paths_;
  std::vector<GDSBuffer> gpu_buffers_;
  std::vector<GDSFile> files_;
  int64_t block_bytes_ = 0;
  bool driver_opened_ = false;
  bool available_ = false;
};

}  // namespace miniflex

#endif  // MINIFLEX_GDS_IO_H
