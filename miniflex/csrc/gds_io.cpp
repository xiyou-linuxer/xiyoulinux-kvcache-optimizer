#include "gds_io.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <unistd.h>
#include <utility>

namespace miniflex {
#ifdef MINIFLEX_WITH_CUFILE
namespace {

auto checked_mul(int64_t lhs, int64_t rhs, int64_t* result) -> bool {
  if (lhs < 0 || rhs < 0 ||
      (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

auto checked_add(int64_t lhs, int64_t rhs, int64_t* result) -> bool {
  if (lhs < 0 || rhs < 0 ||
      lhs > std::numeric_limits<int64_t>::max() - rhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

auto normalized_cpu_ids(const torch::Tensor& block_ids) -> torch::Tensor {
  TORCH_CHECK(block_ids.dim() == 1, "block IDs must be 1D tensors");
  return block_ids.to(torch::kCPU).to(torch::kLong).contiguous();
}

}  // namespace
#endif

GDSIOCTX::GDSIOCTX(int64_t blocks_per_file,
                   std::vector<torch::Tensor> gpu_tensors,
                   int64_t layer_num,
                   int64_t kv_dim,
                   int64_t gpu_num_blocks,
                   int64_t slice_bytes,
                   int64_t gpu_block_step,
                   int64_t gpu_kv_pitch,
                   std::vector<std::string> file_paths)
    : blocks_per_file_(blocks_per_file),
      gpu_tensors_(std::move(gpu_tensors)),
      layer_num_(layer_num),
      kv_dim_(kv_dim),
      gpu_num_blocks_(gpu_num_blocks),
      slice_bytes_(slice_bytes),
      gpu_block_step_(gpu_block_step),
      gpu_kv_pitch_(gpu_kv_pitch),
      file_paths_(std::move(file_paths)) {
  init();
}

GDSIOCTX::~GDSIOCTX() {
  cleanup();
}

void GDSIOCTX::init() {
  available_ = false;

#ifndef MINIFLEX_WITH_CUFILE
  fprintf(stderr,
          "[GDSIOCTX] built without cuFile support; GDS backend is disabled.\n");
  return;
#else
  if (blocks_per_file_ <= 0 || layer_num_ <= 0 || kv_dim_ <= 0 ||
      gpu_num_blocks_ <= 0 || slice_bytes_ <= 0 || gpu_block_step_ <= 0 ||
      gpu_kv_pitch_ <= 0 || file_paths_.empty() ||
      static_cast<int64_t>(gpu_tensors_.size()) != layer_num_) {
    fprintf(stderr, "[GDSIOCTX] invalid GDS geometry or resource list.\n");
    return;
  }

  int64_t layer_bytes = 0;
  int64_t bytes_per_file = 0;
  if (!checked_mul(slice_bytes_, layer_num_, &layer_bytes) ||
      !checked_mul(layer_bytes, kv_dim_, &block_bytes_) ||
      !checked_mul(blocks_per_file_, block_bytes_, &bytes_per_file) ||
      bytes_per_file > std::numeric_limits<off_t>::max()) {
    fprintf(stderr, "[GDSIOCTX] GDS geometry overflows int64 byte offsets.\n");
    return;
  }

  int64_t last_block_offset = 0;
  int64_t last_kv_offset = 0;
  int64_t required_bytes = 0;
  if (!checked_mul(gpu_num_blocks_ - 1, gpu_block_step_,
                   &last_block_offset) ||
      !checked_mul(kv_dim_ - 1, gpu_kv_pitch_, &last_kv_offset) ||
      !checked_add(last_block_offset, last_kv_offset, &required_bytes) ||
      !checked_add(required_bytes, slice_bytes_, &required_bytes) ||
      required_bytes > std::numeric_limits<off_t>::max()) {
    fprintf(stderr, "[GDSIOCTX] GPU buffer range overflows int64.\n");
    return;
  }

  gpu_buffers_.reserve(gpu_tensors_.size());
  for (size_t layer = 0; layer < gpu_tensors_.size(); ++layer) {
    auto& tensor = gpu_tensors_[layer];
    if (!tensor.is_cuda() || !tensor.is_contiguous() || tensor.numel() <= 0) {
      fprintf(stderr,
              "[GDSIOCTX] GPU tensor %zu must be non-empty, CUDA, and contiguous.\n",
              layer);
      cleanup();
      return;
    }
    auto* base = static_cast<char*>(tensor.data_ptr());
    const size_t bytes = tensor.nbytes();
    if (base == nullptr || bytes < static_cast<size_t>(required_bytes)) {
      fprintf(stderr,
              "[GDSIOCTX] GPU tensor %zu is too small for its declared layout "
              "(%zu < %ld bytes).\n",
              layer, bytes, static_cast<long>(required_bytes));
      cleanup();
      return;
    }
    gpu_buffers_.push_back({base, bytes, false});
  }

  CUfileError_t status = cuFileDriverOpen();
  if (status.err != CU_FILE_SUCCESS) {
    fprintf(stderr, "[GDSIOCTX] cuFileDriverOpen failed: %d\n", status.err);
    cleanup();
    return;
  }
  driver_opened_ = true;

#ifndef O_DIRECT
  fprintf(stderr, "[GDSIOCTX] O_DIRECT is unavailable on this platform.\n");
  cleanup();
  return;
#else
  int open_flags = O_RDWR | O_DIRECT;
#ifdef O_CLOEXEC
  open_flags |= O_CLOEXEC;
#endif

  files_.reserve(file_paths_.size());
  for (const auto& path : file_paths_) {
    const int fd = open(path.c_str(), open_flags);
    if (fd < 0) {
      fprintf(stderr, "[GDSIOCTX] O_DIRECT open failed for %s: %s\n",
              path.c_str(), strerror(errno));
      cleanup();
      return;
    }

    files_.push_back({fd, nullptr, false});
    GDSFile& file = files_.back();
    CUfileDescr_t descriptor{};
    descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descriptor.handle.fd = file.fd;
    status = cuFileHandleRegister(&file.handle, &descriptor);
    if (status.err != CU_FILE_SUCCESS) {
      fprintf(stderr, "[GDSIOCTX] cuFileHandleRegister failed for %s: %d\n",
              path.c_str(), status.err);
      cleanup();
      return;
    }
    file.handle_registered = true;
  }
#endif

  for (auto& buffer : gpu_buffers_) {
    status = cuFileBufRegister(buffer.base, buffer.bytes, 0);
    if (status.err != CU_FILE_SUCCESS) {
      fprintf(stderr, "[GDSIOCTX] cuFileBufRegister failed: %d\n", status.err);
      cleanup();
      return;
    }
    buffer.registered = true;
  }

  available_ = true;
#endif
}

void GDSIOCTX::cleanup() noexcept {
  available_ = false;

#ifdef MINIFLEX_WITH_CUFILE
  for (auto it = gpu_buffers_.rbegin(); it != gpu_buffers_.rend(); ++it) {
    if (it->registered) {
      const CUfileError_t status = cuFileBufDeregister(it->base);
      if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "[GDSIOCTX] cuFileBufDeregister failed: %d\n", status.err);
      }
      it->registered = false;
    }
  }

  for (auto it = files_.rbegin(); it != files_.rend(); ++it) {
    if (it->handle_registered) {
      cuFileHandleDeregister(it->handle);
      it->handle = nullptr;
      it->handle_registered = false;
    }
  }
#endif

  for (auto it = files_.rbegin(); it != files_.rend(); ++it) {
    if (it->fd >= 0) {
      close(it->fd);
      it->fd = -1;
    }
  }

#ifdef MINIFLEX_WITH_CUFILE
  if (driver_opened_) {
    const CUfileError_t status = cuFileDriverClose();
    if (status.err != CU_FILE_SUCCESS) {
      fprintf(stderr, "[GDSIOCTX] cuFileDriverClose failed: %d\n", status.err);
    }
    driver_opened_ = false;
  }
#endif

  gpu_buffers_.clear();
  files_.clear();
  block_bytes_ = 0;
}

auto GDSIOCTX::transfer_blocks(const torch::Tensor& src_block_ids,
                               const torch::Tensor& dst_block_ids,
                               bool is_read) -> bool {
  if (!available_) {
    return false;
  }
  // Match SSDIOCTX ordering: a read receives SSD IDs as source and GPU IDs as
  // destination, while a write receives GPU IDs as source and SSD IDs as
  // destination.
  if (is_read) {
    return transfer_blocks_read(dst_block_ids, src_block_ids);
  }
  return transfer_blocks_write(src_block_ids, dst_block_ids);
}

auto GDSIOCTX::transfer_blocks_read(const torch::Tensor& gpu_block_ids,
                                    const torch::Tensor& ssd_block_ids)
    -> bool {
#ifndef MINIFLEX_WITH_CUFILE
  (void)gpu_block_ids;
  (void)ssd_block_ids;
  return false;
#else
  TORCH_CHECK(gpu_block_ids.numel() == ssd_block_ids.numel(),
              "gpu/ssd block ID count mismatch");
  const auto gpu_ids_tensor = normalized_cpu_ids(gpu_block_ids);
  const auto ssd_ids_tensor = normalized_cpu_ids(ssd_block_ids);
  const int64_t num_blocks = gpu_ids_tensor.numel();
  if (num_blocks == 0) {
    return true;
  }

  const int64_t* gpu_ids = gpu_ids_tensor.data_ptr<int64_t>();
  const int64_t* ssd_ids = ssd_ids_tensor.data_ptr<int64_t>();
  for (int64_t i = 0; i < num_blocks; ++i) {
    const int64_t gpu_block_id = gpu_ids[i];
    const int64_t ssd_block_id = ssd_ids[i];
    if (gpu_block_id < 0 || gpu_block_id >= gpu_num_blocks_ ||
        ssd_block_id < 0) {
      fprintf(stderr,
              "[GDSIOCTX] read block ID out of range: gpu=%ld ssd=%ld\n",
              static_cast<long>(gpu_block_id), static_cast<long>(ssd_block_id));
      return false;
    }

    const int64_t file_index = ssd_block_id / blocks_per_file_;
    const int64_t file_block_index = ssd_block_id % blocks_per_file_;
    if (file_index < 0 || file_index >= static_cast<int64_t>(files_.size())) {
      fprintf(stderr, "[GDSIOCTX] read file index out of range: %ld\n",
              static_cast<long>(file_index));
      return false;
    }

    const auto& file = files_[file_index];
    off_t file_offset = static_cast<off_t>(file_block_index * block_bytes_);
    for (int64_t layer = 0; layer < layer_num_; ++layer) {
      const auto& buffer = gpu_buffers_[layer];
      for (int64_t kv = 0; kv < kv_dim_; ++kv) {
        const off_t buffer_offset = static_cast<off_t>(
            gpu_block_id * gpu_block_step_ + kv * gpu_kv_pitch_);
        const ssize_t read_bytes = cuFileRead(
            file.handle, buffer.base, static_cast<size_t>(slice_bytes_),
            file_offset, buffer_offset);
        if (read_bytes != static_cast<ssize_t>(slice_bytes_)) {
          fprintf(stderr,
                  "[GDSIOCTX] cuFileRead failed: gpu=%ld ssd=%ld layer=%ld "
                  "kv=%ld ret=%zd expected=%ld\n",
                  static_cast<long>(gpu_block_id),
                  static_cast<long>(ssd_block_id), static_cast<long>(layer),
                  static_cast<long>(kv), read_bytes,
                  static_cast<long>(slice_bytes_));
          return false;
        }
        file_offset += static_cast<off_t>(slice_bytes_);
      }
    }
  }
  return true;
#endif
}

auto GDSIOCTX::transfer_blocks_write(const torch::Tensor& gpu_block_ids,
                                     const torch::Tensor& ssd_block_ids)
    -> bool {
#ifndef MINIFLEX_WITH_CUFILE
  (void)gpu_block_ids;
  (void)ssd_block_ids;
  return false;
#else
  TORCH_CHECK(gpu_block_ids.numel() == ssd_block_ids.numel(),
              "gpu/ssd block ID count mismatch");
  const auto gpu_ids_tensor = normalized_cpu_ids(gpu_block_ids);
  const auto ssd_ids_tensor = normalized_cpu_ids(ssd_block_ids);
  const int64_t num_blocks = gpu_ids_tensor.numel();
  if (num_blocks == 0) {
    return true;
  }

  const int64_t* gpu_ids = gpu_ids_tensor.data_ptr<int64_t>();
  const int64_t* ssd_ids = ssd_ids_tensor.data_ptr<int64_t>();
  for (int64_t i = 0; i < num_blocks; ++i) {
    const int64_t gpu_block_id = gpu_ids[i];
    const int64_t ssd_block_id = ssd_ids[i];
    if (gpu_block_id < 0 || gpu_block_id >= gpu_num_blocks_ ||
        ssd_block_id < 0) {
      fprintf(stderr,
              "[GDSIOCTX] write block ID out of range: gpu=%ld ssd=%ld\n",
              static_cast<long>(gpu_block_id), static_cast<long>(ssd_block_id));
      return false;
    }

    const int64_t file_index = ssd_block_id / blocks_per_file_;
    const int64_t file_block_index = ssd_block_id % blocks_per_file_;
    if (file_index < 0 || file_index >= static_cast<int64_t>(files_.size())) {
      fprintf(stderr, "[GDSIOCTX] write file index out of range: %ld\n",
              static_cast<long>(file_index));
      return false;
    }

    const auto& file = files_[file_index];
    off_t file_offset = static_cast<off_t>(file_block_index * block_bytes_);
    for (int64_t layer = 0; layer < layer_num_; ++layer) {
      const auto& buffer = gpu_buffers_[layer];
      for (int64_t kv = 0; kv < kv_dim_; ++kv) {
        const off_t buffer_offset = static_cast<off_t>(
            gpu_block_id * gpu_block_step_ + kv * gpu_kv_pitch_);
        const ssize_t written_bytes = cuFileWrite(
            file.handle, buffer.base, static_cast<size_t>(slice_bytes_),
            file_offset, buffer_offset);
        if (written_bytes != static_cast<ssize_t>(slice_bytes_)) {
          fprintf(stderr,
                  "[GDSIOCTX] cuFileWrite failed: gpu=%ld ssd=%ld layer=%ld "
                  "kv=%ld ret=%zd expected=%ld\n",
                  static_cast<long>(gpu_block_id),
                  static_cast<long>(ssd_block_id), static_cast<long>(layer),
                  static_cast<long>(kv), written_bytes,
                  static_cast<long>(slice_bytes_));
          return false;
        }
        file_offset += static_cast<off_t>(slice_bytes_);
      }
    }
  }
  return true;
#endif
}

}  // namespace miniflex
