#ifndef MINIFLEX_SSD_IO_URING_H
#define MINIFLEX_SSD_IO_URING_H

#include <liburing.h>
#include <torch/torch.h>
#include <cstdint>
#include <string>
#include <vector>
#include <sys/uio.h>

namespace miniflex {
class SSDIOCTX {
 private:
  std::vector<int> fds_;
  int queue_depth_;
  int64_t blocks_per_file_;
  torch::Tensor cpu_tensor_;
  int64_t layer_num_;
  int64_t kv_dim_;
  int64_t cpu_num_blocks_;
  int64_t slice_bytes_;
  std::vector<std::string> file_paths_;
  struct io_uring ring_;
  bool use_io_uring_;
  bool use_direct_io_;
  std::vector<struct iovec> iovecs_;
 public:
  SSDIOCTX(int queue_depth,
           int64_t blocks_per_file,
           torch::Tensor cpu_tensor,
           int64_t layer_num,
           int64_t kv_dim,
           int64_t cpu_num_blocks,
           int64_t slice_bytes,
           std::vector<std::string> file_paths,
           bool use_direct_io = true);
  ~SSDIOCTX();
  auto transfer_blocks(const torch::Tensor& src_block_ids,
                       const torch::Tensor& dst_block_ids,
                       bool is_read) -> bool;
  auto is_using_io_uring() const -> bool { return use_io_uring_; }

 private:
  void init();
  auto transfer_blocks_write(const torch::Tensor& cpu_block_ids,
                             const torch::Tensor& ssd_block_ids) -> bool;
  auto transfer_blocks_read(const torch::Tensor& cpu_block_ids,
                            const torch::Tensor& ssd_block_ids) -> bool;
  auto write_with_io_uring(const torch::Tensor& cpu_block_ids,
                           const torch::Tensor& ssd_block_ids) -> bool;
  auto read_with_io_uring(const torch::Tensor& cpu_block_ids,
                          const torch::Tensor& ssd_block_ids) -> bool;
  auto write_with_file_io(const torch::Tensor& cpu_block_ids,
                          const torch::Tensor& ssd_block_ids) -> bool;
  auto read_with_file_io(const torch::Tensor& cpu_block_ids,
                         const torch::Tensor& ssd_block_ids) -> bool;
};
}  // namespace miniflex

#endif
