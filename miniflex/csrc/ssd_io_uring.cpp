#include "ssd_io_uring.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>
#include <sys/uio.h>
#include <cstdint>
#include <cstdio>

namespace miniflex{
  
  static ssize_t pwritev_retry(int fd, const struct iovec* iov, int iovcnt, off_t offset) {
    ssize_t n;
    do {
      n = pwritev(fd, iov, iovcnt, offset);
    } while (n < 0 && errno == EINTR);
    return n;
  }

  static ssize_t preadv_retry(int fd, const struct iovec* iov, int iovcnt, off_t offset) {
    ssize_t n;
    do {
      n = preadv(fd, iov, iovcnt, offset);
    } while (n < 0 && errno == EINTR);
    return n;
  }

  SSDIOCTX::SSDIOCTX(int queue_depth,
                     int64_t blocks_per_file,
                     torch::Tensor cpu_tensor,
                     int64_t layer_num,
                     int64_t kv_dim,
                     int64_t cpu_num_blocks,
                     int64_t slice_bytes,
                     std::vector<std::string> file_paths,
                     bool use_direct_io)
    :queue_depth_(queue_depth),
     blocks_per_file_(blocks_per_file),
     cpu_tensor_(cpu_tensor),
     layer_num_(layer_num),
     kv_dim_(kv_dim),
     cpu_num_blocks_(cpu_num_blocks),
     slice_bytes_(slice_bytes),
     file_paths_(file_paths),
     use_direct_io_(use_direct_io){
    init();
  }

  void SSDIOCTX::init(){
    int use_O_DIRECT = (slice_bytes_ % 4096 == 0 && use_direct_io_) ? O_DIRECT : 0;
    if(use_O_DIRECT){
      TORCH_CHECK(
        reinterpret_cast<uintptr_t>(cpu_tensor_.data_ptr()) % 4096 == 0,
        "CPU tensor must be aligned to 4096 bytes for O_DIRECT"
      );
    }
    for (const auto& file_path : file_paths_){
      int fd = open(file_path.c_str(), O_RDWR | O_CLOEXEC | use_O_DIRECT);
      if(fd == -1){
        std::string error_msg = "failed to open SSD cache file " + file_path + ": " + std::strerror(errno);
        for (int opened_fd : fds_) {
          if (opened_fd >= 0) {
            close(opened_fd);
          }
        }
        fds_.clear();
        throw std::runtime_error(error_msg);
      }
      fds_.push_back(fd);
    }
    int ret = io_uring_queue_init(queue_depth_, &ring_, 0);
    if (ret == 0){
      use_io_uring_ = true;
    }else{
      use_io_uring_ = false;
      iovecs_.reserve(layer_num_ * kv_dim_);
    }
    
  }

  SSDIOCTX::~SSDIOCTX(){
    if (use_io_uring_) {
      io_uring_queue_exit(&ring_);
      use_io_uring_ = false;
    }
    for (int fd : fds_) {
      if (fd >= 0) {
        close(fd);
      }
    }
    fds_.clear();
  }

  auto SSDIOCTX::transfer_blocks(const torch::Tensor& src_block_ids,
                                 const torch::Tensor& dst_block_ids,
                                 bool is_read) -> bool{
    if (is_read) {
      return transfer_blocks_read(dst_block_ids, src_block_ids);
    }else{
      return transfer_blocks_write(src_block_ids, dst_block_ids);
    }
  }


  auto SSDIOCTX::transfer_blocks_read(const torch::Tensor& cpu_block_ids,
                                      const torch::Tensor& ssd_block_ids) -> bool{
    if (use_io_uring_) {
      return read_with_io_uring(cpu_block_ids, ssd_block_ids);
    }else{
      return read_with_file_io(cpu_block_ids, ssd_block_ids);
    }
  }

  auto SSDIOCTX::transfer_blocks_write(const torch::Tensor& cpu_block_ids,
                                      const torch::Tensor& ssd_block_ids) -> bool{
    if (use_io_uring_) {
      return write_with_io_uring(cpu_block_ids, ssd_block_ids);
    }else{
      return write_with_file_io(cpu_block_ids, ssd_block_ids);
    }
  }

  auto SSDIOCTX::write_with_io_uring(const torch::Tensor& cpu_block_ids,
                                     const torch::Tensor& ssd_block_ids) -> bool{
    const int64_t num_transfer_blocks = cpu_block_ids.numel();
    if (num_transfer_blocks == 0) {
      return true;
    }
    if (queue_depth_ <= 0) {
      fprintf(stderr, "[SSDIOCTX] invalid io_uring queue_depth=%d\n", queue_depth_);
      return false;
    }

    const int64_t* cpu_ids = cpu_block_ids.data_ptr<int64_t>();
    const int64_t* ssd_ids = ssd_block_ids.data_ptr<int64_t>();
    const int64_t cpu_stride = cpu_block_ids.stride(0);
    const int64_t ssd_stride = ssd_block_ids.stride(0);
    const int64_t block_bytes = slice_bytes_ * layer_num_ * kv_dim_;
    char* base = reinterpret_cast<char*>(cpu_tensor_.data_ptr());
    const int64_t iovec_size = layer_num_ * kv_dim_;
    const int64_t chunk_capacity = std::min<int64_t>(queue_depth_, num_transfer_blocks);
    std::vector<struct iovec> iovecs(iovec_size * chunk_capacity);

    for (int64_t chunk_start = 0;
         chunk_start < num_transfer_blocks;
         chunk_start += chunk_capacity) {
      const int64_t chunk_size =
        std::min<int64_t>(chunk_capacity, num_transfer_blocks - chunk_start);

      for (int64_t j = 0; j < chunk_size; ++j) {
        const int64_t i = chunk_start + j;
        const int64_t cpu_block_id = cpu_ids[i * cpu_stride];
        const int64_t ssd_block_id = ssd_ids[i * ssd_stride];
        struct iovec* iov = &iovecs[j * iovec_size];
        int iov_idx = 0;
        for(int64_t layer_id = 0; layer_id < layer_num_; ++layer_id){
          for(int64_t kv_id = 0; kv_id < kv_dim_; ++kv_id){
            iov[iov_idx].iov_base = base + (
              (layer_id * kv_dim_ * cpu_num_blocks_ + kv_id * cpu_num_blocks_ + cpu_block_id)
              * slice_bytes_
            );
            iov[iov_idx].iov_len = static_cast<size_t>(slice_bytes_);
            ++iov_idx;
          }
        }

        const int64_t file_idx = ssd_block_id / blocks_per_file_;
        const int64_t block_idx = ssd_block_id % blocks_per_file_;
        const off_t offset = static_cast<off_t>(block_idx * block_bytes);
        const int fd = fds_.at(file_idx);
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if(!sqe){
          fprintf(stderr,"[SSDIOCTX] Fatal: io_uring SQ ring is full\n");
          return false;
        }
        io_uring_prep_writev(sqe, fd, iov, static_cast<unsigned>(iovec_size), offset);
        io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(static_cast<uintptr_t>(ssd_block_id)));
      }

      int submitted = 0;
      while (submitted < chunk_size) {
        const int ret = io_uring_submit(&ring_);
        if(ret < 0){
          fprintf(stderr, "[SSDIOCTX] io_uring_submit failed: %s\n", std::strerror(-ret));
          return false;
        }
        if(ret == 0){
          fprintf(stderr,
            "[SSDIOCTX] io_uring_submit submitted 0 requests: submitted=%d chunk_size=%ld\n",
            submitted, (long)chunk_size);
          return false;
        }
        submitted += ret;
      }

      bool chunk_success = true;
      for(int64_t j = 0; j < chunk_size; ++j){
        struct io_uring_cqe* cqe;
        int ret = io_uring_wait_cqe(&ring_, &cqe);
        if(ret < 0){
          fprintf(stderr, "[SSDIOCTX] io_uring_wait_cqe failed: %s\n",std::strerror(-ret));
          return false;
        }
        const uint64_t finished_ssd_id = reinterpret_cast<uintptr_t>(io_uring_cqe_get_data(cqe));
        if(cqe->res < 0){
          fprintf(stderr,
            "[SSDIOCTX] io_uring write failed for ssd_block=%lu: %s\n",
            (unsigned long)finished_ssd_id, std::strerror(-cqe->res));
          chunk_success = false;
        }else if(cqe->res != block_bytes){
          fprintf(stderr,
            "[SSDIOCTX] io_uring short write for ssd_block=%lu: wrote %d / %ld bytes\n",
            (unsigned long)finished_ssd_id, cqe->res, (long)block_bytes);
          chunk_success = false;
        }
        io_uring_cqe_seen(&ring_, cqe);
      }
      if (!chunk_success) {
        return false;
      }
    }
    return true;
  }

  auto SSDIOCTX::read_with_io_uring(const torch::Tensor& cpu_block_ids,
                                    const torch::Tensor& ssd_block_ids) -> bool{
    const int64_t num_transfer_blocks = cpu_block_ids.numel();
    if (num_transfer_blocks == 0) {
      return true;
    }
    if (queue_depth_ <= 0) {
      fprintf(stderr, "[SSDIOCTX] invalid io_uring queue_depth=%d\n", queue_depth_);
      return false;
    }

    const int64_t* cpu_ids = cpu_block_ids.data_ptr<int64_t>();
    const int64_t* ssd_ids = ssd_block_ids.data_ptr<int64_t>();
    const int64_t cpu_stride = cpu_block_ids.stride(0);
    const int64_t ssd_stride = ssd_block_ids.stride(0);
    const int64_t block_bytes = slice_bytes_ * layer_num_ * kv_dim_;
    char* base = reinterpret_cast<char*>(cpu_tensor_.data_ptr());
    const int64_t iovec_size = layer_num_ * kv_dim_;
    const int64_t chunk_capacity = std::min<int64_t>(queue_depth_, num_transfer_blocks);
    std::vector<struct iovec> iovecs(iovec_size * chunk_capacity);

    for (int64_t chunk_start = 0;
         chunk_start < num_transfer_blocks;
         chunk_start += chunk_capacity) {
      const int64_t chunk_size =
        std::min<int64_t>(chunk_capacity, num_transfer_blocks - chunk_start);

      for (int64_t j = 0; j < chunk_size; ++j) {
        const int64_t i = chunk_start + j;
        const int64_t cpu_block_id = cpu_ids[i * cpu_stride];
        const int64_t ssd_block_id = ssd_ids[i * ssd_stride];
        struct iovec* iov = &iovecs[j * iovec_size];
        int iov_idx = 0;
        for(int64_t layer_id = 0; layer_id < layer_num_; ++layer_id){
          for(int64_t kv_id = 0; kv_id < kv_dim_; ++kv_id){
            iov[iov_idx].iov_base = base + (
              (layer_id * kv_dim_ * cpu_num_blocks_ + kv_id * cpu_num_blocks_ + cpu_block_id)
              * slice_bytes_
            );
            iov[iov_idx].iov_len = static_cast<size_t>(slice_bytes_);
            ++iov_idx;
          }
        }

        const int64_t file_idx = ssd_block_id / blocks_per_file_;
        const int64_t block_idx = ssd_block_id % blocks_per_file_;
        const off_t offset = static_cast<off_t>(block_idx * block_bytes);
        const int fd = fds_.at(file_idx);
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if(!sqe){
          fprintf(stderr,"[SSDIOCTX] Fatal: io_uring SQ ring is full\n");
          return false;
        }
        io_uring_prep_readv(sqe, fd, iov, static_cast<unsigned>(iovec_size), offset);
        io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(static_cast<uintptr_t>(ssd_block_id)));
      }

      int submitted = 0;
      while (submitted < chunk_size) {
        const int ret = io_uring_submit(&ring_);
        if(ret < 0){
          fprintf(stderr, "[SSDIOCTX] io_uring_submit failed: %s\n", std::strerror(-ret));
          return false;
        }
        if(ret == 0){
          fprintf(stderr,
            "[SSDIOCTX] io_uring_submit submitted 0 requests: submitted=%d chunk_size=%ld\n",
            submitted, (long)chunk_size);
          return false;
        }
        submitted += ret;
      }

      bool chunk_success = true;
      for(int64_t j = 0; j < chunk_size; ++j){
        struct io_uring_cqe* cqe;
        int ret = io_uring_wait_cqe(&ring_, &cqe);
        if(ret < 0){
          fprintf(stderr, "[SSDIOCTX] io_uring_wait_cqe failed: %s\n",std::strerror(-ret));
          return false;
        }
        const uint64_t finished_ssd_id = reinterpret_cast<uintptr_t>(io_uring_cqe_get_data(cqe));
        if(cqe->res < 0){
          fprintf(stderr,
            "[SSDIOCTX] io_uring read failed for ssd_block=%lu: %s\n",
            (unsigned long)finished_ssd_id, std::strerror(-cqe->res));
          chunk_success = false;
        }else if(cqe->res != block_bytes){
          fprintf(stderr,
            "[SSDIOCTX] io_uring short read for ssd_block=%lu: got %d / %ld bytes\n",
            (unsigned long)finished_ssd_id, cqe->res, (long)block_bytes);
          chunk_success = false;
        }
        io_uring_cqe_seen(&ring_, cqe);
      }
      if (!chunk_success) {
        return false;
      }
    }
    return true;
  }

  auto SSDIOCTX::write_with_file_io(const torch::Tensor& cpu_block_ids,
                                    const torch::Tensor& ssd_block_ids) -> bool{
    const int64_t num_transfer_blocks = cpu_block_ids.numel();
    const int64_t* cpu_ids = cpu_block_ids.data_ptr<int64_t>();
    const int64_t* ssd_ids = ssd_block_ids.data_ptr<int64_t>();
    const int64_t cpu_stride = cpu_block_ids.stride(0);
    const int64_t ssd_stride = ssd_block_ids.stride(0);
    const int64_t block_bytes = slice_bytes_ * layer_num_ * kv_dim_;
    char* base = reinterpret_cast<char*>(cpu_tensor_.data_ptr());
    for(int64_t i = 0; i < num_transfer_blocks; i++){
      const int64_t cpu_block_id = cpu_ids[i * cpu_stride];
      const int64_t ssd_block_id = ssd_ids[i * ssd_stride];
      iovecs_.clear();
      for (int64_t layer_id = 0; layer_id < layer_num_; ++layer_id){
        for (int64_t kv_id = 0; kv_id < kv_dim_; kv_id++){
          struct iovec iov;
          iov.iov_base = base + (
            (layer_id * kv_dim_ * cpu_num_blocks_ + kv_id * cpu_num_blocks_ + cpu_block_id)
            * slice_bytes_
          );
          iov.iov_len = static_cast<size_t>(slice_bytes_);
          iovecs_.push_back(iov);
        }
      }
      const int64_t file_idx = ssd_block_id / blocks_per_file_;
      const int64_t block_idx = ssd_block_id % blocks_per_file_;
      const off_t offset = static_cast<off_t>(block_idx * block_bytes);
      const int fd = fds_.at(file_idx);
      const ssize_t written = pwritev_retry(fd, iovecs_.data(), static_cast<int>(iovecs_.size()), offset);
      if (written < 0) {
        const int err = errno;
        fprintf(stderr,
          "[SSDIOCTX] pwritev failed: cpu_block=%ld ssd_block=%ld "
          "file_idx=%ld offset=%ld: %s\n",
          (long)cpu_block_id, (long)ssd_block_id,
          (long)file_idx, (long)offset, std::strerror(err));
        return false;
      }
      if (written != block_bytes) {
        fprintf(stderr,
            "[SSDIOCTX] partial write: ssd_block=%ld wrote %zd / %ld bytes "
            "(disk full? device error?)\n",
            (long)ssd_block_id, written, (long)block_bytes);
        return false;
      }
    }
    return true;
  }

  auto SSDIOCTX::read_with_file_io(const torch::Tensor& cpu_block_ids,
                                   const torch::Tensor& ssd_block_ids) -> bool{
    const int64_t num_transfer_blocks = cpu_block_ids.numel();
    const int64_t* cpu_ids = cpu_block_ids.data_ptr<int64_t>();
    const int64_t* ssd_ids = ssd_block_ids.data_ptr<int64_t>();
    const int64_t cpu_stride = cpu_block_ids.stride(0);
    const int64_t ssd_stride = ssd_block_ids.stride(0);
    const int64_t block_bytes = slice_bytes_ * layer_num_ * kv_dim_;
    char* base = reinterpret_cast<char*>(cpu_tensor_.data_ptr());
    
    for(int64_t i = 0; i < num_transfer_blocks; ++i){
      const int64_t cpu_block_id = cpu_ids[i * cpu_stride];
      const int64_t ssd_block_id = ssd_ids[i * ssd_stride];
      iovecs_.clear();
      for(int64_t layer_id = 0; layer_id < layer_num_; ++layer_id){
        for(int64_t kv_id = 0; kv_id < kv_dim_; ++kv_id){
          struct iovec iov;
          iov.iov_base = base + (
            (layer_id * kv_dim_ * cpu_num_blocks_ + kv_id * cpu_num_blocks_ + cpu_block_id)
            * slice_bytes_
          );
          iov.iov_len = static_cast<size_t>(slice_bytes_);
          iovecs_.push_back(iov);
        }
      }
      const int64_t file_idx = ssd_block_id / blocks_per_file_;
      const int64_t block_idx = ssd_block_id % blocks_per_file_;
      const off_t offset = static_cast<off_t>(block_idx * block_bytes);
      const int fd = fds_.at(file_idx);
      const ssize_t read_bytes = preadv_retry(fd, iovecs_.data(), static_cast<int>(iovecs_.size()), offset);
      if (read_bytes < 0) {
        const int err = errno;
        fprintf(stderr,
          "[SSDIOCTX] preadv failed: cpu_block=%ld ssd_block=%ld "
          "file_idx=%ld offset=%ld: %s\n",
          (long)cpu_block_id, (long)ssd_block_id,
          (long)file_idx, (long)offset, std::strerror(err));
        return false;
      }
      if (read_bytes != block_bytes) {
        fprintf(stderr,
          "[SSDIOCTX] short read: ssd_block=%ld got %zd / %ld bytes "
          "(file not sized correctly? reading unwritten block?)\n",
          (long)ssd_block_id, read_bytes, (long)block_bytes);
        return false;
      }
    }
    return true;
  }   
}
