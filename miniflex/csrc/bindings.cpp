#include <torch/extension.h>

#include "ssd_io_uring.h"
#include "transfer.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<miniflex::SSDIOCTX>(m, "SSDIOCTX")
      .def(pybind11::init<int,
                          int64_t,
                          torch::Tensor,
                          int64_t,
                          int64_t,
                          int64_t,
                          int64_t,
                          std::vector<std::string>,
                          bool>(),
           pybind11::arg("queue_depth"),
           pybind11::arg("blocks_per_file"),
           pybind11::arg("cpu_tensor"),
           pybind11::arg("layer_num"),
           pybind11::arg("kv_dim"),
           pybind11::arg("cpu_num_blocks"),
           pybind11::arg("slice_bytes"),
           pybind11::arg("file_paths"),
           pybind11::arg("use_direct_io") = true)
      .def("transfer_blocks",
           &miniflex::SSDIOCTX::transfer_blocks,
           pybind11::arg("src_block_ids"),
           pybind11::arg("dst_block_ids"),
           pybind11::arg("is_read"))
      .def("is_using_io_uring", &miniflex::SSDIOCTX::is_using_io_uring);

  pybind11::class_<miniflex::GPUCPUTransferCTX>(m, "GPUCPUTransferCTX")
      .def(pybind11::init([](torch::Tensor cpu_tensor,
                             std::vector<torch::Tensor> gpu_tensors,
                             int64_t num_layers,
                             int64_t kv_dim,
                             int64_t cpu_num_blocks,
                             int64_t gpu_num_blocks,
                             int64_t slice_bytes) {
             if (static_cast<int64_t>(gpu_tensors.size()) != num_layers) {
               throw std::invalid_argument(
                   "gpu_tensors length must equal num_layers");
             }
             char* cpu_ptr = static_cast<char*>(cpu_tensor.data_ptr());
             std::vector<char*> gpu_ptrs;
             gpu_ptrs.reserve(gpu_tensors.size());
             for (auto& t : gpu_tensors) {
               gpu_ptrs.push_back(static_cast<char*>(t.data_ptr()));
             }
             return std::make_unique<miniflex::GPUCPUTransferCTX>(
                 cpu_ptr, gpu_ptrs, num_layers, kv_dim, cpu_num_blocks,
                 gpu_num_blocks, slice_bytes);
           }),
           pybind11::arg("cpu_tensor"),
           pybind11::arg("gpu_tensors"),
           pybind11::arg("num_layers"),
           pybind11::arg("kv_dim"),
           pybind11::arg("cpu_num_blocks"),
           pybind11::arg("gpu_num_blocks"),
           pybind11::arg("slice_bytes"))
      .def("transfer_blocks",
           &miniflex::GPUCPUTransferCTX::transfer_blocks,
           pybind11::arg("src_block_ids"),
           pybind11::arg("dst_block_ids"),
           pybind11::arg("is_h2d"));
}
