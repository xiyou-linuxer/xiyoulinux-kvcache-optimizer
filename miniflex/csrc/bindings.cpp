#include <torch/extension.h>

#include <stdexcept>

#include "gds_io.h"
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

  pybind11::class_<miniflex::GDSIOCTX>(m, "GDSIOCTX")
      .def(pybind11::init([](int64_t blocks_per_file,
                             std::vector<torch::Tensor> gpu_tensors,
                             int64_t layer_num,
                             int64_t kv_dim,
                             int64_t gpu_num_blocks,
                             int64_t slice_bytes,
                             int64_t gpu_block_step,
                             int64_t gpu_kv_pitch,
                             std::vector<std::string> file_paths) {
             if (static_cast<int64_t>(gpu_tensors.size()) != layer_num) {
               throw std::invalid_argument(
                   "gpu_tensors length must equal layer_num");
             }
             if (gpu_tensors.empty()) {
               throw std::invalid_argument("gpu_tensors must not be empty");
             }

             int gpu_device = -1;
             for (size_t layer = 0; layer < gpu_tensors.size(); ++layer) {
               const auto& tensor = gpu_tensors[layer];
               if (!tensor.defined() || !tensor.is_cuda()) {
                 throw std::invalid_argument(
                     "all gpu_tensors must be defined CUDA tensors");
               }
               if (!tensor.is_contiguous()) {
                 throw std::invalid_argument(
                     "all gpu_tensors must be contiguous for GDS");
               }
               if (tensor.numel() == 0) {
                 throw std::invalid_argument(
                     "all gpu_tensors must be non-empty for GDS");
               }

               const int tensor_device = tensor.get_device();
               if (layer == 0) {
                 gpu_device = tensor_device;
               } else if (tensor_device != gpu_device) {
                 throw std::invalid_argument(
                     "all gpu_tensors must be on the same CUDA device");
               }
             }

             return std::make_unique<miniflex::GDSIOCTX>(
                 blocks_per_file, std::move(gpu_tensors), layer_num, kv_dim,
                 gpu_num_blocks, slice_bytes, gpu_block_step, gpu_kv_pitch,
                 std::move(file_paths));
           }),
           pybind11::arg("blocks_per_file"),
           pybind11::arg("gpu_tensors"),
           pybind11::arg("layer_num"),
           pybind11::arg("kv_dim"),
           pybind11::arg("gpu_num_blocks"),
           pybind11::arg("slice_bytes"),
           pybind11::arg("gpu_block_step"),
           pybind11::arg("gpu_kv_pitch"),
           pybind11::arg("file_paths"))
      .def("transfer_blocks",
           &miniflex::GDSIOCTX::transfer_blocks,
           pybind11::arg("src_block_ids"),
           pybind11::arg("dst_block_ids"),
           pybind11::arg("is_read"))
      .def("is_available", &miniflex::GDSIOCTX::is_available);

  pybind11::class_<miniflex::GPUCPUTransferCTX>(m, "GPUCPUTransferCTX")
      .def(pybind11::init([](std::vector<torch::Tensor> cpu_tensors,
                             std::vector<torch::Tensor> gpu_tensors,
                             int64_t num_layers,
                             int64_t kv_dim,
                             int64_t slice_bytes,
                             int64_t cpu_block_step,
                             int64_t cpu_kv_pitch,
                             int64_t gpu_block_step,
                             int64_t gpu_kv_pitch) {
             if (static_cast<int64_t>(gpu_tensors.size()) != num_layers ||
                 static_cast<int64_t>(cpu_tensors.size()) != num_layers) {
               throw std::invalid_argument(
                   "cpu_tensors/gpu_tensors length must equal num_layers");
             }
             std::vector<char*> cpu_ptrs;
             cpu_ptrs.reserve(cpu_tensors.size());
             for (auto& t : cpu_tensors) {
               cpu_ptrs.push_back(static_cast<char*>(t.data_ptr()));
             }
             std::vector<char*> gpu_ptrs;
             gpu_ptrs.reserve(gpu_tensors.size());
             for (auto& t : gpu_tensors) {
               gpu_ptrs.push_back(static_cast<char*>(t.data_ptr()));
             }
             return std::make_unique<miniflex::GPUCPUTransferCTX>(
                 cpu_ptrs, gpu_ptrs, num_layers, kv_dim, slice_bytes,
                 cpu_block_step, cpu_kv_pitch, gpu_block_step, gpu_kv_pitch);
           }),
           pybind11::arg("cpu_tensors"),
           pybind11::arg("gpu_tensors"),
           pybind11::arg("num_layers"),
           pybind11::arg("kv_dim"),
           pybind11::arg("slice_bytes"),
           pybind11::arg("cpu_block_step"),
           pybind11::arg("cpu_kv_pitch"),
           pybind11::arg("gpu_block_step"),
           pybind11::arg("gpu_kv_pitch"))
      .def("transfer_blocks",
           &miniflex::GPUCPUTransferCTX::transfer_blocks,
           pybind11::arg("src_block_ids"),
           pybind11::arg("dst_block_ids"),
           pybind11::arg("is_h2d"));
}
