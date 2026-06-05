#include <torch/extension.h>

#include "ssd_io_uring.h"

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
}
