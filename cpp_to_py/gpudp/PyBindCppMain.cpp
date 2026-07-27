#include "common/common.h"
#include "common/db/Database.h"
#include "gpudp/db/dp_torch.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace Xplace {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<dp::DPTorchRawDB, std::shared_ptr<dp::DPTorchRawDB>>(m, "DPTorchRawDB")
        .def(pybind11::init<torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            float,
                            float,
                            float,
                            float,
                            int,
                            int,
                            int,
                            float,
                            float>())
        .def("check", &dp::DPTorchRawDB::check)
        .def("scale", &dp::DPTorchRawDB::scale)
        .def("commit", &dp::DPTorchRawDB::commit)
        .def("rollback", &dp::DPTorchRawDB::rollback)
        .def("commit_from", &dp::DPTorchRawDB::commit_from)
        .def("commit_from_partial", &dp::DPTorchRawDB::commit_from_partial)
        .def("get_curr_cposx", &dp::DPTorchRawDB::get_curr_cposx, py::return_value_policy::move)
        .def("get_curr_cposy", &dp::DPTorchRawDB::get_curr_cposy, py::return_value_policy::move)
        .def("get_curr_lposx", &dp::DPTorchRawDB::get_curr_lposx, py::return_value_policy::move)
        .def("get_curr_lposy", &dp::DPTorchRawDB::get_curr_lposy, py::return_value_policy::move);

    m.def("create_dp_rawdb",
          [](torch::Tensor node_lpos_init_,
             torch::Tensor node_size_,
             torch::Tensor node_weight_,
             torch::Tensor is_macro_,
             torch::Tensor pin_rel_lpos_,
             torch::Tensor pin_id2node_id_,
             torch::Tensor pin_id2net_id_,
             torch::Tensor node2pin_list_,
             torch::Tensor node2pin_list_end_,
             torch::Tensor hyperedge_list_,
             torch::Tensor hyperedge_list_end_,
             torch::Tensor net_mask_,
             torch::Tensor node_id2region_id_,
             torch::Tensor region_boxes_,
             torch::Tensor region_boxes_end_,
             float xl_,
             float xh_,
             float yl_,
             float yh_,
             int num_conn_movable_nodes_,
             int num_movable_nodes_,
             int num_nodes_,
             float site_width_,
             float row_height_,
             std::vector<float> row_yl_,
             std::vector<float> row_h_) {
              return std::make_shared<dp::DPTorchRawDB>(node_lpos_init_,
                                                        node_size_,
                                                        node_weight_,
                                                        is_macro_,
                                                        pin_rel_lpos_,
                                                        pin_id2node_id_,
                                                        pin_id2net_id_,
                                                        node2pin_list_,
                                                        node2pin_list_end_,
                                                        hyperedge_list_,
                                                        hyperedge_list_end_,
                                                        net_mask_,
                                                        node_id2region_id_,
                                                        region_boxes_,
                                                        region_boxes_end_,
                                                        xl_,
                                                        xh_,
                                                        yl_,
                                                        yh_,
                                                        num_conn_movable_nodes_,
                                                        num_movable_nodes_,
                                                        num_nodes_,
                                                        site_width_,
                                                        row_height_,
                                                        std::move(row_yl_),
                                                        std::move(row_h_));
          },
          // The last two arguments are the physical placement-row table (bottom edges + heights) in
          // the same prescaled coordinate system as yl/yh. Defaulted to empty = the classic uniform
          // `row_height` grid, so every existing caller is unaffected.
          py::arg("node_lpos_init"), py::arg("node_size"), py::arg("node_weight"), py::arg("is_macro"),
          py::arg("pin_rel_lpos"), py::arg("pin_id2node_id"), py::arg("pin_id2net_id"), py::arg("node2pin_list"),
          py::arg("node2pin_list_end"), py::arg("hyperedge_list"), py::arg("hyperedge_list_end"),
          py::arg("net_mask"), py::arg("node_id2region_id"), py::arg("region_boxes"), py::arg("region_boxes_end"),
          py::arg("xl"), py::arg("xh"), py::arg("yl"), py::arg("yh"), py::arg("num_conn_movable_nodes"),
          py::arg("num_movable_nodes"), py::arg("num_nodes"), py::arg("site_width"), py::arg("row_height"),
          py::arg("row_yl") = std::vector<float>{}, py::arg("row_h") = std::vector<float>{});
    m.def("macroLegalization", [](std::shared_ptr<dp::DPTorchRawDB> at_db_ptr, int num_bins_x, int num_bins_y) {
        return dp::macroLegalization(*at_db_ptr, num_bins_x, num_bins_y);
    });
    m.def("abacusLegalization", [](std::shared_ptr<dp::DPTorchRawDB> at_db_ptr, int num_bins_x, int num_bins_y) {
        return dp::abacusLegalization(*at_db_ptr, num_bins_x, num_bins_y);
    });
    m.def("greedyLegalization",
          [](std::shared_ptr<dp::DPTorchRawDB> at_db_ptr, int num_bins_x, int num_bins_y, bool legalize_filler) {
              return dp::greedyLegalization(*at_db_ptr, num_bins_x, num_bins_y, legalize_filler);
          });
    m.def("kReorder",
          [](std::shared_ptr<dp::DPTorchRawDB> at_db_ptr, int num_bins_x, int num_bins_y, int K, int max_iters) {
              return dp::kReorder(*at_db_ptr, num_bins_x, num_bins_y, K, max_iters);
          });
    m.def(
        "globalSwap",
        [](std::shared_ptr<dp::DPTorchRawDB> at_db_ptr, int num_bins_x, int num_bins_y, int batch_size, int max_iters, float displacement_region_ratio) {
            return dp::globalSwap(*at_db_ptr, num_bins_x, num_bins_y, batch_size, max_iters, displacement_region_ratio);
        });
    m.def("independentSetMatching",
          [](std::shared_ptr<dp::DPTorchRawDB> at_db_ptr,
             int num_bins_x,
             int num_bins_y,
             int batch_size,
             int set_size,
             int max_iters) {
              return dp::independentSetMatching(*at_db_ptr, num_bins_x, num_bins_y, batch_size, set_size, max_iters);
          });
    m.def("legalityCheck", [](std::shared_ptr<dp::DPTorchRawDB> at_db_ptr, float scale_factor) {
        return dp::legalityCheck(*at_db_ptr, scale_factor);
    });
    m.def("fillerLegalization",
          [](std::shared_ptr<dp::DPTorchRawDB> at_db_ptr) { return dp::fillerLegalization(*at_db_ptr); });
}

}  // namespace Xplace
