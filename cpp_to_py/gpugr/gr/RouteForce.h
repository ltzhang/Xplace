#pragma once
#include "common/common.h"
#include "gpugr/gr/GPURouter.h"

namespace gr {
class GRDatabase;
}

namespace gr {

class RouteForce {
public:
    RouteForce(std::shared_ptr<gr::GRDatabase> grdb_);
    void run_ggr();

public:
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getDemandMap();
    torch::Tensor getCapacityMap();
    torch::Tensor calcRouteGrad(torch::Tensor mask_map,
                                torch::Tensor wire_dmd_map_2d,
                                torch::Tensor via_dmd_map_2d,
                                torch::Tensor cap_map_2d,
                                torch::Tensor dist_weights,
                                torch::Tensor wirelength_weights,
                                torch::Tensor route_gradmat,
                                torch::Tensor node2pin_list,
                                torch::Tensor node2pin_list_end,
                                float grad_weight,
                                float unit_wire_cost,
                                float unit_via_cost,
                                int num_nodes);
    torch::Tensor calcFillerRouteGrad(torch::Tensor filler_pos,
                                      torch::Tensor filler_size,
                                      torch::Tensor filler_weight,
                                      torch::Tensor expand_ratio,
                                      torch::Tensor grad_mat,
                                      float grad_weight,
                                      float unit_len_x,
                                      float unit_len_y,
                                      int num_bin_x,
                                      int num_bin_y,
                                      int num_fillers);
    torch::Tensor calcPseudoPinGrad(torch::Tensor node_pos, torch::Tensor pseudo_pin_pos, float gamma);
    torch::Tensor calcNodeInflateRatio(torch::Tensor node_pos,
                                       torch::Tensor node_size,
                                       torch::Tensor node_weight,
                                       torch::Tensor expand_ratio,
                                       torch::Tensor inflate_mat,
                                       float grad_weight,
                                       float unit_len_x,
                                       float unit_len_y,
                                       int num_bin_x,
                                       int num_bin_y,
                                       bool use_weighted_inflation);
    torch::Tensor calcInflatedPinRelCpos(torch::Tensor node_inflate_ratio,
                                         torch::Tensor old_pin_rel_cpos,
                                         torch::Tensor pin_id2node_id,
                                         int num_movable_nodes);
    int getNumOvflNets();
    int getMicrons();
    std::tuple<int, int> getGcellStep();
    std::vector<int> getLayerPitch();
    std::vector<int> getLayerWidth();

    // Edge-level routability of the EMITTED solution (WiseSyn R2-19). This is the primary
    // routability number; getNumOvflNets() is a secondary diagnostic that counts reach, not
    // magnitude. Zeroed until a route completes, so a declined GGR never reads as clean -- callers
    // must check that a guide was produced, exactly as before.
    double getEdgeOverflow() const { return report.total_edge_ovfl; }
    double getMaxEdgeOverflow() const { return report.max_edge_ovfl; }
    long long getOverflowEdges() const { return report.ovfl_edges; }
    double getWireEdgeOverflow() const { return report.wire_edge_ovfl; }
    long long getWireOverflowEdges() const { return report.wire_ovfl_edges; }
    long long getRoutableEdges() const { return report.routable_edges; }
    int getUnroutedNets() const { return report.unrouted_nets; }
    std::vector<double> getLayerEdgeOverflow() const { return report.layer_total; }
    std::vector<double> getLayerMaxEdgeOverflow() const { return report.layer_max; }
    // The resource model the numbers above were measured against, so a verdict is reproducible.
    std::tuple<int, double> getResourceModel() const {
        return {resource.via_mode, static_cast<double>(resource.capacity_derate)};
    }
    // How the rip-up loop ended: passes run, which pass was kept, why it stopped.
    std::tuple<int, int, std::string> getRrrStatus() const { return {rrrPasses, rrrBestIter, rrrStop}; }

private:
    gr::GRDatabase& grdb;
    gr::GPURouter router;
    OverflowReport report{};
    RouteResourceModel resource{};
    int rrrPasses = 0;
    int rrrBestIter = -1;
    std::string rrrStop = "not run";
};

}  // namespace gr