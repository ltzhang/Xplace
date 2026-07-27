#pragma once
#include "MazeRoute.h"
#include "PatternRoute.h"
#include "common/common.h"
#include <torch/extension.h>
#include <vector>
#include "gpugr/db/GrNet.h"
#include "gpugr/gr/RouteResource.h"

namespace gr {

typedef int dtype;

// Edge-level routability of a routed solution (WiseSyn R2-19).
//
// `ovfl_nets` -- the historical number -- counts NETS that touch at least one over-capacity gcell
// edge or via node. It is a reach measure, not a magnitude one: a single hot gcell crossed by 300
// nets contributes 300, and the count scales with design size rather than with how badly the design
// is congested. The magnitude quantities below are what a routability verdict should key off; the
// net count stays available as a secondary diagnostic.
struct OverflowReport {
    double total_edge_ovfl = 0.0;   // sum of max(0, demand - capacity) over gcell edges, in tracks
    double max_edge_ovfl = 0.0;     // worst single edge, in tracks
    long long ovfl_edges = 0;       // number of over-capacity gcell edges
    // The same quantity counting WIRE demand only, i.e. excluding the via resource. OpenROAD's GRT
    // congestion table is a wire-demand-vs-capacity table, so this is the number that compares
    // like-for-like with an external router's verdict; total_edge_ovfl is GGR's own full model.
    double wire_edge_ovfl = 0.0;
    long long wire_ovfl_edges = 0;
    long long routable_edges = 0;   // edges with nonzero capacity, i.e. the denominator
    int ovfl_nets = 0;              // secondary: nets touching any over-capacity resource
    int unrouted_nets = 0;          // nets the router failed to resolve at all
    std::vector<double> layer_total;  // per-layer total_edge_ovfl
    std::vector<double> layer_max;    // per-layer max_edge_ovfl
};

// Query free/total GPU memory in bytes (cudaMemGetInfo). Returns false if the CUDA query fails. Defined
// in GPURouter.cu so host translation units (RouteForce.cpp) can gate GGR on GPU memory without pulling
// in <cuda_runtime.h> into their host compile.
bool ggrQueryGpuMem(size_t* freeBytes, size_t* totalBytes);

class GPURouter {
public:
    GPURouter(){};
    GPURouter(
        int device_id, int layer, int x, int y, int N_, int cgxsize_, int cgysize_, int direction, int csrn_scale) {
        (void)initialize(device_id, layer, x, y, N_, cgxsize_, cgysize_, direction, csrn_scale);
    }
    ~GPURouter();

    // Returns false (having freed nothing new) if any device allocation failed — the caller must then
    // decline routing (skip writeGuides) so the driver falls back to its interconnect estimate (#7).
    [[nodiscard]] bool initialize(
        int device_id, int layer, int x, int y, int N_, int cgxsize_, int cgysize_, int direction, int csrn_scale);

    void setMap(const std::vector<float> &cap,
                const std::vector<float> &wir,
                const std::vector<float> &fixedL,
                const std::vector<float> &fix);
    // P2g (WiseSyn WS2): every stage that allocates or copies on the device is checked and
    // fail-closed — false means "decline GGR" (the caller skips writeGuides so the driver falls
    // back loudly to its interconnect estimate), never a crash on a null/partial buffer.
    bool setFromNets(std::vector<GrNet> &nets, int numPlPin_);
    // `routesOverride`, when non-null, is a host snapshot taken by snapshotRoutes() -- the rip-up
    // loop hands back the BEST iteration's routes rather than whatever the last pass produced.
    bool setToNets(std::vector<GrNet> &nets, const std::vector<int> *routesOverride = nullptr);
    bool route(std::vector<GrNet> &nets, int iterleft);
    void setUnitViaMultiplier(float w);
    void setUnitVioCost(std::vector<float>& cost, float discount);
    void setLogisticSlope(float value);
    void setUnitViaCost(float value);
    void query();

    // The resource model every capacity comparison in this router uses. Set once before routing:
    // the router must optimize against the SAME capacity the verdict is judged against.
    void setResourceModel(const RouteResourceModel &model) { resourceModel = model; }
    const RouteResourceModel &getResourceModel() const { return resourceModel; }

    // Edge-level routability of the solution currently on the device, recomputed at the end of each
    // routing pass. Valid after route().
    const OverflowReport &getOverflow() const { return overflow; }

    // Copy the device route array into `out` so a later pass that turns out worse can be discarded.
    // Bounded: `out` is exactly the size of the read-back buffer setToNets already allocates.
    [[nodiscard]] bool snapshotRoutes(std::vector<int> &out);

    // Put a snapshot back on the device and rebuild the wire/via demand grids and the overflow
    // report from it, so the congestion maps and the emitted guide describe the same solution.
    [[nodiscard]] bool restoreRoutes(const std::vector<int> &snapshot, std::vector<GrNet> &nets);

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
    int getNumOvflNets() { return numOvflNets; }

private:
    // Recompute the overflow marking + the edge-level report from the demand grids currently on the
    // device. Run at the end of every routing pass and after a snapshot restore.
    void recomputeOverflow(std::vector<GrNet> &nets);

    GPUMazeRouter gpuMR;
    // routes:
    //    (x, y): starting point x, length |y|; negative y implies vias

    int DEVICE_ID;
    int LAYER, N, X, Y, NET_NUM, DIRECTION;
    int COARSENING_SCALE;
    int cgxsize, cgysize;

    int *pinNum = nullptr, *pinNumOffset = nullptr, *pins = nullptr;
    int *routes = nullptr, *routesOffset = nullptr, *routesOffsetCPU = nullptr, *pinNumCPU = nullptr;
    int *allpins;
    int *points = nullptr, *gbpoints = nullptr;
    int *gbpinRoutes = nullptr, *gbpin2netId = nullptr, *plPinId2gbPinId = nullptr;
    float *capacity, *wireDist, *fixedLength, *fixed;
    int *wires, *vias, *prev;
    int *isOverflowWire, *isOverflowVia, *isOverflowNet = nullptr;
    int *boundaries, *isLocked;
    int *wiresCPU, *viasCPU;
    int *cudaIndex, *cudaCostIndex;
    int *modifiedVia, *modifiedWire, *viasToBeUpdated, *wiresToBeUpdated;
    dtype *dist, *cost, *viaCost;
    int64_t *costSum;
    float *unitShortCostDiscounted, unitViaCost, unitViaMultiplier = 1, logisticSlope = 1, *cell_resource;

    int numGbPin, numPlPin;

    int numOvflNets = 0;
    RouteResourceModel resourceModel{};
    OverflowReport overflow{};
    // Managed accumulators for the per-layer edge-overflow reduction (LAYER entries each) plus the
    // three scalars. Allocated in initialize(), freed in the destructor.
    // 2*LAYER doubles: [0,LAYER) = combined per-layer total, [LAYER,2*LAYER) = wire-only.
    double *layerOvflTotal = nullptr;
    double *layerOvflMax = nullptr;
    // [0] = over-capacity edges, [1] = edges with capacity, [2] = wire-only over-capacity edges
    unsigned long long *edgeOvflCounters = nullptr;

    const int MAX_BATCH_SIZE = 100, MAX_PIN_SIZE_PER_NET = 500000;

    std::vector<std::vector<short>> vis, visLL, visRR;
};

}  // namespace gr
