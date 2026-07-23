#include "RouteForce.h"
#include "common/db/Database.h"
#include "io_parser/gp/GPDatabase.h"
#include "gpugr/db/GRDatabase.h"

namespace gr {

RouteForce::RouteForce(std::shared_ptr<gr::GRDatabase> grdb_) : grdb(*grdb_) {}

void RouteForce::run_ggr() {
    logger.enable_logger();
    utils::timer T_total;
    T_total.start();

    // Macro-aware GGR feasibility guard (rule #7 — decline loudly, never crash). Large hard macros push
    // the die (and thus the gcell grid) big enough that the per-batch dist/prev arrays — each
    // (MAX_BATCH_SIZE+6)*gridGraphSize*4 bytes, the dominant GPU allocation — can exceed the device's
    // free memory. The int32 index overflow that used to corrupt this path is fixed (see GPURouter.cu),
    // but a grid that genuinely does not fit must be declined here rather than dying in a failed
    // cudaMalloc + downstream exit(0). Skipping the routing (no writeGuides) makes the driver report
    // "no guide", so the caller falls back loudly to its FLUTE estimate.
    {
        const int64_t gridGraphSize = (int64_t)grdb.nLayers * grdb.nMaxGrid * grdb.nMaxGrid;
        // dist+prev dominate: 2 * (MAX_BATCH_SIZE+6=106) * grid * 4B; plus ~15 single-grid arrays * 4B
        // and costSum * 8B. Round the per-cell byte factor up for the coarse-grid + scratch extras.
        const int64_t bytesPerCell = 2 * 106 * 4 + 15 * 4 + 8;  // ~= 916 B/cell
        const int64_t needBytes = gridGraphSize * bytesPerCell;
        size_t freeB = 0, totalB = 0;
        const bool queried = ggrQueryGpuMem(&freeB, &totalB);
        // Fail CLOSED when the query itself fails: if we cannot read free GPU memory we cannot admit the
        // grid, so decline rather than blindly proceed into cudaMalloc (the estimate is only an early
        // admission check, not a substitute for the checked allocations in GPURouter::initialize).
        if (!queried) {
            logger.warning(
                "GGR skipped: cudaMemGetInfo failed, cannot verify the routing grid (~%.1f GB, "
                "nMaxGrid=%d, layers=%d) fits — declining GGR; the caller keeps its interconnect "
                "estimate.",
                needBytes / 1e9, grdb.nMaxGrid, grdb.nLayers);
            logger.reset_logger();
            return;  // no router.initialize / route / writeGuides -> driver sees no guide -> loud fallback
        }
        if (static_cast<size_t>(needBytes) + static_cast<size_t>(needBytes) / 10 > freeB) {
            logger.warning(
                "GGR skipped: routing grid needs ~%.1f GB but only ~%.1f GB is free on the GPU "
                "(nMaxGrid=%d, layers=%d). Declining GGR for this design; the caller keeps its "
                "interconnect estimate.",
                needBytes / 1e9, freeB / 1e9, grdb.nMaxGrid, grdb.nLayers);
            logger.reset_logger();
            return;  // no router.initialize / route / writeGuides -> driver sees no guide -> loud fallback
        }
    }
    // we only need the PR segment, our current data structure unsupport MR route force
    // if rrrIters > 0, this router can only be used for congestion map computation or solution evaluation
    int runMazeRouteTimes = grSetting.rrrIters;
    // Parameters
    int rrrIterLimit = 1 + runMazeRouteTimes;
    double _unitWireCostRaw = 0.5 * grdb.microns / grdb.m2pitch;
    double _unitViaCostRaw = 4;
    double _unitViaCost = _unitViaCostRaw / _unitWireCostRaw * grdb.microns;
    double _unitShortVioCostRaw = 500;
    double rrrInitVioCostDiscount = 0.1;

    if (!router.initialize(grSetting.deviceId,
                           grdb.nLayers,
                           grdb.xSize,
                           grdb.ySize,
                           grdb.nMaxGrid,
                           grdb.cgxsize,
                           grdb.cgysize,
                           grdb.m1direction,
                           grdb.csrnScale)) {
        // A device allocation failed despite passing the estimate (fragmentation / concurrent GPU
        // use). Decline atomically — no route / writeGuides — so the driver falls back loudly (#7).
        logger.warning("GGR skipped: GPU router initialization failed to allocate; declining GGR, the "
                       "caller keeps its interconnect estimate.");
        logger.reset_logger();
        return;
    }
    router.setMap(grdb.capacity, grdb.wireDist, grdb.fixedLength, grdb.fixedUsage);

    std::vector<float> _unitShortVioCost(grdb.nLayers), _unitShortVioCostDiscounted(grdb.nLayers);
    if (!router.setFromNets(grdb.grNets, grdb.gpdb.getPins().size())) {
        // P2g: a net-buffer allocation failed after initialize passed — decline atomically (#7).
        logger.warning("GGR skipped: GPU net-buffer allocation failed; declining GGR, the caller "
                       "keeps its interconnect estimate.");
        logger.reset_logger();
        return;
    }
    router.setUnitViaCost(_unitViaCost);
    for (int i = 0; i < grdb.nLayers; ++i) {
        _unitShortVioCost[i] =
            _unitShortVioCostRaw * grdb.layerWidth[i] * grdb.microns / grdb.m2pitch / grdb.m2pitch / _unitWireCostRaw;
    }
    double tot_time = 0;
    for (int iter = 0; iter < rrrIterLimit; iter++) {
        router.setLogisticSlope(1 << iter);
        router.setUnitVioCost(_unitShortVioCost, 0.1);
        if (iter == 0) {
            router.setUnitViaMultiplier(1);
        } else {
            router.setUnitViaMultiplier(std::max(100 / pow(5, iter - 1), 4.0));
            router.setUnitVioCost(_unitShortVioCost,
                                  rrrInitVioCostDiscount + (1.0 - rrrInitVioCostDiscount) / (rrrIterLimit - 1) * iter);
        }
        utils::timer T;
        T.start();
        if (!router.route(grdb.grNets, iter)) {
            // P2g: an in-loop scratch allocation failed — decline (no setToNets / writeGuides).
            logger.warning("GGR skipped: GPU routing scratch allocation failed; declining GGR, the "
                           "caller keeps its interconnect estimate.");
            logger.reset_logger();
            return;
        }
        tot_time += T.elapsed();
        logger.info("##### GPU Routing Iter: %d Time: %.4f #####", iter, T.elapsed());
        // break;
    }
    if (!router.setToNets(grdb.grNets)) {
        // P2g: the route read-back failed — the routes buffer would be garbage; decline (#7).
        logger.warning("GGR skipped: GPU route read-back failed; declining GGR, the caller keeps "
                       "its interconnect estimate.");
        logger.reset_logger();
        return;
    }
    logger.info("Total GPU Routing time: %.4f", tot_time);

    if (grSetting.routeGuideFile != "") {
        grdb.writeGuides(grSetting.routeGuideFile);
    }

    logger.info("Total GPU GR Time: %.4f", T_total.elapsed());
    logger.reset_logger();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RouteForce::getDemandMap() { return router.getDemandMap(); }

torch::Tensor RouteForce::getCapacityMap() { return router.getCapacityMap(); }

torch::Tensor RouteForce::calcRouteGrad(torch::Tensor mask_map,
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
                                        int num_nodes) {
    return router.calcRouteGrad(mask_map,
                                wire_dmd_map_2d,
                                via_dmd_map_2d,
                                cap_map_2d,
                                dist_weights,
                                wirelength_weights,
                                route_gradmat,
                                node2pin_list,
                                node2pin_list_end,
                                grad_weight,
                                unit_wire_cost,
                                unit_via_cost,
                                num_nodes);
};

torch::Tensor RouteForce::calcFillerRouteGrad(torch::Tensor filler_pos,
                                              torch::Tensor filler_size,
                                              torch::Tensor filler_weight,
                                              torch::Tensor expand_ratio,
                                              torch::Tensor grad_mat,
                                              float grad_weight,
                                              float unit_len_x,
                                              float unit_len_y,
                                              int num_bin_x,
                                              int num_bin_y,
                                              int num_fillers) {
    return router.calcFillerRouteGrad(filler_pos,
                                      filler_size,
                                      filler_weight,
                                      expand_ratio,
                                      grad_mat,
                                      grad_weight,
                                      unit_len_x,
                                      unit_len_y,
                                      num_bin_x,
                                      num_bin_y,
                                      num_fillers);
}

torch::Tensor RouteForce::calcPseudoPinGrad(torch::Tensor node_pos, torch::Tensor pseudo_pin_pos, float gamma) {
    return router.calcPseudoPinGrad(node_pos, pseudo_pin_pos, gamma);
}

torch::Tensor RouteForce::calcNodeInflateRatio(torch::Tensor node_pos,
                                               torch::Tensor node_size,
                                               torch::Tensor node_weight,
                                               torch::Tensor expand_ratio,
                                               torch::Tensor inflate_mat,
                                               float grad_weight,
                                               float unit_len_x,
                                               float unit_len_y,
                                               int num_bin_x,
                                               int num_bin_y,
                                               bool use_weighted_inflation) {
    return router.calcNodeInflateRatio(node_pos,
                                       node_size,
                                       node_weight,
                                       expand_ratio,
                                       inflate_mat,
                                       grad_weight,
                                       unit_len_x,
                                       unit_len_y,
                                       num_bin_x,
                                       num_bin_y,
                                       use_weighted_inflation);
}

torch::Tensor RouteForce::calcInflatedPinRelCpos(torch::Tensor node_inflate_ratio,
                                                 torch::Tensor old_pin_rel_cpos,
                                                 torch::Tensor pin_id2node_id,
                                                 int num_movable_conn_nodes) {
    return router.calcInflatedPinRelCpos(node_inflate_ratio, old_pin_rel_cpos, pin_id2node_id, num_movable_conn_nodes);
}

int RouteForce::getNumOvflNets() { return router.getNumOvflNets(); }

int RouteForce::getMicrons() { return grdb.microns; }

std::tuple<int, int> RouteForce::getGcellStep() { return {grdb.mainGcellStepX, grdb.mainGcellStepY}; }

std::vector<int> RouteForce::getLayerPitch() { return grdb.layerPitch; }

std::vector<int> RouteForce::getLayerWidth() { return grdb.layerWidth; }

}  // namespace gr