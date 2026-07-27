#pragma once
// Global-route resource model: how a gcell's routing tracks are consumed, and therefore what
// "over capacity" means (WiseSyn R2-19).
//
// The router grids each layer into gcells and gives every gcell edge a capacity equal to the number
// of routing tracks that cross it. Wire demand is a plain track count: one net crossing the edge
// takes one track. Via demand is NOT a track count -- a via lands *inside* a gcell rather than
// crossing its boundary -- so it has to be converted into track units before it can be compared
// against a track capacity.
//
// Two things about the historical conversion made the overflow verdict untrustworthy:
//
//   * The via term was `sqrt(0.5 * (n_i + n_{i+1})) * 1.5`. That is a smooth *steering* cost lifted
//     out of the maze router's objective, not a resource: a SINGLE via already charges 1.06 tracks,
//     the term keeps growing without bound past the edge's own capacity, and its units are not
//     tracks at all. Reused verbatim as the hard over-capacity predicate it declares edges
//     overflowing that carry no excess wire demand whatsoever -- which is why a design could report
//     thousands of "overflowing nets" while its edge-level wire overflow was exactly zero.
//
//   * Capacity was the raw track count with no reservation, while every real global router holds
//     part of each layer back for pin access, local connections and non-default rules.
//
// Both are replaced here by quantities that are in tracks and are bounded by the resource that
// actually exists. This header is deliberately dependency-free (no CUDA, no torch, no xplace types)
// so it is usable from device code, from host code, and from a unit test.

#include <cmath>

#if defined(__CUDACC__)
#define GGR_RESOURCE_FN __host__ __device__ inline
#else
#define GGR_RESOURCE_FN inline
#endif

namespace gr {

// How a via landing is charged against the routing tracks of the layer it lands on.
enum class ViaResourceMode : int {
    // Historical steering surcharge, kept selectable so a result can be reproduced against the old
    // model. Not a track quantity; do not judge routability with it.
    LegacySurcharge = 0,
    // Expected number of DISTINCT tracks the landings block. In tracks, bounded by the tracks that
    // exist, and exactly 1 for a single via.
    TrackOccupancy = 1,
};

// The complete description of the resource model a route was measured against. Carried alongside
// every overflow number so a reported verdict can be reproduced.
struct RouteResourceModel {
    int via_mode = static_cast<int>(ViaResourceMode::TrackOccupancy);
    // Scale of the LegacySurcharge term. Ignored by TrackOccupancy.
    float legacy_scale = 1.5f;
    // Fraction of each layer's raw track count held back for pin access / local nets / non-default
    // rules, i.e. effective capacity = raw * (1 - capacity_derate). Applied once when the capacity
    // grid is built, so the router optimizes against the same capacity the verdict is judged
    // against -- a router steering by one capacity and reporting against another cannot converge.
    float capacity_derate = 0.0f;
};

// Clamp a derate to a range that leaves a usable grid. A derate of 1.0 would zero every layer and
// make every edge overflow by construction.
GGR_RESOURCE_FN float clampDerate(float derate) {
    if (!(derate > 0.0f)) return 0.0f;  // also catches NaN
    return derate > 0.9f ? 0.9f : derate;
}

// Effective per-edge capacity, in tracks, after the layer reservation.
GGR_RESOURCE_FN float derateCapacity(float raw_tracks, float derate) {
    const float eff = raw_tracks * (1.0f - clampDerate(derate));
    return eff > 0.0f ? eff : 0.0f;
}

// Expected number of DISTINCT routing tracks blocked when `n` via landings fall inside a gcell that
// offers `tracks` routing tracks.
//
// Landings are spread over the tracks rather than stacked on one, so `n` vias block fewer than `n`
// tracks once n approaches the track count; the balls-in-bins expectation
// `tracks * (1 - (1 - 1/tracks)^n)` is the standard closed form. It is 0 at n = 0, exactly 1 at
// n = 1, monotone and concave in n, and saturates at `tracks` -- so unlike the legacy surcharge it
// can never on its own push an edge past its own capacity.
GGR_RESOURCE_FN float viaOccupiedTracks(float n, float tracks) {
    if (!(n > 0.0f)) return 0.0f;
    if (!(tracks > 1.0f)) {
        // One track or less: the first landing takes whatever is there.
        const float cap = tracks > 0.0f ? tracks : 0.0f;
        return n < cap ? n : cap;
    }
    // (1 - 1/tracks)^n, evaluated in log space so large n stays well-conditioned.
    const float miss = exp(n * log(1.0f - 1.0f / tracks));
    const float used = tracks * (1.0f - miss);
    return used > 0.0f ? used : 0.0f;
}

// Via demand charged to ONE gcell edge, in tracks. `n_lo` / `n_hi` are the via landing counts of
// the two gcells the edge spans; `tracks` is the edge's capacity. Both models keep the "an edge
// sees the two cells it joins" structure; only the conversion differs.
GGR_RESOURCE_FN float viaEdgeDemand(float n_lo, float n_hi, float tracks,
                                    const RouteResourceModel& model) {
    if (model.via_mode == static_cast<int>(ViaResourceMode::LegacySurcharge)) {
        // Evaluated in double, as the historical expression was, so selecting the legacy model
        // reproduces an old result exactly rather than approximately.
        const double avg = 0.5 * (static_cast<double>(n_lo) + static_cast<double>(n_hi));
        return static_cast<float>(sqrt(avg) * static_cast<double>(model.legacy_scale));
    }
    return 0.5f * (viaOccupiedTracks(n_lo, tracks) + viaOccupiedTracks(n_hi, tracks));
}

// Amount by which one gcell edge is over capacity, in tracks (0 when it fits).
GGR_RESOURCE_FN float edgeOverflow(float wire_demand, float via_demand, float capacity) {
    const float excess = wire_demand + via_demand - capacity;
    return excess > 0.0f ? excess : 0.0f;
}

}  // namespace gr
