#pragma once
// Rip-up-and-reroute loop control: violation-cost schedule, keep-best, and termination
// (WiseSyn R2-19).
//
// The historical loop ran a fixed number of passes, kept whatever the LAST pass produced, and had
// no termination test at all (its `break` was commented out). Two consequences, both observed:
//
//   * More effort made congestion WORSE. The violation-cost ramp was normalized by the iteration
//     BUDGET -- `0.1 + 0.9/(limit-1) * iter` -- so asking for more passes made every early pass
//     softer on congestion than it would have been with a small budget. With a budget of 1 the
//     single maze pass ran at full violation cost; with a budget of 8 the first maze pass ran at
//     0.21 of it, ripping up routed nets and re-routing them under a wirelength-dominated
//     objective. The schedule below is indexed by the iteration alone, so a larger budget never
//     softens an earlier pass, and it keeps escalating past the nominal cost instead of asymptoting
//     to it.
//
//   * A pass that made things worse was still what got emitted. `RrrController` keeps the best
//     result seen and reports THAT, which turns "more effort must not increase overflow" from a
//     tuning hope into a structural property of the loop.
//
// Scoring is lexicographic on (unrouted nets, edge overflow): a solution that resolves more nets
// always wins, so the loop can never buy a lower congestion number by leaving nets unrouted.
//
// Deliberately dependency-free (no CUDA, no torch, no xplace types) so the policy is unit-testable
// without a GPU.

namespace gr {

// One iteration's routability, ordered better-first by `betterThan`.
struct RrrScore {
    long long unrouted_nets = 0;   // nets the router could not resolve at all -- dominates
    double edge_overflow = 0.0;    // total over-capacity demand summed over gcell edges, in tracks

    bool betterThan(const RrrScore& other) const {
        if (unrouted_nets != other.unrouted_nets) return unrouted_nets < other.unrouted_nets;
        return edge_overflow < other.edge_overflow;
    }
    bool clean() const { return unrouted_nets == 0 && !(edge_overflow > 0.0); }
};

// Why the loop stopped. Reported so a route that stops short says which limit it hit.
enum class RrrStop : int {
    Running = 0,
    Clean,        // no unrouted nets and no edge overflow left -- converged
    Stalled,      // consecutive passes stopped improving the score
    Budget,       // the iteration budget ran out with overflow left
};

class RrrController {
public:
    // `max_iters` counts every pass including the initial pattern route (so it is >= 1).
    // `stall_limit` is how many consecutive non-improving passes end the loop.
    // `min_rel_gain` is the relative overflow reduction a pass must achieve to count as improving;
    // a pass that shaves a fraction of a track off a large overflow is not progress.
    RrrController(int max_iters, int stall_limit, double min_rel_gain)
        : max_iters_(max_iters < 1 ? 1 : max_iters),
          stall_limit_(stall_limit < 1 ? 1 : stall_limit),
          min_rel_gain_(min_rel_gain > 0.0 ? min_rel_gain : 0.0) {}

    // Violation-cost scale for iteration `iter`. Independent of the iteration budget by
    // construction: iteration 0 (the pattern route) runs soft so the first solution is built for
    // wirelength, iteration 1 runs at the nominal cost, and every later pass escalates
    // geometrically so violation cost eventually dominates wirelength.
    static double vioCostScale(int iter, double pattern_discount, double escalation) {
        if (iter <= 0) return pattern_discount;
        const double step = escalation > 1.0 ? escalation : 1.0;
        double scale = 1.0;
        for (int i = 1; i < iter; ++i) {
            scale *= step;
            if (scale > kMaxVioScale) return kMaxVioScale;  // bounded: cost feeds an int cost grid
        }
        return scale;
    }

    // The schedule this replaces, kept only so a test can pin the defect it had: normalizing by the
    // budget means growing the budget SHRINKS the scale of every pass but the last.
    static double legacyVioCostScale(int iter, int iter_limit, double pattern_discount) {
        if (iter <= 0) return pattern_discount;
        if (iter_limit <= 1) return pattern_discount;
        return pattern_discount + (1.0 - pattern_discount) / (iter_limit - 1) * iter;
    }

    // Record the result of iteration `iter`. Returns true when it is the new best and the caller
    // must therefore snapshot the routes.
    bool record(int iter, const RrrScore& score) {
        ++iters_run_;
        bool is_best = false;
        if (!have_best_ || score.betterThan(best_)) {
            const bool real_gain = !have_best_ || isRealGain(best_, score);
            best_ = score;
            best_iter_ = iter;
            have_best_ = true;
            is_best = true;
            if (real_gain) {
                stall_ = 0;
            } else {
                ++stall_;
            }
        } else {
            ++stall_;
        }
        if (best_.clean()) {
            stop_ = RrrStop::Clean;
        } else if (stall_ >= stall_limit_) {
            stop_ = RrrStop::Stalled;
        } else if (iters_run_ >= max_iters_) {
            stop_ = RrrStop::Budget;
        }
        return is_best;
    }

    bool shouldStop() const { return stop_ != RrrStop::Running; }
    RrrStop stopReason() const { return stop_; }
    const RrrScore& best() const { return best_; }
    int bestIter() const { return best_iter_; }
    int itersRun() const { return iters_run_; }
    bool haveBest() const { return have_best_; }

    static const char* stopText(RrrStop s) {
        switch (s) {
            case RrrStop::Clean: return "congestion resolved";
            case RrrStop::Stalled: return "stopped improving";
            case RrrStop::Budget: return "iteration budget exhausted";
            default: return "running";
        }
    }

    static constexpr double kMaxVioScale = 1e6;

private:
    bool isRealGain(const RrrScore& prev, const RrrScore& cur) const {
        if (cur.unrouted_nets < prev.unrouted_nets) return true;
        if (!(prev.edge_overflow > 0.0)) return false;
        return (prev.edge_overflow - cur.edge_overflow) >= min_rel_gain_ * prev.edge_overflow;
    }

    int max_iters_;
    int stall_limit_;
    double min_rel_gain_;
    RrrScore best_{};
    int best_iter_ = -1;
    int iters_run_ = 0;
    int stall_ = 0;
    bool have_best_ = false;
    RrrStop stop_ = RrrStop::Running;
};

}  // namespace gr
