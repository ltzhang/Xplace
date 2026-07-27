#pragma once

// The set of placement rows the legalizer may put a cell in.
//
// UNIFORM mode -- one row height across the whole core, which is the overwhelming majority of
// floorplans -- keeps the classic closed-form arithmetic. Every accessor below reproduces LITERALLY
// the expression it replaced at each call site (`floorDiv`, plain truncation, `ceil`, ...), so a
// single-height design legalizes exactly as it did before this grid existed.
//
// MIXED mode carries an explicit per-row (yl, height) table. That is what a floorplan interleaving
// two standard-cell row heights needs: a cell may only occupy a run of rows whose heights sum
// EXACTLY to its own height, so a 9-track cell can never be handed a 7-track row. A cell that
// matches no row at all is reported as such (`rows_spanned` < 0) and the caller must fail loudly
// rather than drop it somewhere plausible.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace dp {

namespace rowgrid {
inline int floorDivTol(float a, float b, float rtol = 1e-4f) { return static_cast<int>(std::floor((a + rtol * b) / b)); }
inline int ceilDivTol(float a, float b, float rtol = 1e-4f) { return static_cast<int>(std::ceil((a - rtol * b) / b)); }
}  // namespace rowgrid

class RowGrid {
public:
    RowGrid() = default;

    // Uniform grid: rows at yl + k * row_height.
    RowGrid(float yl, float yh, float row_height) : yl_(yl), yh_(yh), h_(row_height) {}

    // Explicit table. `row_yl` must be ascending and the rows non-overlapping (the producer
    // validates that); `row_h` is the matching per-row height.
    RowGrid(float yl, float yh, std::vector<float> row_yl, std::vector<float> row_h)
        : yl_(yl), yh_(yh), row_yl_(std::move(row_yl)), row_h_(std::move(row_h)) {
        if (row_yl_.size() != row_h_.size()) {  // defensive: an inconsistent table degrades to uniform
            row_yl_.clear();
            row_h_.clear();
        }
        if (row_yl_.empty()) return;
        h_ = *std::min_element(row_h_.begin(), row_h_.end());
        max_h_ = *std::max_element(row_h_.begin(), row_h_.end());
        uniform_ = (h_ == max_h_);
        if (uniform_) {  // a "table" that turns out uniform is just the uniform grid
            row_yl_.clear();
            row_h_.clear();
        }
    }

    bool uniform() const { return uniform_; }
    float min_row_height() const { return h_; }
    float max_row_height() const { return uniform_ ? h_ : max_h_; }

    // Row count as `floorDiv(yh - yl, row_height)` in uniform mode -- the blank-bin count the
    // greedy legalizer has always used.
    int num_rows() const {
        return uniform_ ? rowgrid::floorDivTol(yh_ - yl_, h_) : static_cast<int>(row_yl_.size());
    }
    // Row count as `ceilDiv(yh - yl, row_height)` in uniform mode -- the bin count the abacus
    // legalizer and the legality check have always used (they include a trailing partial row).
    int num_rows_incl_partial() const {
        return uniform_ ? rowgrid::ceilDivTol(yh_ - yl_, h_) : static_cast<int>(row_yl_.size());
    }

    float row_yl(int r) const { return uniform_ ? yl_ + r * h_ : row_yl_[static_cast<size_t>(r)]; }
    float row_height(int r) const { return uniform_ ? h_ : row_h_[static_cast<size_t>(r)]; }

    // Index of the row containing `y`, tolerant of float noise on an exact row boundary.
    // Uniform mode is exactly `floorDiv(y - yl, row_height)`; NOT clamped (callers clamp as before).
    int row_index_floor_tol(float y, float rtol = 1e-4f) const {
        if (uniform_) return rowgrid::floorDivTol(y - yl_, h_, rtol);
        return search_floor(y, rtol);
    }
    // Uniform mode is exactly the plain truncating `(y - yl) / row_height` some call sites use.
    int row_index_floor(float y) const {
        if (uniform_) return static_cast<int>((y - yl_) / h_);
        return search_floor(y, 0.0f);
    }
    // Index one past the last row strictly below `y`: uniform mode is `ceil((y - yl) / row_height)`.
    int row_index_ceil(float y, float rtol = 0.0f) const {
        if (uniform_) return rtol > 0 ? rowgrid::ceilDivTol(y - yl_, h_, rtol)
                                      : static_cast<int>(std::ceil((y - yl_) / h_));
        // first row whose lower edge is >= y (within tolerance)
        const float tol = rtol * h_;
        int n = static_cast<int>(row_yl_.size());
        int lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (row_yl_[static_cast<size_t>(mid)] + tol < y) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    // How many rows a cell of height `height` occupies when its bottom sits on row `r`.
    // Uniform mode: `ceilDiv(height, row_height)` -- unchanged, and never negative.
    // Mixed mode: the length of the contiguous run starting at `r` whose heights sum to exactly
    // `height`, or -1 when no such run exists (the cell does not fit that row).
    int rows_spanned(int r, float height) const {
        if (uniform_) return rowgrid::ceilDivTol(height, h_);
        if (r < 0 || r >= static_cast<int>(row_yl_.size())) return -1;
        const float tol = 1e-3f * h_;
        float sum = 0;
        for (size_t i = static_cast<size_t>(r); i < row_yl_.size(); ++i) {
            if (i > static_cast<size_t>(r) && std::abs(row_yl_[i] - (row_yl_[i - 1] + row_h_[i - 1])) > tol) break;
            sum += row_h_[i];
            if (std::abs(sum - height) <= tol) return static_cast<int>(i - static_cast<size_t>(r)) + 1;
            if (sum > height + tol) break;
        }
        return -1;
    }

    // Upper bound on rows_spanned() over all rows -- used to size per-cell scratch before the
    // candidate-row search. Uniform mode is `ceilDiv(height, row_height)`.
    int max_rows_spanned(float height) const {
        if (uniform_) return rowgrid::ceilDivTol(height, h_);
        int n = rowgrid::ceilDivTol(height, h_);
        return n < 1 ? 1 : n;
    }

    // True when a cell of `height` whose bottom is at `y` sits exactly on a row (or a run of rows).
    bool aligned(float y, float height) const {
        int r = row_index_floor_tol(y);
        if (r < 0 || r >= num_rows_incl_partial()) return false;
        if (std::abs(row_yl(r) - y) > 1e-3f * h_) return false;
        return rows_spanned(r, height) > 0;
    }

private:
    int search_floor(float y, float rtol) const {
        const float tol = rtol * h_;
        int n = static_cast<int>(row_yl_.size());
        // last row whose lower edge is <= y (+tol); -1 when y is below the first row
        int lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (row_yl_[static_cast<size_t>(mid)] <= y + tol) lo = mid + 1; else hi = mid;
        }
        return lo - 1;
    }

    float yl_ = 0;
    float yh_ = 0;
    float h_ = 1;      // uniform row height, or the minimum row height of the table
    float max_h_ = 1;  // maximum row height of the table
    bool uniform_ = true;
    std::vector<float> row_yl_;
    std::vector<float> row_h_;
};

}  // namespace dp
