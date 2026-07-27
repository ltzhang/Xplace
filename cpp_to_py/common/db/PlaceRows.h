#pragma once

// Physical placement-row table.
//
// The placer's classic floorplan model is a single uniform grid: one site width, one row height,
// rows at coreLY + k * rowHeight. That model cannot describe a floorplan whose core interleaves
// standard-cell rows of TWO different heights -- e.g. a PDK that mixes 7-track and 9-track cells,
// written in the DEF as alternating ROW records of two different SITEs. Collapsing such a core onto
// the shorter of the two heights silently misplaces every taller cell: it lands on a y that is not
// the lower edge of any real row, so its power rails do not line up and the legalizer's own overlap
// bookkeeping disagrees with the physical rows.
//
// This header derives the real row table (one entry per physical row, each with its OWN height) from
// the raw DEF ROW records, and validates that the records describe a consistent tiling. It is kept
// dependency-free -- no logger, no database types -- so the derivation is unit-testable on its own.

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace db {

// One physical placement row: the strip [y, y + h) that a standard cell of height exactly h
// (or a contiguous run of rows summing to the cell height) may occupy.
struct PlaceRow {
    int y = 0;         // bottom edge
    int h = 0;         // height
    int orient = 0;    // DEF row orientation code (0:N 1:W 2:S 3:E 4:FN 5:FW 6:FS 7:FE, -1:NONE)
    bool flip = false; // DEF row flip flag

    PlaceRow() = default;
    PlaceRow(int y_, int h_, int orient_ = 0, bool flip_ = false)
        : y(y_), h(h_), orient(orient_), flip(flip_) {}
};

enum class PlaceRowStatus {
    kUniform,   // every row has the same height -- the classic uniform grid; nothing changes
    kMixed,     // two or more distinct row heights, and the table validated
    kRejected,  // the ROW records do not describe a consistent tiling; the table must not be used
};

struct PlaceRowTable {
    std::vector<PlaceRow> rows;  // ascending in y, non-overlapping
    PlaceRowStatus status = PlaceRowStatus::kUniform;
    std::string message;         // why the table was rejected (empty otherwise)

    bool mixed() const { return status == PlaceRowStatus::kMixed; }
    bool rejected() const { return status == PlaceRowStatus::kRejected; }
    bool empty() const { return rows.empty(); }
    // Lowest row edge / highest row edge. Only meaningful when !empty().
    int yl() const { return rows.front().y; }
    int yh() const {
        int top = rows.front().y;
        for (const PlaceRow& r : rows) top = std::max(top, r.y + r.h);
        return top;
    }
};

namespace detail {

// A DEF may describe the same physical strip more than once. A "hybrid" SITE -- one whose LEF
// definition carries a ROWPATTERN naming two shorter sites -- is commonly written out BOTH as the
// tall pattern row and as the shorter rows it expands into, all sharing a lower edge. Keeping the
// SHORTEST record at each y therefore recovers the finest-grained (i.e. the real) placement rows;
// the coarser duplicates are then re-checked below as exact multi-row spans.
inline std::vector<PlaceRow> shortestPerLowerEdge(const std::vector<PlaceRow>& records) {
    std::map<int, PlaceRow> by_y;
    for (const PlaceRow& r : records) {
        auto it = by_y.find(r.y);
        if (it == by_y.end() || r.h < it->second.h) by_y[r.y] = r;
    }
    std::vector<PlaceRow> out;
    out.reserve(by_y.size());
    for (const auto& kv : by_y) out.push_back(kv.second);
    return out;
}

// Number of table rows, starting at `first`, whose heights sum to exactly `height`.
// Returns 0 when no such contiguous run exists (the caller treats that as "does not fit").
inline int spanRows(const std::vector<PlaceRow>& rows, size_t first, int height) {
    int sum = 0;
    for (size_t i = first; i < rows.size(); ++i) {
        if (i > first && rows[i].y != rows[i - 1].y + rows[i - 1].h) break;  // gap: not contiguous
        sum += rows[i].h;
        if (sum == height) return static_cast<int>(i - first) + 1;
        if (sum > height) break;
    }
    return 0;
}

}  // namespace detail

// Derive the placement-row table from the raw DEF ROW records.
//
// The table is REJECTED (and the caller must fall back to the uniform grid) whenever the records are
// not a consistent tiling: a non-positive height, rows that overlap in y, or an original record that
// is not exactly covered by a contiguous run of table rows. Rejecting loudly is the point -- a
// half-understood row structure must never be turned into a placement.
inline PlaceRowTable buildPlaceRowTable(const std::vector<PlaceRow>& records) {
    PlaceRowTable table;
    if (records.empty()) return table;  // kUniform + empty: caller keeps its existing model

    for (const PlaceRow& r : records) {
        if (r.h <= 0) {
            table.status = PlaceRowStatus::kRejected;
            table.message = "row at y=" + std::to_string(r.y) + " has non-positive height " + std::to_string(r.h);
            return table;
        }
    }

    table.rows = detail::shortestPerLowerEdge(records);
    std::sort(table.rows.begin(), table.rows.end(), [](const PlaceRow& a, const PlaceRow& b) { return a.y < b.y; });

    // Rows may leave gaps (a carved-out macro region is legal) but must never overlap.
    for (size_t i = 1; i < table.rows.size(); ++i) {
        if (table.rows[i - 1].y + table.rows[i - 1].h > table.rows[i].y) {
            table.status = PlaceRowStatus::kRejected;
            table.message = "row at y=" + std::to_string(table.rows[i - 1].y) + " height " +
                            std::to_string(table.rows[i - 1].h) + " overlaps the row at y=" +
                            std::to_string(table.rows[i].y);
            table.rows.clear();
            return table;
        }
    }

    // Every original record must be exactly reproduced by a contiguous run of table rows. This is
    // what makes the "shortest per lower edge" collapse safe: a taller record survives only if it is
    // genuinely the concatenation of the shorter rows it shadows.
    std::map<int, size_t> index_of_y;
    for (size_t i = 0; i < table.rows.size(); ++i) index_of_y[table.rows[i].y] = i;
    for (const PlaceRow& r : records) {
        auto it = index_of_y.find(r.y);
        if (it == index_of_y.end() || detail::spanRows(table.rows, it->second, r.h) == 0) {
            table.status = PlaceRowStatus::kRejected;
            table.message = "row record at y=" + std::to_string(r.y) + " height " + std::to_string(r.h) +
                            " is not an exact run of placement rows";
            table.rows.clear();
            return table;
        }
    }

    const int h0 = table.rows.front().h;
    for (const PlaceRow& r : table.rows) {
        if (r.h != h0) {
            table.status = PlaceRowStatus::kMixed;
            return table;
        }
    }
    table.status = PlaceRowStatus::kUniform;
    return table;
}

}  // namespace db
