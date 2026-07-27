#pragma once

#include "common/common.h"
#include "gpudp/db/dp_torch.h"
#include "gpudp/lg/row_grid.h"

namespace dp {

inline int floorDiv(float a, float b, float rtol = 1e-4) { return std::floor((a + rtol * b) / b); }

inline int ceilDiv(float a, float b, float rtol = 1e-4) { return std::ceil((a - rtol * b) / b); }

inline int roundDiv(float a, float b) { return std::round(a / b); }

template <typename T>
struct Space {
    T xl;
    T xh;
};

struct RowMapIndex {
    int row_id;
    int sub_id;
};

struct BinMapIndex {
    int bin_id;
    int sub_id;
};

struct Box {
    float xl;
    float yl;
    float xh;
    float yh;
    Box() {
        xl = std::numeric_limits<float>::max();
        yl = std::numeric_limits<float>::max();
        xh = std::numeric_limits<float>::lowest();
        yh = std::numeric_limits<float>::lowest();
    }
    Box(float xxl, float yyl, float xxh, float yyh) : xl(xxl), yl(yyl), xh(xxh), yh(yyh) {}

    float center_x() const { return (xl + xh) / 2; }
    float center_y() const { return (yl + yh) / 2; }
    float width() const { return (xh - xl); }
    float height() const { return (yh - yl); }
    float area() const { return (xh - xl) * (yh - yl); }
};

class LegalizationData {
public:
    LegalizationData() {}
    LegalizationData(DPTorchRawDB& at_db)
        : x(at_db.x.data_ptr<float>()),
          y(at_db.y.data_ptr<float>()),
          init_x(at_db.init_x.data_ptr<float>()),
          init_y(at_db.init_y.data_ptr<float>()),
          node_size_x(at_db.node_size_x.data_ptr<float>()),
          node_size_y(at_db.node_size_y.data_ptr<float>()),
          pin_offset_x(at_db.pin_offset_x.data_ptr<float>()),
          pin_offset_y(at_db.pin_offset_y.data_ptr<float>()),
          flat_node2pin_start_map(at_db.flat_node2pin_start_map.data_ptr<int>()),
          flat_node2pin_map(at_db.flat_node2pin_map.data_ptr<int>()),
          pin2node_map(at_db.pin2node_map.data_ptr<int>()),
          flat_net2pin_start_map(at_db.flat_net2pin_start_map.data_ptr<int>()),
          flat_net2pin_map(at_db.flat_net2pin_map.data_ptr<int>()),
          pin2net_map(at_db.pin2net_map.data_ptr<int>()),
          flat_region_boxes_start(at_db.flat_region_boxes_start.data_ptr<int>()),
          flat_region_boxes(at_db.flat_region_boxes.data_ptr<float>()),
          node2fence_region_map(at_db.node2fence_region_map.data_ptr<int>()),
          net_mask(at_db.net_mask.data_ptr<bool>()),
          node_weight(at_db.node_weight.data_ptr<float>()),
          is_macro(at_db.is_macro.data_ptr<bool>()),
          xl(at_db.xl),
          xh(at_db.xh),
          yl(at_db.yl),
          yh(at_db.yh),
          row_height(at_db.row_height),
          site_width(at_db.site_width),
          rows(at_db.rows()),
          num_sites_x(at_db.num_sites_x),
          num_sites_y(at_db.num_sites_y),
          num_threads(at_db.num_threads),
          num_nodes(at_db.num_nodes),
          num_movable_nodes(at_db.num_movable_nodes),
          num_conn_movable_nodes(at_db.num_conn_movable_nodes),
          num_nets(at_db.num_nets),
          num_pins(at_db.num_pins),
          num_regions(at_db.num_regions) {}

public:
    float* x;             // new pos x, need to be checked their legality
    float* y;             // new pos y, need to be checked their legality
    const float* init_x;  // original pos x
    const float* init_y;  // original pos y
    const float* node_size_x;
    const float* node_size_y;

    const float* node_weight;
    const bool* is_macro;

    const float* pin_offset_x;
    const float* pin_offset_y;

    const int* flat_node2pin_start_map;
    const int* flat_node2pin_map;
    const int* pin2node_map;

    const int* flat_net2pin_start_map;
    const int* flat_net2pin_map;
    const int* pin2net_map;

    const int* flat_region_boxes_start;
    const float* flat_region_boxes;
    const int* node2fence_region_map;

    const bool* net_mask;

    /* chip info */
    float xl;
    float yl;
    float xh;
    float yh;

    /* row info */
    int num_sites_x;
    int num_sites_y;
    float row_height;  // uniform row height, or the SHORTEST row height on a mixed-height core
    float site_width;
    /* The authoritative row model. On a single-height core it reproduces the closed-form
     * `row_height` arithmetic exactly; on a mixed-height core it carries every row's own height. */
    RowGrid rows;

    int num_nets;
    int num_conn_movable_nodes;
    int num_movable_nodes;
    int num_nodes;
    int num_pins;
    int num_regions;

    int num_threads;

    int num_bins_x;
    int num_bins_y;
    float bin_size_x;
    float bin_size_y;

public:
    void set_num_bins(int num_bins_x_, int num_bins_y_) {
        num_bins_x = num_bins_x_;
        num_bins_y = num_bins_y_;
        bin_size_x = (xh - xl) / num_bins_x_;
        bin_size_y = (yh - yl) / num_bins_y_;
    }
    inline bool is_mov_macro(int node_id) const {
        return (node_id < num_movable_nodes && is_macro[node_id]);
    }

    // Snap `y` down to the lower edge of a row that a cell of `height` can legally occupy.
    // On a mixed-height core that means skipping down past rows whose height the cell cannot tile.
    inline float align2row(float y, float height) const {
        float yy = std::max(std::min(y, yh - height), yl);
        if (rows.uniform()) {
            return floorDiv(yy - yl, row_height) * row_height + yl;
        }
        int r = rows.row_index_floor_tol(yy);
        r = std::min(std::max(r, 0), rows.num_rows() - 1);
        for (int i = r; i >= 0; --i) {
            if (rows.rows_spanned(i, height) > 0) return rows.row_yl(i);
        }
        for (int i = r + 1; i < rows.num_rows(); ++i) {
            if (rows.rows_spanned(i, height) > 0) return rows.row_yl(i);
        }
        return rows.row_yl(r);  // no row fits this height at all -- the legality check reports it
    }

    inline float align2site(float x, float width) const {
        float xx = std::max(std::min(x, xh - width), xl);
        xx = floorDiv(xx - xl, site_width) * site_width + xl;
        return xx;
    }
};

}  // namespace dp