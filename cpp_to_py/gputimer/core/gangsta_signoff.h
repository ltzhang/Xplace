// GangSTA signoff-timer adapter for Xplace (Mode B differential comparison).
//
// A thin, torch-free wrapper over the GangSTA C API (gangsta/gangsta.h). It reads the STATIC design
// (Verilog netlist + early/late Liberty + SDC) ONCE, then on each comparison point accepts the
// placer's per-net interconnect parasitics in-memory (name-keyed, no SPEF file round-trip) and reports
// WNS/TNS for the early (hold) and late (setup) corners. This lets Xplace report GangSTA's timing
// alongside its own GPU timer under `--signoff_timer gangsta`, so the two engines can be compared on
// the same placement. Compiled only when the gangsta library is available (XPLACE_HAVE_GANGSTA); the
// header is always includable so call sites need no #ifdef.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gt {

class GangstaSignoff {
public:
    // Per-net interconnect parasitics, name-keyed to the Verilog netlist, in CSR form (mirrors the
    // gangsta_set_parasitics_inmem layout). net i owns grounded caps [cap_start[i], cap_start[i+1])
    // and resistors [res_start[i], res_start[i+1]). Caps are in ff, resistors in kohm. Node names are
    // pin names ("inst:pin" / top ports); internal Steiner nodes may use any per-net-unique name.
    // IMPORTANT: caps here are the WIRE caps only — GangSTA adds each sink's Liberty pin capacitance
    // itself, so including pin caps here would double-count the net load.
    struct Parasitics {
        std::vector<std::string> net_names;
        std::vector<float> net_total_cap;            // per net, informational (ff)
        std::vector<uint32_t> cap_start;             // size net_names.size()+1
        std::vector<std::string> cap_node;
        std::vector<float> cap_val;                  // ff
        std::vector<uint32_t> res_start;             // size net_names.size()+1
        std::vector<std::string> res_a, res_b;
        std::vector<float> res_val;                  // kohm
    };

    struct WnsTns {
        float wns_early = 0.0F, tns_early = 0.0F;   // min corner (hold)
        float wns_late = 0.0F, tns_late = 0.0F;     // max corner (setup)
        bool valid = false;                          // false => no constrained endpoints / not built
    };

    GangstaSignoff();
    ~GangstaSignoff();
    GangstaSignoff(const GangstaSignoff&) = delete;
    GangstaSignoff& operator=(const GangstaSignoff&) = delete;

    // True when the build was compiled against the gangsta library. When false, every other method is
    // a safe no-op (build() returns false with a diagnostic) so the caller degrades gracefully.
    static bool available();

    // Read the static design and build the timing graph. Returns false (see error()) on any failure.
    bool build(const std::string& verilog, const std::string& early_lib, const std::string& late_lib,
               const std::string& sdc);
    bool is_built() const { return built_; }

    // Inject the parasitics and re-time; report WNS/TNS for both corners. Returns {valid=false} if not
    // built or on an injection/report failure (with error() set).
    WnsTns report(const Parasitics& p);

    const std::string& error() const { return err_; }

private:
    void* timer_ = nullptr;  // GangstaTimer* held opaquely so the C API stays out of this header
    bool built_ = false;
    std::string err_;
};

}  // namespace gt
