// GangSTA signoff-timer adapter implementation. See gangsta_signoff.h.
//
// Two compilations: with XPLACE_HAVE_GANGSTA the methods drive the real gangsta C API; without it they
// are safe no-ops (available()==false) so a checkout lacking the gangsta library still builds and the
// `--signoff_timer gangsta` path degrades to a clear diagnostic rather than a link error.
#include "gputimer/core/gangsta_signoff.h"

#ifdef XPLACE_HAVE_GANGSTA
#include <vector>

#include "gangsta/gangsta.h"
#endif

namespace gt {

#ifdef XPLACE_HAVE_GANGSTA

bool GangstaSignoff::available() { return true; }

GangstaSignoff::GangstaSignoff() : timer_(gangsta_new()) {}

GangstaSignoff::~GangstaSignoff() {
    if (timer_) gangsta_free(static_cast<GangstaTimer*>(timer_));
}

bool GangstaSignoff::build(const std::string& verilog, const std::string& early_lib,
                           const std::string& late_lib, const std::string& sdc) {
    built_ = false;
    err_.clear();
    auto* t = static_cast<GangstaTimer*>(timer_);
    if (!t) {
        err_ = "gangsta_new() failed (out of memory)";
        return false;
    }
    // Canonical order: liberties -> netlist -> sdc -> build. Parasitics are injected later via report().
    if (!gangsta_read_liberty(t, GANGSTA_EARLY, early_lib.c_str())) {
        err_ = "failed to read early Liberty: " + early_lib;
        return false;
    }
    if (!gangsta_read_liberty(t, GANGSTA_LATE, late_lib.c_str())) {
        err_ = "failed to read late Liberty: " + late_lib;
        return false;
    }
    if (!gangsta_read_netlist(t, verilog.c_str(), nullptr)) {
        err_ = "failed to read Verilog netlist: " + verilog;
        return false;
    }
    if (!sdc.empty() && !gangsta_read_sdc(t, sdc.c_str())) {
        err_ = "failed to read SDC: " + sdc;
        return false;
    }
    gangsta_build_graph(t);
    if (!gangsta_is_built(t)) {
        err_ = std::string("gangsta build_graph failed: ") + gangsta_last_error(t);
        return false;
    }
    gangsta_update_delay(t, false);  // initial timing (lumped model until parasitics are injected)
    built_ = true;
    return true;
}

GangstaSignoff::WnsTns GangstaSignoff::report(const Parasitics& p) {
    WnsTns out;
    if (!built_) {
        err_ = "report() called before a successful build()";
        return out;
    }
    auto* t = static_cast<GangstaTimer*>(timer_);

    // Materialize the borrowed C-string pointer arrays from the owning std::string vectors.
    const std::size_t nn = p.net_names.size();
    std::vector<const char*> net_names(nn), cap_node(p.cap_node.size()), res_a(p.res_a.size()),
        res_b(p.res_b.size());
    for (std::size_t i = 0; i < nn; ++i) net_names[i] = p.net_names[i].c_str();
    for (std::size_t i = 0; i < p.cap_node.size(); ++i) cap_node[i] = p.cap_node[i].c_str();
    for (std::size_t i = 0; i < p.res_a.size(); ++i) res_a[i] = p.res_a[i].c_str();
    for (std::size_t i = 0; i < p.res_b.size(); ++i) res_b[i] = p.res_b[i].c_str();

    const bool ok = gangsta_set_parasitics_inmem(
        t, nn, net_names.data(), p.net_total_cap.empty() ? nullptr : p.net_total_cap.data(),
        p.cap_start.data(), cap_node.empty() ? nullptr : cap_node.data(),
        p.cap_val.empty() ? nullptr : p.cap_val.data(), p.res_start.data(),
        res_a.empty() ? nullptr : res_a.data(), res_b.empty() ? nullptr : res_b.data(),
        p.res_val.empty() ? nullptr : p.res_val.data());
    if (!ok) {
        err_ = std::string("set_parasitics_inmem failed: ") + gangsta_last_error(t);
        return out;
    }
    gangsta_update_delay(t, false);  // re-time under the injected RC

    // Early = min corner (hold), late = max corner (setup).
    const bool e_ok = gangsta_report_wns_tns(t, GANGSTA_MIN, &out.wns_early, &out.tns_early, false);
    const bool l_ok = gangsta_report_wns_tns(t, GANGSTA_MAX, &out.wns_late, &out.tns_late, false);
    out.valid = e_ok || l_ok;
    if (!out.valid) err_ = "gangsta reported no constrained endpoints for either corner";
    return out;
}

#else  // !XPLACE_HAVE_GANGSTA — safe stubs so the module builds and runs without the gangsta library.

bool GangstaSignoff::available() { return false; }
GangstaSignoff::GangstaSignoff() = default;
GangstaSignoff::~GangstaSignoff() = default;
bool GangstaSignoff::build(const std::string&, const std::string&, const std::string&,
                           const std::string&) {
    err_ = "GangSTA signoff timer not available (Xplace built without the gangsta library)";
    return false;
}
GangstaSignoff::WnsTns GangstaSignoff::report(const Parasitics&) {
    err_ = "GangSTA signoff timer not available (Xplace built without the gangsta library)";
    return WnsTns{};
}

#endif

}  // namespace gt
