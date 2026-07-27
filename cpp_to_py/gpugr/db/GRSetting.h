#pragma once
#include <string>

namespace gr {

class GRSetting {
public:
    // 1. SystemSetting
    int deviceId = 0;

    // 2. Gridgraph setting
    int routeXSize = 0;
    int routeYSize = 0;
    int csrnScale = 0;

    // 3. The number of Rip-up and Reroute iterations (if 0, only PR is invoked). This is a BUDGET,
    //    not a schedule: the loop keeps the best pass and stops when overflow stops improving, so
    //    raising it can no longer make the reported congestion worse.
    int rrrIters = 0;
    // Consecutive non-improving passes that end the rip-up loop.
    int rrrStallLimit = 2;
    // Relative overflow reduction a pass must reach to count as improvement.
    double rrrMinRelGain = 0.02;
    // Per-iteration escalation of the violation cost past its nominal value. > 1 makes congestion
    // cost dominate wirelength as passes accumulate; 1 reproduces a flat nominal cost.
    double rrrVioEscalation = 2.0;

    // 4. Routing resource model (see gpugr/gr/RouteResource.h).
    //    viaResourceMode: 0 = legacy sqrt steering surcharge, 1 = track-occupancy (default).
    int viaResourceMode = 1;
    //    Fraction of each layer's raw track count reserved for pin access / local nets / NDR.
    double capacityDerate = 0.0;

    std::string routeGuideFile = "";

    void reset();
};

extern GRSetting grSetting;
}  //   namespace gr
