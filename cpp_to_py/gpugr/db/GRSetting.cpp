#include "GRSetting.h"

namespace gr {

void GRSetting::reset() {
    deviceId = 0;

    routeXSize = 0;
    routeYSize = 0;
    csrnScale = 0;

    rrrIters = 0;
    rrrStallLimit = 2;
    rrrMinRelGain = 0.02;
    rrrVioEscalation = 2.0;

    viaResourceMode = 1;
    capacityDerate = 0.0;

    routeGuideFile = "";
}

GRSetting grSetting;

}  //   namespace gr