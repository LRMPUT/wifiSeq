//
// Created by janw on 23.08.2019.
//

#include <cmath>

#include "Utils.hpp"

double Utils::toPiRange(double o) {
    // (- 2 * M_PI, 2 * M_PI)
    o = fmod(o, 2 * M_PI);
    // [0, 2 * M_PI)
    o = fmod(o + 2 * M_PI, 2 * M_PI);
    if (o > M_PI){
        return o - 2 * M_PI;
    }
    else {
        return o;
    }
}
