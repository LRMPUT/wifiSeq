//
// Created by janw on 23.08.2019.
//

#include <cmath>
#include <algorithm>

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

double Utils::angDiff(double o1, double o2) {
    if(std::abs(o1 - o2) <= M_PI ){
        return o1 - o2;
    }
    else if(o1 - o2 < - M_PI){
        return o1 - o2 + 2 * M_PI;
    }
    else {
        return o1 - o2 - 2 * M_PI;
    }
}
