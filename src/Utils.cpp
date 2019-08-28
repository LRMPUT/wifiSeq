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

double Utils::meanOrient(const std::vector<double> &orients) {
    double orientOffsetSin = 0;
    double orientOffsetCos = 0;
    for(int i = 0; i < orients.size(); ++i){
        orientOffsetSin += sin(orients[i]);
        orientOffsetCos += cos(orients[i]);
    }
    // mean using sin and cos
    orientOffsetSin /= orients.size();
    orientOffsetCos /= orients.size();

    // if not spread uniformly on the circle
    if(orientOffsetSin*orientOffsetSin + orientOffsetCos*orientOffsetCos > 0.01) {
        return atan2(orientOffsetSin, orientOffsetCos);
    }
    // return middle value
    else {
        return orients[orients.size() / 2];
    }
}
