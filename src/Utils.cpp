//
// Created by janw on 23.08.2019.
//

#include <cmath>
#include <algorithm>

#include "Utils.hpp"
#include "LocationWiFi.hpp"

std::pair<int, int> Utils::mapCoordToGrid(double x, double y){
    return std::make_pair((x - mapMinX)/mapGrid, (y - mapMinY)/mapGrid);
};


LocationXY Utils::mapGridToCoord(int x, int y){
    return LocationXY(mapMinX + x * mapGrid, mapMinY + y * mapGrid, -1);
};

double Utils::orientIdxToOrient(int oIdx){
    return orientSectorLen * oIdx;
}

int Utils::orientToOrientIdx(double o){
    int oIdx = (int)((o + 2 * M_PI + orientSectorLen / 2.0) / orientSectorLen);
    oIdx = ((oIdx % orientSectors) + orientSectors) % orientSectors;
    return oIdx;
}

int Utils::mapGridToVal(int x, int y, int o){
    return o + orientSectors * x + orientSectors * mapGridSizeX * y;
}

void Utils::valToMapGrid(double val, int &xIdx, int &yIdx, int &oIdx){
    int valInt = (int)round(val);
    oIdx = valInt % orientSectors;
    valInt /= orientSectors;

    xIdx = valInt % mapGridSizeX;
    valInt /= mapGridSizeX;

    yIdx = valInt;
}

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

double Utils::meanOrient(const std::vector<double>::const_iterator &beg,
                         const std::vector<double>::const_iterator &end) {
    double orientOffsetSin = 0;
    double orientOffsetCos = 0;
    int cnt = 0;
    for(auto it = beg; it != end; ++it){
        orientOffsetSin += sin(*it);
        orientOffsetCos += cos(*it);
        ++cnt;
    }
    // mean using sin and cos
    orientOffsetSin /= cnt;
    orientOffsetCos /= cnt;

    // if not spread uniformly on the circle
    if(orientOffsetSin*orientOffsetSin + orientOffsetCos*orientOffsetCos > 0.01) {
        return atan2(orientOffsetSin, orientOffsetCos);
    }
    // return middle value
    else {
        return *(beg + cnt / 2);
    }
}
