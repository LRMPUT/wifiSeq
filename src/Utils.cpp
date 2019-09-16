/*
	Copyright (c) 2019,	Mobile Robots Laboratory:
	-Jan Wietrzykowski (jan.wietrzykowski@put.poznan.pl).
	Poznan University of Technology
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification,
	are permitted provided that the following conditions are met:

	1. Redistributions of source code must retain the above copyright notice,
	this list of conditions and the following disclaimer.

	2. Redistributions in binary form must reproduce the above copyright notice,
	this list of conditions and the following disclaimer in the documentation
	and/or other materials provided with the distribution.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
	THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
	ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
	FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
	DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
	AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
	OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cmath>
#include <algorithm>

#include "Utils.hpp"
#include "LocationWifi.hpp"

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
