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

#ifndef WIFISEQ_UTILS_HPP
#define WIFISEQ_UTILS_HPP

#include <vector>
#include "LocationWifi.hpp"

static constexpr int ssThreshold = -100;
static constexpr double sharedPercentThreshold = 0.6;

// in meters
static constexpr double mapGrid = 1.0;
static constexpr double mapMinX = 0.0;
static constexpr double mapMaxX = 130.0;
static constexpr double mapMinY = 0.0;
static constexpr double mapMaxY = 40.0;
// each sector is 45 deg
static constexpr int orientSectors = 1;
static constexpr double orientSectorLen = 2 * M_PI / orientSectors;

static constexpr int mapGridSizeX = ceil((mapMaxX - mapMinX) / mapGrid);
static constexpr int mapGridSizeY = ceil((mapMaxY - mapMinY) / mapGrid);

static constexpr double wifiSigma = 8.0;
static constexpr double errorWifiSigma = 2.0;
static constexpr double errorImageSigma = 2.0;
static constexpr double distSigma = 2.0;
static constexpr double orientSigma = 0.8;

// parameter estimation
//static constexpr double probThresh = 0.075;

static constexpr double probThresh = 0.015;
static constexpr double minProb = 0.01;

static constexpr double probRatio = 2.0;

// change probScale when changing this parameter
static constexpr int useWknn = false;
//static constexpr int useWknn = true;

// prob by wknn
//static constexpr double probScale = 1.0;
// prob by MoG
static constexpr double probScale = 0.2;

static constexpr double probVisScale = 4;

static constexpr int wknnk = 6;

static constexpr bool enableVpr = true;
//static constexpr bool enableVpr = false;
static constexpr double vprTimeThresh = 1.0;


class Utils {
public:


    static double toPiRange(double o);

    static double angDiff(double o1, double o2);

    static double meanOrient(const std::vector<double>::const_iterator &beg,
                             const std::vector<double>::const_iterator &end);

    static std::pair<int, int> mapCoordToGrid(double x, double y);

    static LocationXY mapGridToCoord(int x, int y);

    static double orientIdxToOrient(int oIdx);

    static int mapGridToVal(int x, int y, int o);

    static int orientToOrientIdx(double o);

    static void valToMapGrid(double val, int &xIdx, int &yIdx, int &oIdx);
};


#endif //WIFISEQ_UTILS_HPP
