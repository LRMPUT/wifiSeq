//
// Created by janw on 23.08.2019.
//

#ifndef WIFISEQ_UTILS_HPP
#define WIFISEQ_UTILS_HPP

#include <vector>
#include "LocationWiFi.hpp"

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
static constexpr double errorSigma = 2;
static constexpr double distSigma = 2.0;
static constexpr double orientSigma = 0.8;

// parameter estimation
//static constexpr double probThresh = 0.075;

static constexpr double probThresh = 0.015;
static constexpr double minProb = 0.01;

static constexpr double probRatio = 2.0;

// prob by wknn
static constexpr double probScale = 0.2;
// prob by MoG
//static constexpr double probScale = 1.0/5.0;

static constexpr double probVisScale = 4;


static constexpr int wknnk = 6;


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
