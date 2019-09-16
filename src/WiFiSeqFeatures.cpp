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

#include <boost/serialization/vector.hpp>

#include "pgm/Pgm.h"
#include "Utils.hpp"
#include "WiFiSeqFeatures.hpp"

LocFeature::LocFeature(int iid,
                       int iparamNum,
                       const std::vector <std::shared_ptr<RandVar>> &irandVarsOrdered,
                       const std::vector<int> &iobsNums)
        : Feature(iid,
                  iparamNum,
                  irandVarsOrdered,
                  iobsNums)
{

}

double LocFeature::comp(const std::vector<double> &vals, const std::vector<double> &obsVec) {
    int loc = (int)round(vals[0]);
    // probabilities are the same for all orientations
    loc /= orientSectors;
    
    double ret = log(obsVec[loc]);
    ret = std::max(ret, -20.0);

    return ret;
}

double LocFeature::compParam(const std::vector<double> &vals,
                             const std::vector<double> &params,
                             const std::vector<double> &obsVec)
{
    return params[paramNum()]*comp(vals, obsVec);
}


MoveFeature::MoveFeature(int iid,
                         int iparamNum,
                         const std::vector<std::shared_ptr<RandVar>> &irandVarsOrdered,
                         const std::vector<int> &iobsNums,
                         int imapSize,
                         double isigmaDist,
                         std::shared_ptr<Graph> igraph)
        : Feature(iid,
                 iparamNum,
                 irandVarsOrdered,
                 iobsNums),
          mapSize(imapSize),
          sigmaDist(isigmaDist),
          graph(igraph)
{

}

double MoveFeature::comp(const std::vector<double> &vals, const std::vector<double> &obsVec) {
    int loc1 = (int)round(vals[0]);
    int loc2 = (int)round(vals[1]);
    
    double distStep = obsVec[0];

    double dist = graph->getDist(loc1, loc2);
    double error = dist - distStep;

    double ret = -error * error / (sigmaDist * sigmaDist);
    
    ret = std::max(ret, -20.0);

    return ret;
}

double MoveFeature::compParam(const std::vector<double> &vals,
                              const std::vector<double> &params,
                              const std::vector<double> &obsVec)
{
    return params[paramNum()]*comp(vals, obsVec);
}



OrientMoveFeature::OrientMoveFeature(int iid,
                                     int iparamNum,
                                     const std::vector<std::shared_ptr<RandVar>> &irandVarsOrdered,
                                     const std::vector<int> &iobsNums,
                                     int imapSize,
                                     double isigmaOrient)
        : Feature(iid,
                  iparamNum,
                  irandVarsOrdered,
                  iobsNums),
          mapSize(imapSize),
          sigmaOrient(isigmaOrient)
{

}

double OrientMoveFeature::comp(const std::vector<double> &vals, const std::vector<double> &obsVec) {
    static constexpr double thresh = M_PI / 2.0;
//    static constexpr double alpha =

    int loc1 = (int)round(vals[0]);
    int loc2 = (int)round(vals[1]);

    double orientMeas = obsVec[0];

    double x1 = obsVec[1 + loc1];
    double y1 = obsVec[1 + mapSize + loc1];
    double o1 = obsVec[1 + 2 * mapSize + loc1];
    double x2 = obsVec[1 + loc2];
    double y2 = obsVec[1 + mapSize + loc2];
    double o2 = obsVec[1 + 2 * mapSize + loc2];

    double dx = x2 - x1;
    double dy = y2 - y1;

    double error = 0.0;
    // if there is some movement
    if(sqrt(dx*dx + dy*dy) > 0.5) {
        double orient = atan2(dy, dx);
        error = Utils::angDiff(orient, orientMeas);
    }

    // berHu loss function
//    double ret = 0.0;
//    if(std::abs(error) < thresh){
//        ret = -std::abs(error);
////        ret = 0;
//    }
//    else{
//        ret = -(error * error + thresh * thresh) / (2 * thresh);
////        double errorThresh = std::abs(error) - thresh;
////        ret = -errorThresh * errorThresh;
//    }

//    double ret = -error * error;

    double ret = -exp(std::abs(error) - thresh);

    ret /= (sigmaOrient * sigmaOrient);

    ret = std::max(ret, -20.0);

    return ret;
}

double OrientMoveFeature::compParam(const std::vector<double> &vals,
                                    const std::vector<double> &params,
                                    const std::vector<double> &obsVec)
{
    return params[paramNum()]*comp(vals, obsVec);
}
