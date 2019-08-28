//
// Created by jachu on 15.05.18.
//

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
//    ret = std::max(ret, -100.0);
//    if(std::isnan(ret) || std::isinf(ret)){
//        ret = 0;
//    }
    return ret;
}

double LocFeature::compParam(const std::vector<double> &vals,
                             const std::vector<double> &params,
                             const std::vector<double> &obsVec)
{
    return params[paramNum()]*comp(vals, obsVec);
}


OrientFeature::OrientFeature(int iid,
                                     int iparamNum,
                                     const std::vector<std::shared_ptr<RandVar>> &irandVarsOrdered,
                                     const std::vector<int> &iobsNums,
                                     double isigmaOrient)
        : Feature(iid,
                  iparamNum,
                  irandVarsOrdered,
                  iobsNums),
          sigmaOrient(isigmaOrient)
{

}

double OrientFeature::comp(const std::vector<double> &vals, const std::vector<double> &obsVec) {
    int loc = (int)round(vals[0]);

    double orientMeas = obsVec[0];
    double orient = obsVec[1 + loc];

    double error = Utils::angDiff(orientMeas, orient);

//    double ret = exp(-error*error / (sigmaDist*sigmaDist));
    double ret = -error * error / (sigmaOrient * sigmaOrient);

//    ret = std::max(ret, -20.0);
//    if(std::isnan(ret) || std::isinf(ret)){
//        ret = 0;
//    }
    return ret;
}

double OrientFeature::compParam(const std::vector<double> &vals,
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
                         double isigmaDist)
        : Feature(iid,
                 iparamNum,
                 irandVarsOrdered,
                 iobsNums),
          mapSize(imapSize),
          sigmaDist(isigmaDist)
{

}

double MoveFeature::comp(const std::vector<double> &vals, const std::vector<double> &obsVec) {
    int loc1 = (int)round(vals[0]);
    int loc2 = (int)round(vals[1]);
    
    double distStep = obsVec[0];
    
    double x1 = obsVec[1 + loc1];
    double y1 = obsVec[1 + mapSize + loc1];
    double o1 = obsVec[1 + 2 * mapSize + loc1];
    double x2 = obsVec[1 + loc2];
    double y2 = obsVec[1 + mapSize + loc2];
    double o2 = obsVec[1 + 2 * mapSize + loc2];

    double x2_pred = x1 + distStep * cos(o1);
    double y2_pred = y1 + distStep * sin(o1);
    double error = sqrt((x2_pred - x2)*(x2_pred - x2) + (y2_pred - y2)*(y2_pred - y2));

//    double dist = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
//    double error = dist - distStep;
    
//    double ret = exp(-error*error / (sigmaDist*sigmaDist));
    double ret = -error * error / (sigmaDist * sigmaDist);
    
//    ret = std::max(ret, -20.0);
//    if(std::isnan(ret) || std::isinf(ret)){
//        ret = 0;
//    }
    return ret;
}

double MoveFeature::compParam(const std::vector<double> &vals,
                              const std::vector<double> &params,
                              const std::vector<double> &obsVec)
{
    return params[paramNum()]*comp(vals, obsVec);
}



OrientDiffFeature::OrientDiffFeature(int iid,
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

double OrientDiffFeature::comp(const std::vector<double> &vals, const std::vector<double> &obsVec) {
    int loc1 = (int)round(vals[0]);
    int loc2 = (int)round(vals[1]);

    double orientDiffMeas = obsVec[0];

    double o1 = obsVec[1 + loc1];
    double o2 = obsVec[1 + loc2];

    double orientDiff = Utils::toPiRange(o2 - o1);

    double error = Utils::angDiff(orientDiff, orientDiffMeas);

//    double ret = exp(-error*error / (sigmaDist*sigmaDist));
    double ret = -error * error / (sigmaOrient * sigmaOrient);

//    ret = std::max(ret, -20.0);
//    if(std::isnan(ret) || std::isinf(ret)){
//        ret = 0;
//    }
    return ret;
}

double OrientDiffFeature::compParam(const std::vector<double> &vals,
                                    const std::vector<double> &params,
                                    const std::vector<double> &obsVec)
{
    return params[paramNum()]*comp(vals, obsVec);
}
