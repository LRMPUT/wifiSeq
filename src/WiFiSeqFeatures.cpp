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
    int oOffsetIdx = (int)round(vals[0]);
    int loc = (int)round(vals[1]);

    double orientMeas = obsVec[0];

    double oOffset = obsVec[1 + oOffsetIdx];
    double orient = obsVec[1 + orientSectors + loc];

    double orientAbs = Utils::toPiRange(orient + oOffset);

    double error = Utils::angDiff(orientMeas, orientAbs);

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
    double x2 = obsVec[1 + loc2];
    double y2 = obsVec[1 + mapSize + loc2];

    double dist = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
    
    double distDiff = dist - distStep;
    
//    double ret = exp(-distDiff*distDiff / (sigmaDist*sigmaDist));
    double ret = -distDiff*distDiff / (sigmaDist*sigmaDist);
    
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
