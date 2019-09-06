//
// Created by jachu on 15.05.18.
//

#ifndef WIFISEQ_WIFISEQFEATURES_HPP
#define WIFISEQ_WIFISEQFEATURES_HPP

#include "pgm/Pgm.h"
#include "Graph.hpp"

class LocFeature : public Feature
{
public:
    LocFeature(int iid,
               int iparamNum,
               const std::vector<std::shared_ptr<RandVar>> &irandVarsOrdered,
               const std::vector<int> &iobsNums);
    
    double comp(const std::vector<double> &vals, const std::vector<double> &obsVec) override;
    
    double compParam(const std::vector<double> &vals,
                     const std::vector<double> &params,
                     const std::vector<double> &obsVec) override;
    
private:

};

class MoveFeature : public Feature
{
public:
    MoveFeature(int iid,
                    int iparamNum,
                    const std::vector<std::shared_ptr<RandVar>> &irandVarsOrdered,
                    const std::vector<int> &iobsNums,
                    int imapSize,
                    double isigmaDist,
                    std::shared_ptr<Graph> graph);
    
    double comp(const std::vector<double> &vals, const std::vector<double> &obsVec) override;
    
    double compParam(const std::vector<double> &vals,
                     const std::vector<double> &params,
                     const std::vector<double> &obsVec) override;
    
private:
    int mapSize;
    double sigmaDist;
    std::shared_ptr<Graph> graph;
};


class OrientMoveFeature : public Feature
{
public:
    OrientMoveFeature(int iid,
                      int iparamNum,
                      const std::vector<std::shared_ptr<RandVar>> &irandVarsOrdered,
                      const std::vector<int> &iobsNums,
                      int imapSize,
                      double isigmaOrient);

    double comp(const std::vector<double> &vals, const std::vector<double> &obsVec) override;

    double compParam(const std::vector<double> &vals,
                     const std::vector<double> &params,
                     const std::vector<double> &obsVec) override;

private:
    int mapSize;
    double sigmaOrient;
};


#endif //WIFISEQ_WIFISEQFEATURES_HPP
