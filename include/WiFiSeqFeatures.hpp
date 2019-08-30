//
// Created by jachu on 15.05.18.
//

#ifndef WIFISEQ_WIFISEQFEATURES_HPP
#define WIFISEQ_WIFISEQFEATURES_HPP

#include "pgm/Pgm.h"

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


class OrientFeature : public Feature
{
public:
    OrientFeature(int iid,
                  int iparamNum,
                  const std::vector<std::shared_ptr<RandVar>> &irandVarsOrdered,
                  const std::vector<int> &iobsNums,
                  double isigmaOrient);

    double comp(const std::vector<double> &vals, const std::vector<double> &obsVec) override;

    double compParam(const std::vector<double> &vals,
                     const std::vector<double> &params,
                     const std::vector<double> &obsVec) override;

private:
    double sigmaOrient;
};


class MoveFeature : public Feature
{
public:
    MoveFeature(int iid,
                    int iparamNum,
                    const std::vector<std::shared_ptr<RandVar>> &irandVarsOrdered,
                    const std::vector<int> &iobsNums,
                    int imapSize,
                    double isigmaDist);
    
    double comp(const std::vector<double> &vals, const std::vector<double> &obsVec) override;
    
    double compParam(const std::vector<double> &vals,
                     const std::vector<double> &params,
                     const std::vector<double> &obsVec) override;
    
private:
    int mapSize;
    double sigmaDist;
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


class OrientDiffFeature : public Feature
{
public:
    OrientDiffFeature(int iid,
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
