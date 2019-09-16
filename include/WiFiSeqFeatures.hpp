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
