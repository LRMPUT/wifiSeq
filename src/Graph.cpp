/*
	Copyright (c) 2015,	Mobile Robots Laboratory:
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

#include <limits>
#include <cmath>

#include "Graph.hpp"
#include "Utils.hpp"

using namespace std;

Graph::Graph(const std::set<int> &allowedVals) {
    numNodes = allowedVals.size();
    dists = vector<vector<double>>(numNodes, vector<double>(numNodes, std::numeric_limits<double>::max()));
    for(int n = 0; n < numNodes; ++n){
        dists[n][n] = 0.0;
    }

    int idx = 0;
    for(auto it = allowedVals.begin(); it != allowedVals.end(); ++it){
        varValToIdx[*it] = idx;
        ++idx;
    }

    // for every node
    for(auto it = allowedVals.begin(); it != allowedVals.end(); ++it){
        int val = *it;
        int xIdx, yIdx, oIdx;
        Utils::valToMapGrid(val, xIdx, yIdx, oIdx);

        // 4 neighbors
        int nh[][2] = {{-1, 0}, {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}};
        for(int n = 0; n < sizeof(nh)/sizeof(nh[0]); ++n){
            int nhXIdx = xIdx + nh[n][0];
            int nhYIdx = yIdx + nh[n][1];
            int nhVal = Utils::mapGridToVal(nhXIdx, nhYIdx, oIdx);
            // if both nodes are allowed then initialize distance to gridSize;
            if(allowedVals.count(nhVal) > 0){
                int dx = xIdx - nhXIdx;
                int dy = yIdx - nhYIdx;
                double dist = std::sqrt(dx*dx + dy*dy) * mapGrid;
                dists[varValToIdx[val]][varValToIdx[nhVal]] = dist;
                dists[varValToIdx[nhVal]][varValToIdx[val]] = dist;
            }
        }
    }

    for(int k = 0; k < numNodes; ++k){
        for(int i = 0; i < numNodes; ++i){
            for(int j = 0; j < numNodes; ++j){
                if(dists[i][j] > dists[i][k] + dists[k][j]){
                    dists[i][j] = dists[i][k] + dists[k][j];
                }
            }
        }
    }
}

double Graph::getDist(int val1, int val2) {
    int idx1 = varValToIdx[val1];
    int idx2 = varValToIdx[val2];

    return dists[idx1][idx2];
}
