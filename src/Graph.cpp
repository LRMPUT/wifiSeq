//
// Created by janw on 03.09.2019.
//

#include <limits>
#include <cmath>

#include "Graph.hpp"
#include "Utils.hpp"

using namespace std;

Graph::Graph(const std::set<int> &allowedVals) {
    numNodes = allowedVals.size();
    dists = vector<vector<double>>(numNodes, vector<double>(numNodes, std::numeric_limits<double>::max()));

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
