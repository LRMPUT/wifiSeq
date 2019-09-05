//
// Created by janw on 03.09.2019.
//

#ifndef WIFISEQ_GRAPH_HPP
#define WIFISEQ_GRAPH_HPP

#include <set>
#include <map>
#include <vector>

class Graph {
public:
    Graph(const std::set<int> &allowedVals);

    double getDist(int val1, int val2);
private:
    int numNodes;

    std::map<int, int> varValToIdx;

    // distances between all nodes
    std::vector<std::vector<double>> dists;
};


#endif //WIFISEQ_GRAPH_HPP
