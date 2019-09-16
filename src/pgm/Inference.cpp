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

#include "pgm/Inference.h"

#include <algorithm>
#include <set>
#include <queue>
#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>

#include "pgm/UnionFind.h"
//#include "HopPgm.h"
//#include "HopLearn.h"

using namespace std;

bool operator<(const Inference::InfEdge& lh, const Inference::InfEdge& rh){
	return (lh.w < rh.w);
}

//-------------------PRIVATE--------------------

std::vector<double> Inference::compCurMarginal(std::shared_ptr<Cluster> clust,
												MargType type,
												const std::vector<std::vector<double> >& inMsgs,
												const std::vector<double>& params,
												const std::vector<double>& obsVec)
{
	return clust->marginalize(vector<std::shared_ptr<RandVar>>(),
								type,
								inMsgs,
								params,
								obsVec);
}

std::vector<int> Inference::selectMST(const std::vector<InfEdge>& edges, //weight sorted
										int numClusters)
{
	UnionFind set(numClusters);
	vector<int> selEdges;
	for(int e = 0; e < (int)edges.size(); ++e){
		int rooti = set.findSet(edges[e].i);
		int rootj = set.findSet(edges[e].j);
		if(rooti != rootj){
			selEdges.push_back(e);
			set.unionSets(rooti, rootj);
		}
	}
	return selEdges;
}

void Inference::passMessage(MargType type,
							std::vector<std::vector<std::vector<double> > >& msgs,
							std::vector<std::vector<std::vector<double> > >& prevMsgs,
							std::shared_ptr<Cluster> src,
							std::shared_ptr<Cluster> dst,
							const std::vector<double>& params,
							const std::vector<double>& obsVec)
{
	vector<std::shared_ptr<RandVar>> margRandVars;
	int nhc = 0;
	while(src->nh()[nhc].lock()->id() != dst->id()){
		++nhc;
		if(nhc >= (int)src->nh().size()){
			throw "Inference::passMessage: Error nhc out of range";
		}
	}

//	cout << "src->sepset(" << nhc << ").size() = " << src->sepset(nhc).size() << endl;
	int sepsetVar = 0;
	for(int v = 0; v < (int)src->randVars().size(); ++v){
		int curVarId = src->randVars()[v]->id();
		while(curVarId > src->sepset(nhc)[sepsetVar]->id()){
			if(sepsetVar >= (int)src->sepset(nhc).size() - 1){
				break;
			}
			else{
				++sepsetVar;
			}
		}
//		cout << "curVarId = " << curVarId << endl;
//		cout << "src->sepset(" << nhc << ")[" << sepsetVar << "]->id() = " << src->sepset(nhc)[sepsetVar]->id() << endl;
		if(curVarId != src->sepset(nhc)[sepsetVar]->id()){
			margRandVars.push_back(src->randVars()[v]);
		}
	}

//	cout << "Marginalizing rand vars: ";
//	for(int rv = 0; rv < (int)margRandVars.size(); ++rv){
//		cout << margRandVars[rv]->id() << " ";
//	}
//	cout << endl;
//	if((src->id() == 30247 || src->id() == 30260) && (dst->id() == 23729)){
//		cout << "message to " << dst->id() << endl;
//	}
//	cout << "message from " << src->id() << " to " << dst->id() << endl;

	vector<double> msgSrcDst = src->marginalize(margRandVars,
												type,
												msgs[src->id()],
												params,
												obsVec,
												dst);

//	if(dst->id() == 1 || dst->id() == 148){
//		cout << "message from " << src->id() << " to " << dst->id() << " = " << msgSrcDst << endl;
//	}

//	for(int m = 0; m < (int)msgSrcDst.size(); ++m){
//		if(std::isnan(msgSrcDst[m]) || std::isinf(msgSrcDst[m])){
//			cout << "msg from " << src->id() << " to " << dst->id() << " = " << msgSrcDst << endl;
//			char a;
//			cin >> a;
//		}
//	}

//	cout << "Saving message" << endl;
	int dstNhc = 0;
	while(dst->nh()[dstNhc].lock()->id() != src->id()){
		++dstNhc;
		if(dstNhc >= (int)dst->nh().size()){
			throw "Inference::passMessage: Error dstNhc out of range";
		}
	}
//	if(dst->id() == 80){
//		cout << "Message from " << src->id() << " to " << dst->id() << " : ";
//		for(int idx = 0; idx < (int)msgSrcDst.size(); ++idx){
//			cout << msgSrcDst[idx] << " ";
//		}
//		cout << endl;
//	}

	prevMsgs[dst->id()][dstNhc] = msgs[dst->id()][dstNhc];
	msgs[dst->id()][dstNhc] = msgSrcDst;
}

void Inference::treeReparametrize(const std::vector<std::shared_ptr<Cluster>>& clusters,
									MargType type,
									std::vector<std::vector<std::vector<double> > >& msgs,
									std::vector<std::vector<std::vector<double> > >& prevMsgs,
									const std::vector<InfEdge>& edges,
									const std::vector<int>& selEdges,
									const std::vector<double>& params,
									const std::vector<double>& obsVec)
{
	vector<vector<int> > graph(clusters.size(), vector<int>());
	vector<set<int> > msgsNotReceived(clusters.size(), set<int>());
	for(int se = 0; se < (int)selEdges.size(); se++){
		int i = edges[selEdges[se]].i;
		int j = edges[selEdges[se]].j;
		msgsNotReceived[i].insert(j);
		msgsNotReceived[j].insert(i);
		graph[i].push_back(j);
		graph[j].push_back(i);
	}
	queue<pair<int, int> > msgsToPass;
	vector<int> msgsSent(clusters.size(), -1);

	for(int c = 0; c < (int)msgsNotReceived.size(); ++c){
		if(msgsNotReceived[c].size() == 1){
			msgsToPass.push(pair<int, int>(c, *(msgsNotReceived[c].begin())));
			msgsSent[c] = *(msgsNotReceived[c].begin());
		}
	}
//	cout << "graph.size() = " << graph.size() << endl;

	using namespace std::chrono;

//	duration<double> posExTime = duration<double>::zero();
//	duration<double> inOutTime = duration<double>::zero();
//	int timesPosEx = 0;
//	int timesInOut = 0;



//	int msgCnt = 0;
	while(!msgsToPass.empty()){
		int srcId = msgsToPass.front().first;
		int dstId = msgsToPass.front().second;
		msgsToPass.pop();

//		if(msgCnt % 10000 == 0){
//			cout << "msg " << msgCnt << endl;
//		}
//		cout << "Passing message from " << srcId << " to " << dstId << endl;


//		int clustType = -1;
//		if(dynamic_cast<std::shared_ptr<PosExCluster>>(pgm.constClusters()[srcId])){
//			clustType = 0;
//		}
//		else if(dynamic_cast<std::shared_ptr<InOutCluster>>(pgm.constClusters()[srcId])){
//			clustType = 1;
//		}

//		high_resolution_clock::time_point start = high_resolution_clock::now();

		passMessage(type,
					msgs,
					prevMsgs,
					clusters[srcId],
					clusters[dstId],
					params,
					obsVec);

//		high_resolution_clock::time_point end = high_resolution_clock::now();

//		if(clustType >= 0){
//			duration<double> curTime = duration_cast<duration<double> >(end - start);
//			if(clustType == 0){
//				posExTime += curTime;
//				++timesPosEx;
//			}
//			else if(clustType == 1){
//				inOutTime += curTime;
//				++timesInOut;
//			}
//		}

//		cout << "Message passed" << endl;

		msgsNotReceived[dstId].erase(srcId);

		//Only 1 message not received, passing message to cluster which didn't send us message
		if(msgsNotReceived[dstId].size() == 1){
			msgsToPass.push(pair<int, int>(dstId, *(msgsNotReceived[dstId].begin())));
			msgsSent[dstId] = *(msgsNotReceived[dstId].begin());
		}
		//All messages received, sending messages to all clusters except from the one that was already sent a message
		else if(msgsNotReceived[dstId].size() == 0){
			for(int nhc = 0; nhc < (int)graph[dstId].size(); ++nhc){
				if(msgsSent[dstId] < 0){
					throw "Inference::treeReparametrize: msgsSent[dstId] < 0";
				}
				if(msgsSent[dstId] != graph[dstId][nhc]){
					msgsToPass.push(pair<int, int>(dstId, graph[dstId][nhc]));
				}
			}
		}

//		++msgCnt;
	}

//	cout << "PosExCluster time: " << posExTime.count() << ", " << timesPosEx << " times" << endl;
//	cout << "InOutCluster time: " << inOutTime.count() << ", " << timesInOut << " times" << endl;

}

bool Inference::runBP(const std::vector<std::shared_ptr<Cluster>>& clusters,
							MargType type,
							std::vector<std::vector<std::vector<double> > >& msgs,
							const std::vector<double>& params,
							const std::vector<double>& obsVec,
							const std::vector<bool>& maskClust)
{
	int numClusters = clusters.size();

//	std::vector<double> locObsVec = obsVec;

	vector<set<int> > edgesAdded(numClusters, set<int>());

	/*for(int p = 0; p < params.size(); ++p){
		cout << params[p] << " ";
	}
	cout << endl;*/

//	cout << "Adding edges, numClusters = " << numClusters << endl;
	vector<InfEdge> edges;
	for(int c = 0; c < numClusters; ++c){
//		cout << "Cluster " << c << " nh().size() = " << clusters[c]->nh().size() << endl;
		for(int nhc = 0; nhc < (int)clusters[c]->nh().size(); ++nhc){
			int inode = c;
			int jnode = clusters[c]->nh()[nhc].lock()->id();
//			cout << "jnode = " << jnode << endl;
			if(edgesAdded[inode].count(jnode) == 0){
				bool addEdge = true;
				if(!maskClust.empty()){
					if(maskClust[inode] != true && maskClust[jnode] != true){
						addEdge = false;
					}
				}
				if(addEdge){
					edges.push_back(InfEdge(inode, jnode, 0));
					edgesAdded[inode].insert(jnode);
					edgesAdded[jnode].insert(inode);
				}
//				if(jnode == 29778 || inode == 29778){
//					cout << "Edge from " << jnode << " to " << inode << endl;
//				}
			}
		}
	}

//	cout << "Creating msgs" << endl;
	if(msgs.empty()){
		//Initialize messages
//		cout << "initializing messages" << endl;
		for(int c = 0; c < numClusters; ++c){
//			cout << "Cluster " << c << endl;
			vector<vector<double> > clustMsg;
			for(int nhc = 0; nhc < (int)clusters[c]->nh().size(); ++nhc){
				int numVals = 1;
				for(int sep = 0; sep < (int)clusters[c]->sepset(nhc).size(); ++sep){
					numVals *= clusters[c]->sepset(nhc)[sep]->vals().size();
				}
//				cout << "nhc = " << nhc << ", numVals = " << numVals << endl;
				if(type == SumProduct){
					clustMsg.push_back(vector<double>(numVals, 1.0/numVals));
				}
				else if(type == MaxSum){
					clustMsg.push_back(vector<double>(numVals, log(1.0/numVals)));
				}
			}
			msgs.push_back(clustMsg);
		}
	}

	std::vector<std::vector<std::vector<double> > > prevMsgs;
	for(int c = 0; c < numClusters; ++c){
//			cout << "Cluster " << c << endl;
		vector<vector<double> > clustMsg;
		for(int nhc = 0; nhc < (int)clusters[c]->nh().size(); ++nhc){
			int numVals = 1;
			for(int sep = 0; sep < (int)clusters[c]->sepset(nhc).size(); ++sep){
				numVals *= clusters[c]->sepset(nhc)[sep]->vals().size();
			}
//				cout << "nhc = " << nhc << ", numVals = " << numVals << endl;
			if(type == SumProduct){
				clustMsg.push_back(vector<double>(numVals, 1.0/numVals));
			}
			else if(type == MaxSum){
				clustMsg.push_back(vector<double>(numVals, log(1.0/numVals)));
			}
		}
		prevMsgs.push_back(clustMsg);
	}

	static const int maxIter = 1;
	static const double maxEps = 1e-5;
	int iter = 0;

	bool testPass = false;
	while(iter < maxIter){
//		cout << "Iteration " << iter << endl;

//		cout << "obsVec = " << obsVec << endl;

//		using namespace chrono;
//
//		high_resolution_clock::time_point start = high_resolution_clock::now();
//
//		cout << "Sorting edges, edges.size() = " << edges.size() << endl;
		sort(edges.begin(), edges.end());

//		high_resolution_clock::time_point end = high_resolution_clock::now();
//
//		duration<double> curTime = duration_cast<duration<double> >(end - start);
//		cout << "Sorting time: " << curTime.count() << endl;

		//Edge selection using Kruskal algorithm
//		cout << "Selecting edges for reparametrization" << endl;
		vector<int> selEdges = selectMST(edges, numClusters);

//		std::vector<std::vector<std::vector<double> > > prevMsgs = msgs;
		//Run tree reparametrization
//		cout << "Reparametrizing" << endl;
		treeReparametrize(clusters,
							type,
							msgs,
							prevMsgs,
							edges,
							selEdges,
							params,
							obsVec);

//		static constexpr int clustId = 551539;
//		cout << "msgs[" << clustId << "] = " << msgs[clustId] << endl;
//		vector<int> clustersNotCalib;
		//Calibration test
//		cout << "Calibration test" << endl;
		int calibrated = 0;
		testPass = true;
		for(int c = 0; c < numClusters; ++c){
			bool isCalib = true;
			for(int nhc = 0; nhc < (int)msgs[c].size(); ++nhc){
				for(int val = 0; val < (int)msgs[c][nhc].size(); ++val){
					if(fabs(msgs[c][nhc][val] - prevMsgs[c][nhc][val]) > maxEps){
						testPass = false;
						isCalib = false;
//						if(iter == maxIter - 1){
//							clustersNotCalib.push_back(c);
//							cout << "edge (" << c << ", " << clusters[c]->nh()[nhc]->id() << ") not calibrated" << endl;
////							cout << msgs[c][nhc][val] << " - " << prevMsgs[c][nhc][val] << endl;
//						}
						break;
					}
				}
			}
			if(isCalib){
				++calibrated;
			}
		}
//		cout << "Calibrated " << double(calibrated)/numClusters << endl;
		if(testPass == true){
//			cout << "Graph calibrated" << endl;
			break;
		}

//		if(iter == maxIter - 1){
//			cout << "Clusters not calib:" << endl;
//			for(int c : clustersNotCalib){
//				cout << c << endl;
//			}
//		}

		//Increase weight
		for(int se = 0; se < (int)selEdges.size(); ++se){
			edges[selEdges[se]].w += 1;
		}

//		if(iter % 10 == 9){
//			for(int p = 0; p < int(curDebug->hopGraph.nodes.size()); ++p){
//				int outVarClustIdx = curDebug->hopGraph.nodes[p].outVarClusterIdx;
//				vector<double> bel = compCurMarginal(clusters[outVarClustIdx],
//																		type,
//																		msgs[clusters[outVarClustIdx]->id()],
//																		params,
//																		obsVec);
//
//			}
//		}

//		if(iter % 30 == 29){
//			for(int p = 0; p < int(curDebug->hopGraph.nodes.size()); ++p){
//				int exVarClustIdx = curDebug->hopGraph.nodes[p].exVarClusterIdx;
//				vector<double> bel = compCurMarginal(clusters[exVarClustIdx],
//																		type,
//																		msgs[clusters[exVarClustIdx]->id()],
//																		params,
//																		locObsVec);
//				double belNotEx = exp(bel[0]);
//
//				cout << "belNotEx = " << belNotEx << endl;
//				if(belNotEx > 0.5){
//					locObsVec[2*p + 1] *= 1.0 - (belNotEx);
//				}
//
//				cout << "pert for part " << p << " = " << locObsVec[2*p + 1] << endl;
//			}
//		}

//		//Compute marginals
//		for(int c = 0; c < numClusters; ++c){
//			if(clusters[c]->feats().size() == 0){
//				vector<double> bel = compCurMarginal(clusters[c], type, msgs[clusters[c]->id()], params, obsVec);
//				cout << "Beliefs for cluster " << c << ": ";
//				for(int b = 0; b < bel.size(); ++b){
//					cout << bel[b] << " ";
//				}
//				cout << endl;
//			}
//		}

		iter++;

//		char a;
//		cin >> a;
	}

	return testPass;
}

//-------------------PUBLIC--------------------

std::vector<std::vector<double> > Inference::compMarginalsBF(const Pgm& pgm,
														double& logPartFunc,
														const std::vector<double>& params,
														const std::vector<double>& obsVec)
{

	const vector<std::shared_ptr<RandVar>>& vars = pgm.constRandVars();
	vector<int> varValIdxs(vars.size(), 0);

	vector<vector<double> > marg(vars.size());
	for(int rv = 0; rv < (int)vars.size(); ++rv){
		marg[rv] = vector<double>(vars[rv]->vals().size(), 0.0);
	}
	logPartFunc = 0;

	int count = 0;
	do{
		vector<double> varVals(vars.size());
		for(int rv = 0; rv < (int)vars.size(); ++rv){
			varVals[rv] = vars[rv]->vals()[varValIdxs[rv]];
		}
		double probUnnorm = compJointProbUnnorm(pgm,
												varVals,
												params,
												obsVec);

		for(int rv = 0; rv < (int)vars.size(); ++rv){
			marg[rv][varValIdxs[rv]] += probUnnorm;
		}

		logPartFunc += probUnnorm;


		static const int div = 1000000;
		static const int exponent = 6;
		if(count % div == 0){
			cout << "..." << count/div << "x10^" << exponent << endl;
		}
		++count;

	}while(Pgm::incVarValIdxs(varValIdxs, vars));


	logPartFunc = log(logPartFunc);

	for(int rv = 0; rv < (int)vars.size(); ++rv){
		double sum = 0;
		for(double& v : marg[rv]){
			sum += v;
		}
		for(double& v : marg[rv]){
			v /= sum;
		}
	}

	return marg;
}

double Inference::compJointProbUnnorm(const Pgm& pgm,
								const std::vector<double>& vals,
								const std::vector<double>& params,
								const std::vector<double>& obsVec)
{
	double ret = 1.0;
	for(int c = 0; c < (int)pgm.constClusters().size(); ++c){

		vector<double> clustVals(pgm.constClusters()[c]->randVars().size(), 0.0);
		for(int rv = 0; rv < (int)pgm.constClusters()[c]->randVars().size(); ++rv){
			clustVals[rv] = vals[pgm.constClusters()[c]->randVars()[rv]->id()];
		}
		double curVal = pgm.constClusters()[c]->compFactorsVal(clustVals,
																SumProduct,
																params,
																obsVec);
		ret *= curVal;
	}
	return ret;
}

bool Inference::compMarginals(const Pgm& pgm,
								std::vector<std::vector<double> >& marg,
								//std::vector<std::shared_ptr<Cluster>> margToCompute,
								double& logPartFunc,
								std::vector<std::vector<std::vector<double> > >& msgs,
								const std::vector<double>& obsVec)
{
	return compMarginalsParam(pgm,
							marg,
							logPartFunc,
							msgs,
							pgm.constParams(),
							obsVec);
}

bool Inference::compMarginalsParam(const Pgm& pgm,
									std::vector<std::vector<double> >& marg,
									double& logPartFunc,
									std::vector<std::vector<std::vector<double> > >& msgs,
									const std::vector<double>& params,
									const std::vector<double>& obsVec)
{
	int numClusters = pgm.constClusters().size();

//	cout << "Running BP" << endl;

	bool calibrated = runBP(pgm.constClusters(),
							SumProduct,
							msgs,
							params,
							obsVec);


//	vector<vector<double> > retMarg(numClusters, vector<double>());
	marg = vector<vector<double>>(numClusters, vector<double>());
	//Compute marginals
	for(int c = 0; c < numClusters; ++c){
		if(pgm.constClusters()[c]->feats().size() == 0){
			marg[c] = compCurMarginal(pgm.constClusters()[c], SumProduct, msgs[pgm.constClusters()[c]->id()], params, obsVec);
		}
	}
	
//	cout << "Computing partition function" << endl;
	//Compute partition function
	//Bethe free energy approximation
	logPartFunc = 0.0;


	for(int c = 0; c < (int)pgm.constClusters().size(); c++){
//		cout << "Considering cluster " << c << endl;

		int numVars = pgm.constClusters()[c]->randVars().size();
		int numFeats = pgm.constClusters()[c]->feats().size();
		//Per variable
		if(numFeats == 0){
			double Hs = 0.0;
			for(int val = 0; val < (int)pgm.constClusters()[c]->randVars().front()->vals().size(); ++val){
				Hs += marg[c][val] * log(marg[c][val]);
//				if(std::isnan(log(marg[c][val]))){
//				    cout << "marg[" << c << "] = " << marg[c] << endl;
//				}
			}
			//Every nh cluster represents factor
			int deg = pgm.constClusters()[c]->nh().size();
			logPartFunc += (deg - 1)*Hs;
//			cout << "Hs = " << Hs << ", deg = " << deg << endl;
		}

		//Per factor
		else if(numFeats > 0){
			double Hcdiv = pgm.constClusters()[c]->compSumHcdiv(msgs[pgm.constClusters()[c]->id()],
																params,
																obsVec);
//            if(std::isnan(Hcdiv)){
//                cout << "Hcdiv = " << Hcdiv << endl;
//            }
//
//            Hcdiv = pgm.constClusters()[c]->compSumHcdiv(msgs[pgm.constClusters()[c]->id()],
//                                                                params,
//                                                                obsVec);
			//			cout << "Hcdiv = " << Hcdiv << endl;

			logPartFunc -= Hcdiv;
		}
//		cout << "logPartFunc = " << logPartFunc << endl;
	}

	return calibrated;
}

bool Inference::compMAPParam(const Pgm& pgm,
									std::vector<std::vector<double> >& marg,
									std::vector<std::vector<std::vector<double> > >& msgs,
									const std::vector<double>& params,
									const std::vector<double>& obsVec,
									const std::vector<bool>& maskClust)
{
	int numClusters = pgm.constClusters().size();

	bool calibrated = runBP(pgm.constClusters(),
							MaxSum,
							msgs,
							params,
							obsVec,
							maskClust);

	marg = vector<vector<double> >(numClusters);
	for(int c = 0; c < numClusters; ++c){
		vector<double> curMarg;
		if(pgm.constClusters()[c]->feats().size() == 0){
			curMarg = compCurMarginal(pgm.constClusters()[c], MaxSum, msgs[pgm.constClusters()[c]->id()], params, obsVec);
		}
		marg[c] = curMarg;
	}

	return calibrated;
}

std::vector<std::vector<double> > Inference::decodeMAP(const Pgm& pgm,
													std::vector<std::vector<double> >& marg,
													std::vector<std::vector<std::vector<double> > >& msgs,
													const std::vector<double>& params,
													const std::vector<double>& obsVec)
{
	int numClusters = pgm.constClusters().size();

	vector<vector<vector<int> > > testValsIdxs;

	//Compute max assignments
//	cout << "computing testValsIdxs" << endl;
//	cout << "Clusters with more than 1 bestValsIdx:" << endl;
	for(int c = 0; c < numClusters; ++c){

//		cout << "Cluster " << c << endl;

		vector<vector<int> > bestValsIdxs = pgm.constClusters()[c]->getBestValsIdxs(msgs[pgm.constClusters()[c]->id()],
																					params,
																					obsVec,
																					1e-9 /*eps*/);

//		cout << "bestValsIdxs = " << bestValsIdxs << endl;

//		if(bestValsIdxs.size() > 1){
//			cout << c << endl;
//		}
		testValsIdxs.push_back(bestValsIdxs);

//		cout << "Max vals for cluster " << c << endl;
//		for(int val = 0; val < maxVals.back().size(); ++val){
//			cout << maxVals.back()[val] << " ";
//		}
//		cout << endl;
	}

//	ofstream testValsIdxsFile("log/testValIdxs.log");
	int numComb = 1;
	for(int c = 0; c < numClusters; ++c){
//		cout << "testValsIdxs[" << c << "].size() = " << testValsIdxs[c].size() << endl;
//		testValsIdxsFile << "testValsIdxs[" << c << "].size() = " << testValsIdxs[c].size() << endl;
		if(testValsIdxs[c].size() == 0){
			cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! zero test vals idxs" << endl;
		}
		numComb *= testValsIdxs[c].size();
	}
//	testValsIdxsFile.close();
//	cout << "numComb = " << numComb << endl;

	//resolve ties
	double bestScore = 0;
	bool bestScoreConsistent = true;
	vector<int> bestVarValIdxs(numClusters, 0);
	vector<int> testVarValIdxs(numClusters, 0);

//	cout << "Resolving ties" << endl;
	int counter = 0;
	do{
//		cout << "combination " << counter++ << endl;

//		vector<bool> valChosen(pgm.constRandVars().size(), false);
		vector<int> curValsIdxs(pgm.constRandVars().size(), -1);
		bool isConsistent = true;

		for(int c = 0; c < numClusters; ++c){
			for(int rv = 0; rv < (int)pgm.constClusters()[c]->randVars().size(); ++rv){
				int curId = pgm.constClusters()[c]->randVars()[rv]->id();
				int curValIdx = testValsIdxs[c][testVarValIdxs[c]][rv];
				if(curValsIdxs[curId] == -1){
					curValsIdxs[curId] = curValIdx;
				}
				else if((curValsIdxs[curId] != curValIdx) &&
						(curValIdx != -1))
				{
					cout << "Inconsistent at var " << curId << endl;
					isConsistent = false;
				}

			}
		}
		vector<double> curVals(pgm.constRandVars().size(), 0.0);
		for(int cvIdx = 0; cvIdx < (int)curValsIdxs.size(); ++cvIdx){
			if(curValsIdxs[cvIdx] != -1){
				curVals[cvIdx] = pgm.constRandVars()[cvIdx]->vals()[curValsIdxs[cvIdx]];
			}
		}
		double curScore = compJointProbUnnorm(pgm,
												curVals,
												params,
												obsVec);
		if(curScore > bestScore){
			bestScore = curScore;
			bestScoreConsistent = isConsistent;
			bestVarValIdxs = testVarValIdxs;
		}

	}while(Pgm::incVarValIdxs(testVarValIdxs, testValsIdxs));

	if(!bestScoreConsistent){
		cout << "\n\n\n\n\n\n\n\n BEST SCORE INCONSISTENT \n\n\n\n\n\n\n\n" << endl;
	}

	std::vector<std::vector<double> > retVals;
	for(int c = 0; c < numClusters; ++c){
		retVals.push_back(vector<double>());
		for(int rv = 0; rv < (int)testValsIdxs[c][bestVarValIdxs[c]].size(); ++rv){
			const vector<double>& curVarVals = pgm.constClusters()[c]->randVars()[rv]->vals();
			if(testValsIdxs[c][bestVarValIdxs[c]][rv] != -1){
				retVals.back().push_back(curVarVals[testValsIdxs[c][bestVarValIdxs[c]][rv]]);
			}
			else{
				retVals.back().push_back(-1e9);
			}
		}
	}

	return retVals;
}
