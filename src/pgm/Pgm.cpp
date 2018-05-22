/*
	Copyright (c) 2015,	TAPAS Team:
	-Jan Wietrzykowski (jan.wietrzykowski@cie.put.poznan.pl).
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

#include "pgm/Pgm.h"

#include <iostream>
#include <cmath>
#include <queue>
#include <set>
#include <algorithm>


using namespace std;

std::vector<double> featScales;

//----------------RandVar----------------

RandVar::RandVar(int iid, std::vector<double> ivals) :
		idData(iid),
		isObserved(false),
		valsData(ivals)
{

}

RandVar::RandVar(int iid, double observedVal) :
		idData(iid),
		isObserved(true)
{
	valsData = vector<double>(1, observedVal);
}

RandVar::~RandVar(){
//	cout << "~RandVar() id " << idData << endl;
}

void RandVar::makeObserved(double val)
{
	isObserved = true;
	valsData = vector<double>{val};
}

void RandVar::makeNotObserved(std::vector<double> vals)
{
	isObserved = false;
	valsData = vals;
}

void RandVar::setVals(const std::vector<double>& newVals)
{
	valsData = newVals;
}

//----------------Feature----------------

Feature::Feature(int iid,
		int iparamNum,
		const std::vector<std::shared_ptr<RandVar>>& irandVarsOrdered,
		const std::vector<int>& iobsNums) :
	idData(iid),
	paramNumData(iparamNum),
	randVarsOrderedData(irandVarsOrdered),
	obsNumsData(iobsNums)
//	obsVecBeg(iobsVecBeg),
//	obsVecEnd(iobsVecEnd)
{

}

Feature::~Feature(){
//	cout << "~Feature() id " << idData << endl;
}

//----------------Cluster----------------

void Cluster::normalizeMarg(std::vector<double>& marg,
							MargType type) const
{
	//normalization
	if(type == SumProduct){
		double sumVal = 0;
		for(int v = 0; v < (int)marg.size(); ++v){
			sumVal += marg[v];
		}
		for(int v = 0; v < (int)marg.size(); ++v){
			marg[v] /= sumVal;
		}
	}
	else if(type == MaxSum){
		double maxVal = -1e9;
		for(int v = 0; v < (int)marg.size(); ++v){
			maxVal = max(marg[v], maxVal);
		}

		double sumVal = 0;
		for(int v = 0; v < (int)marg.size(); ++v){
			sumVal += exp(marg[v] - maxVal);
		}

		double normVal = log(1.0) - (maxVal + log(sumVal));

//		cout << "sumVal = " << sumVal << ", log(1.0/sumVal) = " << log(1.0/sumVal) << endl;
		for(int v = 0; v < (int)marg.size(); ++v){
			marg[v] += normVal;
		}
	}
}

Cluster::Cluster(int iid,
			const std::vector<std::shared_ptr<Feature>>& ifeats,
			const std::vector<std::shared_ptr<RandVar>>& irandVars) :
	idData(iid),
	featuresData(ifeats),
	randVarsData(irandVars)
{
	map<int, int> randVarIdToIdx;
	for(int rv = 0; rv  < randVarsData.size(); ++rv){
		randVarIdToIdx[randVarsData[rv]->id()] = rv;
	}
	for(int f = 0; f < featuresData.size(); ++f){
		const vector<shared_ptr<RandVar>>& featRandVarsOrdered = featuresData[f]->randVarsOrdered();
		randVarsOrderData.push_back(vector<int>(featRandVarsOrdered.size()));

		for(int rv = 0; rv < featRandVarsOrdered.size(); ++rv){
			int curRvId = featRandVarsOrdered[rv]->id();
			if(randVarIdToIdx.count(curRvId) > 0){
				randVarsOrderData.back()[rv] = randVarIdToIdx[curRvId];
			}
			else{
				throw "Error - feature's random variable not found in cluster's random variables";
			}
		}

		obsVecIdxsData.push_back(featuresData[f]->obsNums());
	}
}

Cluster::~Cluster(){
//	cout << "~Cluster id " << idData << endl;
}

void Cluster::setNh(std::vector<std::weak_ptr<Cluster>> newNh /*id sorted*/){
//	cout << "nhData.size() = " << nhData.size() << endl;
//	cout << "newNh.size() = " << newNh.size() << endl;
////	nhData = vector<std::shared_ptr<Cluster>>();
//	vector<std::shared_ptr<Cluster>> a = newNh;
//	cout << "setting" << endl;
	nhData = newNh;
//	cout << "end setting" << endl;
}

void Cluster::setSepsets(std::vector<std::vector<std::shared_ptr<RandVar>>> newSepsets /*id sorted*/){
	sepsetsData = newSepsets;
}

double Cluster::compFactorsVal(const std::vector<double>& varVals,
							MargType type,
							const std::vector<double>& params,
							const std::vector<double>& obsVec) const
{
	double ret = 0;
	vector<vector<double>> curObsVecs = getCurObsVecs(obsVec);

	for(int f = 0; f < (int)featuresData.size(); ++f){
		vector<double> varValsOrdered(randVarsOrderData[f].size());
		for(int rv = 0; rv < int(randVarsOrderData[f].size()); ++rv){
			varValsOrdered[rv] = varVals[randVarsOrderData[f][rv]];
		}
		double val = featuresData[f]->compParam(varValsOrdered, params, curObsVecs[f]);
		ret += val;
	}
	if(type == SumProduct){
		return exp(ret);
	}
	else if(type == MaxSum){
		return ret;
	}

	return exp(ret);
}

double Cluster::compSumHcdiv(const std::vector<std::vector<double> >& inMsgs,
									const std::vector<double>& params,
									const std::vector<double>& obsVec)
{
	vector<double> marg = marginalize(vector<std::shared_ptr<RandVar>>(),
										SumProduct,
										inMsgs,
										params,
										obsVec);
	int numVars = randVarsData.size();
	int numFeats = featuresData.size();
	double Hcdiv = 0.0;
	{
		vector<int> varValIdxs(numVars, 0);
		int numVarValIdxs = 0;

		//compute curObsVecs
		vector<vector<double>> curObsVecs = getCurObsVecs(obsVec);

		double sumMsg = 0.0;
		do{
	//				cout << "numVarValIdxs = " << numVarValIdxs << endl;

			double curMarg = marg[numVarValIdxs];

			double factorVal = 0;
			for(int f = 0; f < (int)featuresData.size(); ++f){
				vector<double> varValsOrdered(randVarsOrderData[f].size(), 0);
				for(int v = 0; v < randVarsOrderData[f].size(); ++v){
					varValsOrdered[v] = randVarsData[randVarsOrderData[f][v]]->vals()[varValIdxs[randVarsOrderData[f][v]]];
				}
				factorVal += featuresData[f]->compParam(varValsOrdered, params, curObsVecs[f]);
			}
			factorVal = exp(factorVal);
//			cout << "curMarg = " << curMarg << ", factorVal = " << factorVal << endl;

			sumMsg += curMarg;

			Hcdiv += curMarg * log(curMarg/factorVal);

//			if(std::isnan(Hcdiv)){
//				cout << "curMarg = " << curMarg << ", factorVal = " << factorVal << endl;
//			}
			
			numVarValIdxs += 1;
		}while(Pgm::incVarValIdxs(varValIdxs, randVarsData));

//		cout << "sumMsg = " << sumMsg << endl;
	}

//	double HcdivComp = 0.0;
//	double normConstComp = 0.0;
//
//	{
//		vector<int> varValsIdxs(randVarsData.size(), 0);
//
//		do{
//			double curMarg = 1.0;
//			vector<double> varVals(randVarsData.size());
//			for(int rv = 0; rv < (int)randVarsData.size(); ++rv){
//				varVals[rv] = randVarsData[rv]->vals()[varValsIdxs[rv]];
//			}
//
//			for(int rv = 0; rv < (int)randVarsData.size(); ++rv){
//				curMarg *= inMsgs[rv][varValsIdxs[rv]];
//			}
//
//			double factVal = compFactorsVal(varVals, SumProduct, params, obsVec);
//			curMarg *= factVal;
//
//			normConstComp += curMarg;
//		}while(Pgm::incVarValIdxs(varValsIdxs, randVarsData));
//
//		normConstComp = 1.0/normConstComp;
//
//		cout << "normConstComp = " << normConstComp << endl;
//	}
//
//	{
//		double sumMsg = 0.0;
//
//		vector<int> varValsIdxs(randVarsData.size(), 0);
//
//		do{
//			double curMarg = 1.0;
//			vector<double> varVals(randVarsData.size());
//			for(int rv = 0; rv < (int)randVarsData.size(); ++rv){
//				varVals[rv] = randVarsData[rv]->vals()[varValsIdxs[rv]];
//			}
//
//			for(int rv = 0; rv < (int)randVarsData.size(); ++rv){
//				curMarg *= inMsgs[rv][varValsIdxs[rv]];
//			}
//
//			double factVal = compFactorsVal(varVals, SumProduct, params, obsVec);
//			curMarg *= factVal;
//
//			sumMsg += curMarg*normConstComp;
//
//			HcdivComp += curMarg*normConstComp*log(curMarg*normConstComp/factVal);
//		}while(Pgm::incVarValIdxs(varValsIdxs, randVarsData));
//
//		cout << "sumMsg = " << sumMsg << endl;
//		cout << "HcdivComp = " << HcdivComp << endl;
//	}
//
//	cout << "Hcdiv = " << Hcdiv << endl;
//	char a;
//	cin >> a;

	return Hcdiv;
}

std::vector<double> Cluster::compSumModelExpectation(const std::vector<std::vector<double> >& inMsgs,
															const std::vector<double>& params,
															const std::vector<double>& obsVec)
{
	vector<double> marg = marginalize(vector<std::shared_ptr<RandVar>>(),
										SumProduct,
										inMsgs,
										params,
										obsVec);
	int numVars = randVarsData.size();
	int numFeats = featuresData.size();

	vector<double> Efi(numFeats, 0.0);

	vector<int> varValIdxs(randVarsData.size(), 0);
	int numVarValIdxs = 0;
	vector<vector<double>> curObsVecs = getCurObsVecs(obsVec);

	do{
//					cout << "numVarValIdxs = " << numVarValIdxs << endl;

		for(int f = 0; f < (int)featuresData.size(); ++f){
			vector<double> clustVarValsOrdered(randVarsOrderData[f].size(), 0);

			for(int v = 0; v < randVarsOrderData[f].size(); ++v){
				clustVarValsOrdered[v] = randVarsData[randVarsOrderData[f][v]]->vals()[varValIdxs[randVarsOrderData[f][v]]];
			}

			double curVal = featuresData[f]->comp(clustVarValsOrdered,
												curObsVecs[f]);

			Efi[f] += curVal*marg[numVarValIdxs];
//						cout << "paramNum = " << paramNum << ", curVal*marg[c][numVarValIdxs] = " <<
//								curVal << "*" << marg[c][numVarValIdxs] << endl;
		}

		numVarValIdxs += 1;
	}while(Pgm::incVarValIdxs(varValIdxs, randVarsData));

	return Efi;
}

std::vector<double> Cluster::compSumEmpiricalExpectation(const std::vector<double>& varVals,
													const std::vector<double>& obsVec) const
{
	int numFeats = featuresData.size();
	vector<double> Ed(numFeats, 0.0);

	vector<vector<double>> curObsVecs = getCurObsVecs(obsVec);

//	cout << "varValsOrdered = " << varValsOrdered << endl;
//	cout << "curObsVec = " << curObsVec << endl;
	for(int f = 0; f < numFeats; ++f){
		vector<double> varValsOrdered(randVarsOrderData[f].size());
		for(int rv = 0; rv < int(randVarsOrderData[f].size()); ++rv){
			varValsOrdered[rv] = varVals[randVarsOrderData[f][rv]];
		}

		double curVal = featuresData[f]->comp(varValsOrdered, curObsVecs[f]);
		Ed[f] = curVal;
	}

	return Ed;
}

 std::vector<std::vector<int> > Cluster::getBestValsIdxs(const std::vector<std::vector<double> >& inMsgs,
																const std::vector<double>& params,
																const std::vector<double>& obsVec,
																double eps)
{
	vector<double> maxVals = marginalize(vector<std::shared_ptr<RandVar>>(),
										MaxSum,
										inMsgs,
										params,
										obsVec);

//	cout << "inMsgs = " << inMsgs << endl;
//	cout << "maxVals = " << maxVals << endl;

	double bestScore = -1e9;
	vector<vector<int> > bestValsIdxs;

	int numVars = randVarsData.size();
	vector<int> varValIdxs(numVars, 0);
	int numVarValIdxs = 0;

	do{
		if(bestScore < maxVals[numVarValIdxs]){
			bestScore = maxVals[numVarValIdxs];
		}

		numVarValIdxs += 1;
	}while(Pgm::incVarValIdxs(varValIdxs, randVarsData));

	varValIdxs = vector<int>(numVars, 0);
	numVarValIdxs = 0;

	do{
		//possible tie
		if(fabs(bestScore - maxVals[numVarValIdxs]) < eps){
//			vector<double> vals(numVars, 0);
//			for(int v = 0; v < numVars; ++v){
//				vals[v] = randVarsData[v]->vals()[varValIdxs[v]];
//			}
			bestValsIdxs.push_back(varValIdxs);
		}

		numVarValIdxs += 1;
	}while(Pgm::incVarValIdxs(varValIdxs, randVarsData));


//		cout << "Max vals for cluster " << c << endl;
//		for(int val = 0; val < maxVals.back().size(); ++val){
//			cout << maxVals.back()[val] << " ";
//		}
//		cout << endl;

//	char a;
//	cin >> a;
	if(bestValsIdxs.size() > 1){
		int maxComb = 1;
		for(int rv = 0; rv < (int)randVarsData.size(); ++rv){
			maxComb *= randVarsData[rv]->vals().size();
		}
		if(maxComb == (int)bestValsIdxs.size()){
			bestValsIdxs = vector<vector<int> >(randVarsData.size(), vector<int>{-1});
		}
	}

	return bestValsIdxs;
}

std::vector<double> Cluster::marginalize(const std::vector<std::shared_ptr<RandVar>>& margVars /*id sorted*/,
									MargType type,
									const std::vector<std::vector<double> >& inMsgs,
									const std::vector<double>& params,
									const std::vector<double>& obsVec,
									const std::shared_ptr<Cluster> excluded)
{
	vector<int> margVarValIdxs(margVars.size(), 0);
	vector<bool> varIsMarg(randVarsData.size(), false);
	vector<int> varPosList(randVarsData.size(), 0);

	vector<shared_ptr<Cluster>> tmpNhData;
	for(weak_ptr<Cluster>& nh : nhData){
		tmpNhData.emplace_back(nh.lock());
	}
//	vector<vector<shared_ptr<RandVar>>> tmpSepsetsData;
//	for(vector<weak_ptr<RandVar>>& vecSep : sepsetsData){
//		tmpSepsetsData.emplace_back();
//		for(weak_ptr<RandVar>& sep : vecSep){
//			tmpSepsetsData.back().emplace_back(sep.lock());
//		}
//	}

	vector<vector<double>> curObsVecs = getCurObsVecs(obsVec);

	//cout << "Creating otherVars and varPosList, randVarsData.size() = " << randVarsData.size() << endl;
	vector<std::shared_ptr<RandVar>> otherVars;
	int posMargVars = 0;
	for(int rv = 0; rv < (int)randVarsData.size(); ++rv){
		if(margVars.size() > 0){
			while(randVarsData[rv]->id() > margVars[posMargVars]->id()){
				if(posMargVars >= (int)margVars.size() - 1){
					break;
				}
				else{
					++posMargVars;
				}
			}
			//cout << "margVars[posMargVars]->id() = " << margVars[posMargVars]->id() <<
			//		", randVarsData[rv]->id() = " << randVarsData[rv]->id() << endl;
			if(margVars[posMargVars]->id() != randVarsData[rv]->id()){
				otherVars.push_back(randVarsData[rv]);
				varPosList[rv] = otherVars.size() - 1;
			}
			else{
				varIsMarg[rv] = true;
				varPosList[rv] = posMargVars;
			}
		}
		else{
			otherVars.push_back(randVarsData[rv]);
			varPosList[rv] = otherVars.size() - 1;
		}
	}
	vector<int> otherVarValIdxs(otherVars.size(), 0);

	vector<vector<int>> nhSepToRvPos;

	for(int nhc = 0; nhc < (int)sepsetsData.size(); ++nhc){
		nhSepToRvPos.emplace_back(sepsetsData[nhc].size(), -1);
		int rvPos = 0;

		for(int sep = 0; sep < (int)sepsetsData[nhc].size(); ++sep){
			while(randVarsData[rvPos]->id() < sepsetsData[nhc][sep]->id()){
				if(rvPos >= (int)randVarsData.size() - 1){
					break;
				}
				else{
					++rvPos;
				}
			}
			if(randVarsData[rvPos]->id() == sepsetsData[nhc][sep]->id()){
				nhSepToRvPos[nhc][sep] = rvPos;
			}
		}
	}

	//cout << "otherVars.size() = " << otherVars.size() << endl;
	int margLen = 1;
	for(int ov = 0; ov < (int)otherVars.size(); ++ov){
		//cout << "otherVars[ov]->vals().size() = " << otherVars[ov]->vals().size() << endl;
		margLen *= otherVars[ov]->vals().size();
	}

	vector<double> marg;

	if(type == SumProduct){
		marg = vector<double>(margLen, 0.0);
	}
	else if(type == MaxSum){
		marg = vector<double>(margLen, -1e9);
	}
//	cout << "Starting main marginalization loop" << endl;
	int margIdx = 0;
	do{

		do{

			double margCurVal = 1.0;
			if(type == SumProduct){
				margCurVal = 1.0;
			}
			else if(type == MaxSum){
				margCurVal = 0.0;
			}

			vector<double> curVarVals(randVarsData.size(), 0.0);

//			cout << "randVarsOrderData = " << randVarsOrderData << endl;
			int posCurVarVals = 0;
			int mv = 0; //marginal variable index
			int ov = 0;	//other variable index
			while((mv < (int)margVars.size()) || (ov < (int)otherVars.size())){
//				cout << "posCurVarVals = " << posCurVarVals << endl;
				if((mv < (int)margVars.size()) && (ov < (int)otherVars.size())){
					if(margVars[mv]->id() < otherVars[ov]->id()){
						curVarVals[posCurVarVals] = margVars[mv]->vals()[margVarValIdxs[mv]];
						++mv;
					}
					else{
						curVarVals[posCurVarVals] = otherVars[ov]->vals()[otherVarValIdxs[ov]];
						++ov;
					}
				}
				else if(mv < (int)margVars.size()){
					curVarVals[posCurVarVals] = margVars[mv]->vals()[margVarValIdxs[mv]];
					++mv;
				}
				else if(ov < (int)otherVars.size()){
					curVarVals[posCurVarVals] = otherVars[ov]->vals()[otherVarValIdxs[ov]];
					++ov;
				}

				++posCurVarVals;
			}

			//features
//			cout << "Computing features" << endl;
			for(int f = 0; f < (int)featuresData.size(); ++f){

				vector<double> curVarValsOrdered(randVarsOrderData[f].size());
				for(int rv = 0; rv < int(randVarsOrderData[f].size()); ++rv){
					curVarValsOrdered[rv] = curVarVals[randVarsOrderData[f][rv]];
				}
//				cout << "Computing feature id = " << featuresData[f]->id() <<
//						", curVarVals.size() = " << curVarValsOrdered.size() <<
//						", params.size() = " << params.size() <<
//						", curObsVec = " << curObsVec <<
//						", value = " << featuresData[f]->compParam(curVarValsOrdered, params, curObsVec) << endl;
				double exponent = featuresData[f]->compParam(curVarValsOrdered, params, curObsVecs[f]);
//				cout << "end computing" << endl;
//				if(id() == 148){
//					cout << "curVarValsOrdered = " << curVarValsOrdered << endl;
//					cout << "params = " << params << endl;
//					cout << "curObsVec = " << curObsVec << endl;
//					cout << "exponent = " << exponent << endl;
//				}
				if(type == SumProduct){
					margCurVal *= exp(exponent);
				}
				else if(type == MaxSum){
					margCurVal += exponent;
				}
			}
//			cout << "end computing" << endl;

			//received messages
			for(int nhc = 0; nhc < (int)nhData.size(); ++nhc){

				//cout << "Preparing message index from " << nhData[nhc]->id() <<
				//		", nhData.size() = " << nhData.size() << endl;

				bool excludeThisNh = false;
				if(excluded != 0){
					if(tmpNhData[nhc]->id() == excluded->id()){
						//cout << "Excluding " << nhData[nhc]->id() << endl;
						excludeThisNh = true;
					}
				}

				if(!excludeThisNh){
					//computing index of message value

					int msgIdx = 0;
					int msgIdxMul = 1;
					for(int sep = 0; sep < (int)sepsetsData[nhc].size(); ++sep){
						int rvPos = nhSepToRvPos[nhc][sep];

						if(rvPos >= 0){
							int curSepsetVarIdx = 0;
							if(varIsMarg[rvPos]){
								curSepsetVarIdx = margVarValIdxs[varPosList[rvPos]];
							}
							else{
								curSepsetVarIdx = otherVarValIdxs[varPosList[rvPos]];
							}
							msgIdx += msgIdxMul * curSepsetVarIdx;
							msgIdxMul *= sepsetsData[nhc][sep]->vals().size();
						}
					}

	//				cout << "Multiplying in message value from cluster " << nhData[nhc]->id() <<
	//						", value = " << inMsgs[nhc][msgIdx] << endl;

					if(type == SumProduct){
						margCurVal *= inMsgs[nhc][msgIdx];
					}
					else if(type == MaxSum){
						margCurVal += inMsgs[nhc][msgIdx];
					}
				}
			}

//			if(id() == 148){
//				cout << "marg[" << margIdx << "] = max(" << margCurVal << ", " << marg[margIdx] << ")" << endl;
//			}

			if(type == SumProduct){
				marg[margIdx] += margCurVal;
			}
			else if(type == MaxSum){
				marg[margIdx] = max(margCurVal, marg[margIdx]);
			}

			//iterate through all combinations of other variables
		} while(Pgm::incVarValIdxs(margVarValIdxs, margVars));

		//iterate through all combinations of marginalized variables
		++margIdx;
//		cout << "margIdx = " << margIdx << endl;
	} while(Pgm::incVarValIdxs(otherVarValIdxs, otherVars));

//	cout << "Before normalization" << endl;
//	for(int v = 0; v < marg.size(); ++v){
//		cout << marg[v] << ", ";
//	}
//	cout << endl;

//	if(id() == 148){
//		cout << "before norm, marg = " << marg << endl;
//	}

	//normalization
	normalizeMarg(marg, type);

//	cout << "After normalization" << endl;
//	for(int v = 0; v < marg.size(); ++v){
//		cout << marg[v] << ", ";
//	}
//	cout << endl;

//	if(id() == 148){
//		cout << "after norm, marg = " << marg << endl;
//
//		cout << "inMsgs = " << inMsgs << endl;
//		char a;
//		cin >> a;
//	}

//	for(int m = 0; m < (int)marg.size(); ++m){
//		if(std::isnan(marg[m]) || std::isinf(marg[m])){
//			cout << "marg " << id() << " = " << marg << endl;
////			cout << "inMsgs = " << inMsgs << endl;
//			char a;
//			cin >> a;
//		}
//	}

	return marg;
}



VECluster::VECluster(int iid,
				const std::vector<std::shared_ptr<Feature>>& ifeats,
				const std::vector<std::shared_ptr<RandVar>>& irandVars /*id sorted*/) :
		Cluster(iid, ifeats, irandVars)
{

}



double VECluster::compFactorsVal(const std::vector<double>& varVals,
							MargType type,
							const std::vector<double>& params,
							const std::vector<double>& obsVec) const
{

}

double VECluster::compFactorsValOrdered(const std::vector<double>& orderedVarVals,
									MargType type,
									const std::vector<double>& params,
									const std::vector<double>& obsVec) const
{

}

double VECluster::compSumHcdiv(const std::vector<std::vector<double> >& inMsgs,
							const std::vector<double>& params,
							const std::vector<double>& obsVec)
{

}

std::vector<double> VECluster::compSumModelExpectation(const std::vector<std::vector<double> >& inMsgs,
													const std::vector<double>& params,
													const std::vector<double>& obsVec)
{

}

std::vector<double> VECluster::compSumEmpiricalExpectation(const std::vector<double>& varVals,
													const std::vector<double>& obsVec) const
{

}

std::vector<std::vector<int> > VECluster::getBestValsIdxs(const std::vector<std::vector<double> >& inMsgs,
														const std::vector<double>& params,
														const std::vector<double>& obsVec,
														double eps)
{

}

std::vector<double> VECluster::marginalize(const std::vector<std::shared_ptr<RandVar>>& margVars /*id sorted*/,
									MargType type,
									const std::vector<std::vector<double> >& inMsgs,
									const std::vector<double>& params,
									const std::vector<double>& obsVec,
									const std::shared_ptr<Cluster> excluded)
{

}



double VECluster::compNormConstMarg(const std::vector<std::vector<double> >& inMsgs,
						const std::vector<double>& params,
						const std::vector<double>& obsVec) const
{

}




void VariableElimination::buildCliqueTree(const std::vector<std::shared_ptr<Feature>>& feats,
										const std::vector<std::shared_ptr<RandVar>>& randVars /*id sorted*/,
										std::vector<std::shared_ptr<Cluster>>& clusters,
										std::vector<std::shared_ptr<Cluster>>& randVarClusters)
{
	clusters.clear();
	randVarClusters.clear();

	map<int, int> randVarIdToIdx;
	for(int rv = 0; rv < randVars.size(); ++rv){
		randVarIdToIdx[randVars[rv]->id()] = rv;
	}

	vector<VEFactor> factorsVE;
	vector<VERandVar> randVarsVE;

	int nextFactId = 0;
	map<int, int> factIdToClusterIdx;
	for(int f = 0; f < feats.size(); ++f){
		factIdToClusterIdx[nextFactId] = -1;
		vector<int> randVarIds(feats[f]->randVarsOrdered().size());
		for(int rv = 0; rv < feats[f]->randVarsOrdered().size(); ++rv){
			randVarIds[rv] = feats[f]->randVarsOrdered()[rv]->id();
		}
		factorsVE.emplace_back(nextFactId++, randVarIds);
	}

	map<int, int> factIdToIdx;
	makeRandVars(factorsVE, randVarsVE, factIdToIdx);
//				cout << "factors.size() = " << factors.size() << ", randVars.size() = " << randVars.size() << endl;

	long long int maxCard = 0;
	long long int sumCard = 0;
	while(randVarsVE.size() > 0){

//					if(!image.empty()){
//						waitKey(10);
//					}

		int elimRandVarIdx = eliminateRandVar(randVars,
											factorsVE,
											randVarsVE,
											factIdToIdx);

		vector<bool> factToEliminate(factorsVE.size(), false);
		for(int& factId : randVarsVE[elimRandVarIdx].factorClustIds){
			int& factIdx = factIdToIdx[factId];
			factToEliminate[factIdx] = true;
		}

		vector<shared_ptr<Feature>> clusterFeats;
		vector<weak_ptr<Cluster>> clusterNh;

		std::set<int> newFactRandVarIds;
		vector<VEFactor> tmpFactors;
		for(int f = 0; f < factorsVE.size(); ++f){
			if(!factToEliminate[f]){
				tmpFactors.push_back(factorsVE[f]);
			}
			else{
				// factor not associated with any cluster, created from feature - adding to current clusters features
				if(factIdToClusterIdx[factorsVE[f].id] < 0){
					clusterFeats.push_back(feats[factorsVE[f].id]);
				}
				// factor associated with other cluster - adding edge between those clusters
				else{
					clusterNh.push_back(weak_ptr<Cluster>(clusters[factIdToClusterIdx[factorsVE[f].id]]));
				}
				for(int rv = 0; rv < factorsVE[f].randVarIds.size(); ++rv){
					if(factorsVE[f].randVarIds[rv] != randVarsVE[elimRandVarIdx].randVarId){
						newFactRandVarIds.insert(factorsVE[f].randVarIds[rv]);
					}
				}
			}
		}

		VEFactor newFactor;
		for(auto it = newFactRandVarIds.begin(); it != newFactRandVarIds.end(); ++it){
			newFactor.randVarIds.push_back(*it);
		}
		if(newFactor.randVarIds.size() > 0){
//						cout << "Adding new factor with random variables: " << newFactor.randVarIds << endl;
			newFactor.id = nextFactId++;
			factIdToClusterIdx[newFactor.id] = clusters.size();

			vector<shared_ptr<RandVar>> clusterRandVars;
			for(int rv = 0; rv < newFactor.randVarIds.size(); ++rv){
				clusterRandVars.push_back(randVars[randVarIdToIdx[newFactor.randVarIds[rv]]]);
			}

			clusters.emplace_back(new Cluster(clusters.size(),
												clusterFeats,
												clusterRandVars));

			tmpFactors.push_back(newFactor);
		}
		factorsVE = tmpFactors;

		makeRandVars(factorsVE, randVarsVE, factIdToIdx);


		long long int card = randVars[randVarsVE[elimRandVarIdx].randVarId]->vals().size();
		for(auto it = newFactRandVarIds.begin(); it != newFactRandVarIds.end(); ++it){
			card *= randVars[*it]->vals().size();
		}
		maxCard = std::max(maxCard, card);
		sumCard += card;
//					cout << "cluster card = " << card << endl;
	}
}

void VariableElimination::makeRandVars(std::vector<VEFactor>& factors,
									std::vector<VERandVar>& randVars,
									std::map<int, int>& factIdToIdx)
{
	randVars.clear();
	factIdToIdx.clear();
	map<int, int> rvIdsToIdxs;
	for(int f = 0; f < factors.size(); ++f){
		const VEFactor& curFact = factors[f];
		factIdToIdx[curFact.id] = f;
		for(const int& rvId : curFact.randVarIds){
			if(rvIdsToIdxs.count(rvId) == 0){
				rvIdsToIdxs[rvId] = randVars.size();
				randVars.push_back(VERandVar(rvId, vector<int>()));
			}
			int rvIdx = rvIdsToIdxs[rvId];
			randVars[rvIdx].factorClustIds.push_back(curFact.id);
		}
	}
}

int VariableElimination::eliminateRandVar(const std::vector<std::shared_ptr<RandVar>>& randVars,
							const std::vector<VEFactor>& factorsVE,
							const std::vector<VERandVar>& randVarsVE,
							const std::map<int, int>& factIdToIdx)
{
	int bestScoreIdx = 0;
	int bestScore = 1e9;

	for(int rv = 0; rv < randVarsVE.size(); ++rv){
		int score = 0;

		set<pair<int, int>> edges;
		for(int f = 0; f < factorsVE.size(); ++f){
			for(int rv1 = 0; rv1 < factorsVE[f].randVarIds.size(); ++rv1){
				for(int rv2 = rv1 + 1; rv2 < factorsVE[f].randVarIds.size(); ++rv2){
					int i = min(factorsVE[f].randVarIds[rv1], factorsVE[f].randVarIds[rv2]);
					int j = max(factorsVE[f].randVarIds[rv1], factorsVE[f].randVarIds[rv2]);
//								cout << "Adding edge (" << i << ", " << j << ")" << endl;
					edges.insert(make_pair(i, j));
				}
			}
		}
//					cout << "edges.size() = " << edges.size() << endl;

		set<int> connRvs;
		for(int f = 0; f < randVarsVE[rv].factorClustIds.size(); ++f){
			int curFactIdx = factIdToIdx.at(randVarsVE[rv].factorClustIds[f]);
			for(int nhrv = 0; nhrv < factorsVE[curFactIdx].randVarIds.size(); ++nhrv){
				if(factorsVE[curFactIdx].randVarIds[nhrv] != randVarsVE[rv].randVarId){
					connRvs.insert(factorsVE[curFactIdx].randVarIds[nhrv]);
				}
			}
		}
		for(auto it1 = connRvs.begin(); it1 != connRvs.end(); ++it1){
			auto it2start = it1;
			for(auto it2 = ++it2start; it2 != connRvs.end(); ++it2){
				int i = min(*it1, *it2);
				int j = max(*it1, *it2);
//							cout << "Testing edge (" << i << ", " << j << ")" << endl;
				pair<int, int> potNewEdge = make_pair(i, j);
				if(edges.count(potNewEdge) == 0){
					edges.insert(potNewEdge);
					int cardi = randVars[i]->vals().size();
					int cardj = randVars[j]->vals().size();
//								cout << "score += " << cardi << " * " << cardj << endl;
					score += cardi * cardj;
				}
			}
		}
//					cout << "bestScore = " << bestScore << endl;
//					cout << "bestScoreIdx = " << bestScoreIdx << endl;
//					cout << "score for rvId " << randVars[rv].randVarId << " = " << score << endl;
		if(score < bestScore){
			bestScore = score;
			bestScoreIdx = rv;
		}
	}

//					cout << "Eliminating random variable " << randVars[bestScoreIdx].randVarId << endl;
//					cout << "randVars[bestScoreIdx].factorClustIds = " << randVars[bestScoreIdx].factorClustIds << endl;
	return bestScoreIdx;
}




//----------------Pgm----------------

Pgm::Pgm(std::vector<std::shared_ptr<RandVar>> irandVars,
		std::vector<std::shared_ptr<Cluster>> iclusters,
		std::vector<std::shared_ptr<Feature>> ifeats) :
	randVarsData(irandVars),
	clustersData(iclusters),
	featsData(ifeats)
{

}

Pgm::~Pgm(){
//	cout << "~Pgm()" << endl;
//	cout << "randVarsData.size() = " << randVarsData.size() << endl;
//	if(randVarsData.size() > 0){
//		cout << "randVarsData.front().use_count()" << randVarsData.front().use_count() << endl;
//	}
//	if(clustersData.size() > 0){
//		cout << "clustersData.front().use_count()" << clustersData.front().use_count() << endl;
//	}
//	if(featsData.size() > 0){
//		cout << "featsData.front().use_count()" << featsData.front().use_count() << endl;
//	}
}

bool Pgm::incVarValIdxs(std::vector<int>& varValIdxs,
					const std::vector<std::shared_ptr<RandVar>>& vars)
{
	//iterate through all combinations of variables
	int pos = 0;
	bool carry = true;
	while(pos < (int)vars.size() && carry == true){
		carry = false;
		varValIdxs[pos]++;
		if(varValIdxs[pos] >= (int)vars[pos]->vals().size()){
			varValIdxs[pos] = 0;
			pos++;
			carry = true;
		}
	}
	if(pos == (int)vars.size()){
		return false;
	}
	else{
		return true;
	}
}

void Pgm::addEdgeToPgm(std::shared_ptr<Cluster> a,
					std::shared_ptr<Cluster> b,
					std::vector<std::shared_ptr<RandVar>> sepset /* id sorted */)
{
//	if(a->id() == 29778 || b->id() == 29778){
//		cout << "adding edge from " << a->id() << " to " << b->id() << endl;
//	}
	vector<std::weak_ptr<Cluster>> aNh = a->nh();
	auto it = upper_bound(aNh.begin(), aNh.end(), b, compIdClust);
	int pos = it - aNh.begin();
//	cout << "inseting b at pos " << pos << ", aNh.size() = " << aNh.size() << endl;
	aNh.insert(it, b);
	a->setNh(aNh);

	vector<vector<std::shared_ptr<RandVar>> > aSepsets = a->sepsets();
//	cout << "inseting sepset at pos " << pos << ", aSepsets.size() = " << aSepsets.size() << endl;
	aSepsets.insert(aSepsets.begin() + pos, sepset);
	a->setSepsets(aSepsets);

	vector<std::weak_ptr<Cluster>> bNh = b->nh();
	it = upper_bound(bNh.begin(), bNh.end(), a, compIdClust);
	pos = it - bNh.begin();
//	cout << "inseting a at pos " << pos << ", bNh.size() = " << bNh.size() << endl;
	bNh.insert(it, a);
//	cout << "end inserting" << endl;
	b->setNh(bNh);

//	cout << "sep" << endl;
	vector<vector<std::shared_ptr<RandVar>> > bSepsets = b->sepsets();
//	cout << "inseting sepset at pos " << pos << ", bSepsets.size() = " << bSepsets.size() << endl;
	bSepsets.insert(bSepsets.begin() + pos, sepset);
	b->setSepsets(bSepsets);
}

//template<class T>
//bool Pgm::incVarValIdxs(std::vector<int>& varValIdxs,
//					const std::vector<std::vector<T> >& vals)
//{
//	//iterate through all combinations of variables
//	int pos = 0;
//	bool carry = true;
//	while(pos < vals.size() && carry == true){
//		carry = false;
//		varValIdxs[pos]++;
//		if(varValIdxs[pos] >= vals[pos].size()){
//			varValIdxs[pos] = 0;
//			pos++;
//			carry = true;
//		}
//	}
//	if(pos == vals.size()){
//		return false;
//	}
//	else{
//		return true;
//	}
//}

//void Pgm::deleteContents(){
//	for(int c = 0; c < (int)clustersData.size(); ++c){
//		delete clustersData[c];
//	}
////	for(int f = 0; f < (int)featsData.size(); f++){
////		delete featsData[f];
////	}
//	for(int rv = 0; rv < (int)randVarsData.size(); rv++){
//		delete randVarsData[rv];
//	}
//}


