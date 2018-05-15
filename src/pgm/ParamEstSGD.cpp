/*
 * ParamEstSGD.cpp
 *
 *  Created on: 10 cze 2016
 *      Author: jachu
 */

#include <iostream>
#include <thread>
#include <random>
#include <chrono>
#include <string>
#include <fstream>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "pgm/ParamEstSGD.h"
#include "pgm/Exceptions.h"
#include "pgm/Inference.h"

using namespace std;

std::vector<double> ParamEstSGD::estimate(std::shared_ptr<PgmCreator> pgmCreator,
										Params estParams,
										const std::vector<double>& initParams)
{
	static const int maxThreads = 4;

	std::default_random_engine randGen;

	int m = pgmCreator->getM();

	//generate random permutation
	vector<int> perm;
	for(int i = 0; i < m; ++i){
		perm.push_back(i);
	}
	for(int i = 0; i < m - 1; ++i){
		std::uniform_int_distribution<int> dist(i, m - 1);
		int swapIdx = dist(randGen);
		int tmp = perm[i];
		perm[i] = perm[swapIdx];
		perm[swapIdx] = tmp;
	}

	int mVal = estParams.valSetFrac * perm.size();
	vector<int> permVal = vector<int>(perm.begin(), perm.begin() + mVal);

	int mTrain = perm.size() - mVal;
	vector<int> permTrain = vector<int>(perm.begin() + mVal, perm.end());

	vector<double> curParams = initParams;

	vector<double> paramsGrad;

	int iter = 0;
	bool stopFlag = false;
	while(!stopFlag){
		cout << "iteration " << iter << endl;
		cout << "curParams = " << curParams << endl;

		double curLearnRate = 1 / ((estParams.sigma * estParams.sigma) * (estParams.learnRateM0 + iter));
		cout << "curLearnRate = " << curLearnRate << endl;

		//generate random permutation
		vector<int> curPerm = permTrain;

		for(int i = 0; i < mTrain - 1; ++i){
			std::uniform_int_distribution<int> dist(i, mTrain - 1);
			int swapIdx = dist(randGen);
			int tmp = curPerm[i];
			curPerm[i] = curPerm[swapIdx];
			curPerm[swapIdx] = tmp;
		}

		vector<double> lhoodNumers;
		vector<double> lhoodDenoms;
		vector<vector<double>> Efis;
		vector<vector<double>> Eds;

		//take first n samples
		evaluateSamples(vector<int>(curPerm.begin(), curPerm.begin() + estParams.n),
						maxThreads,
						pgmCreator,
						curParams,
						lhoodNumers,
						lhoodDenoms,
						Efis,
						Eds);

		vector<double> curGrad(curParams.size(), 0.0);
		// compute current gradient
		for(int i = 0; i < Efis.size(); ++i){
			for(int p = 0; p < curGrad.size(); ++p){
				curGrad[p] = Eds[i][p] - Efis[i][p];
			}
		}
		for(int p = 0; p < curGrad.size(); ++p){
			curGrad[p] /= Efis.size();
		}
		for(int p = 0; p < curGrad.size(); ++p){
			curGrad[p] -= curParams[p] / (estParams.sigma * estParams.sigma);
		}

		cout << "curGrad = " << curGrad << endl;

		// if it is first iteration and previous gradient is empty
		if(paramsGrad.empty()){
			paramsGrad = curGrad;
		}

		for(int p = 0; p < paramsGrad.size(); ++p){
			paramsGrad[p] = curLearnRate * (paramsGrad[p]*estParams.momentum + curGrad[p]);
		}

		for(int p = 0; p < curParams.size(); ++p){
			curParams[p] += paramsGrad[p];
		}


		//evaluate on validation set
		if(iter % 10 == 0){
			double scoreVal = 0.0;

			vector<double> lhoodNumers;
			vector<double> lhoodDenoms;
			vector<vector<double>> Efis;
			vector<vector<double>> Eds;

			//take first n samples
			evaluateSamples(permVal,
							maxThreads,
							pgmCreator,
							curParams,
							lhoodNumers,
							lhoodDenoms,
							Efis,
							Eds);

			for(int i = 0; i < lhoodNumers.size(); ++i){
				cout << "scoreVal += " << lhoodNumers[i] << " - " << lhoodDenoms[i] << endl;
				scoreVal += lhoodNumers[i] - lhoodDenoms[i];
			}
			scoreVal /= lhoodNumers.size();

			cout << endl << endl << "scoreVal = " << scoreVal << endl << endl << endl;

			//saving a snapshot
			{
				ofstream paramsFile(string("log/params_") + to_string(iter));
				boost::archive::text_oarchive paramsArch(paramsFile);
				paramsArch << curParams;
			}
			{
				ofstream gradFile(string("log/grad_") + to_string(iter));
				boost::archive::text_oarchive gradArch(gradFile);
				gradArch << curGrad;
			}
			{
				ofstream scoreFile(string("log/score"), ios::app);
				scoreFile << iter << " " << scoreVal << endl;
			}
		}

		//stop condition
		//TODO use better stop condition
		if(curLearnRate < 0.005){
			stopFlag = true;
		}

		iter++;
	}

	return curParams;
}

void ParamEstSGD::evaluateSamples(const std::vector<int>& sampIdxs,
							int maxThreads,
							std::shared_ptr<PgmCreator> pgmCreator,
							const std::vector<double>& curParams,
							std::vector<double>& lhoodNumers,
							std::vector<double>& lhoodDenoms,
							std::vector<std::vector<double>>& Efis,
							std::vector<std::vector<double>>& Eds)
{
	int threadCnt = 0;
	vector<shared_ptr<ParamEstSGD::EvaluateThread>> threads(sampIdxs.size());
	vector<shared_ptr<Pgm>> pgms(sampIdxs.size());
	vector<shared_ptr<vector<double>>> obsVecs(sampIdxs.size());
	vector<shared_ptr<vector<double>>> varVals(sampIdxs.size());
	vector<shared_ptr<vector<int>>> varIds(sampIdxs.size());
	vector<shared_ptr<vector<double>>> paramVals(sampIdxs.size());
	lhoodNumers.resize(sampIdxs.size());
	lhoodDenoms.resize(sampIdxs.size());
	Efis.resize(sampIdxs.size());
	Eds.resize(sampIdxs.size());

	//take first n elements from permutation
	for(int i = 0; i < sampIdxs.size(); ++i){
		while(threadCnt == maxThreads){
			endThreadIfAny(threadCnt,
							threads,
							pgms,
							obsVecs,
							varVals,
							varIds,
							paramVals);

			std::this_thread::sleep_for(chrono::milliseconds(50));
		}

		pgms[i] = shared_ptr<Pgm>(new Pgm);
		obsVecs[i] = shared_ptr<vector<double>>(new vector<double>());
		varVals[i] = shared_ptr<vector<double>>(new vector<double>());
		varIds[i] = shared_ptr<vector<int>>(new vector<int>());
		paramVals[i] = shared_ptr<vector<double>>(new vector<double>(curParams));
		lhoodNumers[i] = 0.0;
		lhoodDenoms[i] = 0.0;
		Efis[i] = vector<double>(curParams.size(), 0.0);
		Eds[i] = vector<double>(curParams.size(), 0.0);

		pgmCreator->create(sampIdxs[i],
						*pgms[i],
						*obsVecs[i],
						*varVals[i],
						*varIds[i]);

//		for(int j = 0; j < Efis.size(); ++j){
//			cout << "Efis[" << j << "].data() = " << Efis[j].data() << endl;
//			cout << "&Efis[" << j << "] = " << &Efis[j] << endl;
//		}

		threads[i] = shared_ptr<EvaluateThread>(new EvaluateThread(pgms[i],
												obsVecs[i],
												varVals[i],
												varIds[i],
												paramVals[i],
												lhoodNumers[i],
												lhoodDenoms[i],
												Efis[i],
												Eds[i]));
		++threadCnt;
	}

	while(threadCnt > 0){
		endThreadIfAny(threadCnt,
						threads,
						pgms,
						obsVecs,
						varVals,
						varIds,
						paramVals);
		std::this_thread::sleep_for(chrono::milliseconds(50));
	}
}


void ParamEstSGD::endThreadIfAny(int& threadCnt,
						std::vector<std::shared_ptr<ParamEstSGD::EvaluateThread>>& threads,
						std::vector<std::shared_ptr<Pgm>>& pgms,
						std::vector<std::shared_ptr<std::vector<double>>>& obsVecs,
						std::vector<std::shared_ptr<std::vector<double>>>& varVals,
						std::vector<std::shared_ptr<std::vector<int>>>& varIds,
						std::vector<std::shared_ptr<std::vector<double>>>& paramVals)
{
	for(int t = 0; t < threads.size(); ++t){
		if(threads[t]){
			if(threads[t]->hasFinished()){
				cout << "Ending thread " << t << endl;
				threads[t].reset();
				pgms[t].reset();
				obsVecs[t].reset();
				varVals[t].reset();
				varIds[t].reset();
				paramVals[t].reset();

				--threadCnt;
			}
		}
	}
}


ParamEstSGD::EvaluateThread::EvaluateThread(std::shared_ptr<Pgm> ipgm,
										std::shared_ptr<const std::vector<double>> iobsVec,
										std::shared_ptr<const std::vector<double>> ivarVals,
										std::shared_ptr<const std::vector<int>> ivarIds,
										std::shared_ptr<const std::vector<double>> iparamVals,
										double& ilhoodNumer,
										double& ilhoodDenom,
										std::vector<double>& iEfi,
										std::vector<double>& iEd)
		: pgm(ipgm),
		  obsVec(iobsVec),
		  varVals(ivarVals),
		  varIds(ivarIds),
		  paramVals(iparamVals),
		  lhoodNumer(ilhoodNumer),
		  lhoodDenom(ilhoodDenom),
		  Efi(iEfi),
		  Ed(iEd),
		  finishedFlag(false)
{
//	cout << "Efi.data() constr = " << Efi.data() << ", this = " << this << endl;
//	cout << "&Efi constr = " << &Efi << ", this = " << this << endl;
	runThread = thread(&ParamEstSGD::EvaluateThread::run, this);
}

ParamEstSGD::EvaluateThread::~EvaluateThread()
{
	if(runThread.joinable()){
		runThread.join();
	}
}

bool ParamEstSGD::EvaluateThread::hasFinished()
{
	return finishedFlag;
}

void ParamEstSGD::EvaluateThread::run()
{
	// inference without clamped variables - denominator and model expectation
	{
//		cout << "Efi.data() 1 = " << Efi.data() << ", this = " << this <<  endl;
//		cout << "&Efi 1 = " << &Efi << ", this = " << this << endl;

		vector<vector<double>> marg;
		vector<vector<vector<double>>> msgs;
		double logPartFunc;

//		cout << "Efi.data() 2 = " << Efi.data() << ", this = " << this << endl;
		bool calibrated = Inference::compMarginalsParam(*pgm, marg, logPartFunc, msgs, *paramVals, *obsVec);
//		cout << "Efi.data() 3 = " << Efi.data() << endl;

		if(!calibrated){
			cout << "Warning - inference without clamped variables didn't calibrate" << endl;
		}

		lhoodDenom = logPartFunc;
		cout << "loodDenom = " << lhoodDenom << endl;

		for(int c = 0; c < (int)pgm->constClusters().size(); ++c){
//			cout << "Cluster " << c << endl;
			std::shared_ptr<Cluster> curCluster = pgm->clusters()[c];
//			cout << "curCluster->id() = " << curCluster->id() << endl;
			const vector<vector<double> >& curClustMsgs = msgs[curCluster->id()];
//			cout << "curClustMsgs = " << curClustMsgs << endl;
			vector<double> curEfi = curCluster->compSumModelExpectation(curClustMsgs, *paramVals, *obsVec);
//			cout << "Efi.size() = " << Efi.size() << ", curEfi.size() = " << curEfi.size() << endl;

			for(int f = 0; f < (int)curCluster->feats().size(); ++f){
//				cout << "f = " << f << endl;
				int paramNum = curCluster->feats()[f]->paramNum();
//				cout << "paramNum = " << paramNum << endl;
				Efi[paramNum] += curEfi[f];
			}
		}
	}

	// clamping variables
//	cout << "clamping variables" << endl;
	int posRv = 0;
	for(int v = 0; v < varIds->size(); ++v){
		bool stopFlag = false;
		while(!stopFlag && posRv < pgm->randVars().size() - 1){
			if(pgm->randVars()[posRv]->id() < varIds->at(v)){
				++posRv;
			}
			else{
				stopFlag = true;
			}
		}
		if(pgm->randVars()[posRv]->id() == varIds->at(v)){
			pgm->randVars()[posRv]->makeObserved(varVals->at(v));
		}
		else{
			cout << "pgm->randVars()[posRv]->id() = " << pgm->randVars()[posRv]->id() << endl;
			cout << "varIds->at(v) = " << varIds->at(v) << endl;
			throw PGM_EXCEPTION("Rand var not found");
		}
	}
//	cout << "end clamping variables" << endl;

	// inference with clamped variables - numerator and distribution expectation
	{
		vector<vector<double>> margClamp;
		vector<vector<vector<double>>> msgsClamp;
		double logPartFuncClamp;

		bool calibratedClamp = Inference::compMarginalsParam(*pgm, margClamp, logPartFuncClamp, msgsClamp, *paramVals, *obsVec);

		if(!calibratedClamp){
			cout << "Warning - inference with clamped variables didn't calibrate" << endl;
		}

		lhoodNumer = logPartFuncClamp;
		cout << "loodNumer = " << lhoodNumer << endl;

		// a model expectation with clamped variables is a distribution expectation
		for(int c = 0; c < (int)pgm->constClusters().size(); ++c){
//			cout << "Cluster " << c << endl;
			std::shared_ptr<Cluster> curCluster = pgm->clusters()[c];
			const vector<vector<double> >& curClustMsgs = msgsClamp[curCluster->id()];
			vector<double> curEfi = curCluster->compSumModelExpectation(curClustMsgs, *paramVals, *obsVec);

			for(int f = 0; f < (int)curCluster->feats().size(); ++f){
				int paramNum = curCluster->feats()[f]->paramNum();
				Ed[paramNum] += curEfi[f];
			}
		}
	}

	finishedFlag = true;
}

