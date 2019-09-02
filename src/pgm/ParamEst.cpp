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

#include "pgm/ParamEst.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <queue>
#include <memory>

#include <lbfgs.h>

#include "pgm//Inference.h"

using namespace std;


void EvaluateThread::run()
{
	std::vector<std::vector<double> > marg;

	//For CRF one marginalization per data instance
	if(separateMarg){
		cout << "marginalization" << endl;

		msgs.clear();
		double curLogPartFunc = 0.0;

		Inference::compMarginalsParam(curPgm,
										marg,
										curLogPartFunc,
										msgs,
										paramVals,
										curObsVec);

		logPartFunc += curLogPartFunc;
	}

	//Compute likelihood
//	cout << "Computing likelihood" << endl;
	for(int c = 0; c < (int)curPgm.constClusters().size(); ++c){
		const std::shared_ptr<Cluster> curCluster = curPgm.constClusters()[c];
		vector<double> clustVarVals(curCluster->randVars().size(), 0);
		for(int v = 0; v < (int)curCluster->randVars().size(); ++v){
			clustVarVals[v] = curVarVals[curCluster->randVars()[v]->id()];
		}

		//Only exponent of feature vals - MaxSum
		double factVal = curCluster->compFactorsVal(clustVarVals, MaxSum, paramVals, curObsVec);
		likelihoodNum += factVal;
	}

	//Compute gradients
//		cout << "Computing gradients" << endl;
	//Empirical expectation
//	cout << "Empirical expectation" << endl;
	for(int c = 0; c < (int)curPgm.constClusters().size(); ++c){
		const std::shared_ptr<Cluster> curCluster = curPgm.constClusters()[c];
		vector<double> clustVarVals(curCluster->randVars().size(), 0);
		for(int v = 0; v < (int)curCluster->randVars().size(); ++v){
			clustVarVals[v] = curVarVals[curCluster->randVars()[v]->id()];
		}
//		cout << "cluster " << c << endl;
//		cout << "clustVarVals = " << clustVarVals << endl;
//		cout << "curObsVec = " << curObsVec << endl;
		vector<double> curEd;
		curEd = curCluster->compSumEmpiricalExpectation(clustVarVals, curObsVec);

//			cout << "Ed for cluster " << c << " = " << curEd << endl;
		for(int f = 0; f < (int)curCluster->feats().size(); ++f){
			int paramNum = curCluster->feats()[f]->paramNum();
			Ed[paramNum] += curEd[f];
		}
	}

	if(separateMarg){
		//Model expectation
//		cout << "Model expectation CRF" << endl;
		for(int c = 0; c < (int)curPgm.constClusters().size(); ++c){
//				cout << "Cluster " << c << endl;
			std::shared_ptr<Cluster> curCluster = curPgm.clusters()[c];
			const vector<vector<double> >& curClustMsgs = msgs[curCluster->id()];
			vector<double> curEfi = curCluster->compSumModelExpectation(curClustMsgs, paramVals, curObsVec);

//				bool stop = false;
//				cout << "Efi for cluster " << c << " = " << curEfi << endl;
//				cout << "paramNums = [";
			for(int f = 0; f < (int)curCluster->feats().size(); ++f){
				int paramNum = curCluster->feats()[f]->paramNum();
//					cout << paramNum << ", ";
//					if(paramNum == 20){
//						stop = true;
//					}
				Efi[paramNum] += curEfi[f];
			}
//				cout << endl;

//				if(stop){
//					char a;
//					cin >> a;
//				}


		}
	}

	finished = true;
}


EvaluateThread::EvaluateThread(Pgm& icurPgm,
				const std::vector<double>& icurObsVec,
				const std::vector<double>& icurVarVals,
				const std::vector<double>& iparamVals,
				double& ilogPartFunc,
				double& ilikelihoodNum,
				std::vector<double>& iEfi,
				std::vector<double>& iEd,
				std::vector<std::vector<std::vector<double> > >& imsgs,
				bool iseparateMarg)
	:
		curPgm(icurPgm),
		curObsVec(icurObsVec),
		curVarVals(icurVarVals),
		paramVals(iparamVals),
		logPartFunc(ilogPartFunc),
		likelihoodNum(ilikelihoodNum),
		Efi(iEfi),
		Ed(iEd),
		msgs(imsgs),
		separateMarg(iseparateMarg),
		finished(false)
{
	runThread = thread(&EvaluateThread::run, this);
}

EvaluateThread::~EvaluateThread(){
	runThread.join();
}

bool EvaluateThread::hasFinished(){
	return finished;
}


static lbfgsfloatval_t evaluate(void *instance,
								const lbfgsfloatval_t *theta,
								lbfgsfloatval_t *g,
								const int n,
								const lbfgsfloatval_t step)
{
	lbfgsfloatval_t ftheta = 0.0;
	double likelihood;
	vector<double> grad;

	//Pack values to vector
	vector<double> paramVals(n, 0.0);
	for(int p = 0; p < n; ++p){
		paramVals[p] = theta[p];
	}


	cout << "step = " << step << endl;
	//Call (ParamEst*)(instance)->evaluate()
	((ParamEst*)(instance))->evaluate(paramVals, likelihood, grad);

	//Unpack values from vector
	ftheta = -likelihood;
	for(int p = 0; p < n; ++p){
		g[p] = -grad[p];
	}

	return ftheta;
}

static int progress(void *instance,
					const lbfgsfloatval_t *theta,
					const lbfgsfloatval_t *g,
					const lbfgsfloatval_t ftheta,
					const lbfgsfloatval_t xnorm,
					const lbfgsfloatval_t gnorm,
					const lbfgsfloatval_t step,
					int n,
					int k,
					int ls)
{
	printf("Iteration %d:\n", k);
	printf(" ftheta = %f, theta[0] = %f, theta[1] = %f\n", ftheta, theta[0], theta[1]);
	printf(" xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
	printf("\n");

	ofstream paramsCacheFile("cache/params_write.cache");
	for(int p = 0; p < n; ++p){
		paramsCacheFile << theta[p] << endl;
	}
	paramsCacheFile.close();

	return 0;
}

ParamEst::ParamEst() :
		pgm(NULL),
		varVals(NULL),
		obsVec(NULL),
		paramMap(NULL)
{

}

void ParamEst::evaluate(const std::vector<double>& paramValsMapped,
						double& likelihood,
						std::vector<double>& grad)
{

	cout << "evaluate()" << endl;

	vector<double> paramVals;
	if(paramMap->empty()){
		paramVals = paramValsMapped;
	}
	else{
		paramVals = params;
		for(int p = 0; p < int(paramMap->size()); ++p){
			paramVals[paramMap->at(p)] = paramValsMapped[p];
		}
	}

	bool separateMarg;
	if(obsVec->empty() && pgm->size() == 1){
		separateMarg = false;
	}
	else{
		separateMarg = true;
	}

	vector<double> logPartFuncAll(varVals->size(), 0);
	std::vector<std::vector<std::vector<std::vector<double> > > > msgs(varVals->size());

	std::vector<std::vector<double> > marg;

	//No observation vector and common graph - one marginalization for a whole dataset
	if(!separateMarg){
		Inference::compMarginalsParam(pgm->front(),
										marg,
										logPartFuncAll.front(),
										msgs.front(),
										paramVals);
	}

	std::vector<double> likelihoodNumAll(varVals->size(), 0.0);
	std::vector<vector<double> > EfiAll(varVals->size(), vector<double>(paramVals.size(), 0.0));
	std::vector<vector<double> > EdAll(varVals->size(), vector<double>(paramVals.size(), 0.0));

	static const int maxThreads = 4;

	vector<std::shared_ptr<EvaluateThread>> curEval(maxThreads, std::shared_ptr<EvaluateThread>());
	int nextDataInst = 0;
	int numThreads = 0;

	do{
		if(numThreads < maxThreads && nextDataInst < (int)varVals->size()){
			//spawn new thread
			for(int t = 0; t < (int)curEval.size(); ++t){
				//if empty
				if(!curEval[t]){
					cout << "Spawning new thread at curEval[" << t << "] for data instance " << nextDataInst << endl;
					int pgmIdx = min(nextDataInst, (int)pgm->size() - 1);
					Pgm& curPgm = (*pgm)[pgmIdx];
					const vector<double>& curVarVals = (*varVals)[nextDataInst];

					if(obsVec->empty()){
						curEval[t] = std::shared_ptr<EvaluateThread>(new EvaluateThread(curPgm,
														vector<double>(),
														curVarVals,
														paramVals,
														logPartFuncAll[nextDataInst],
														likelihoodNumAll[nextDataInst],
														EfiAll[nextDataInst],
														EdAll[nextDataInst],
														msgs[nextDataInst],
														separateMarg));
					}
					else{
//						cout << "varVals = " << (*varVals)[nextDataInst] << endl;
//						cout << "obsVec = " << (*obsVec)[nextDataInst] << endl;
						const vector<double>& curObsVec = (*obsVec)[nextDataInst];

						curEval[t] = std::shared_ptr<EvaluateThread>(new EvaluateThread(curPgm,
														curObsVec,
														curVarVals,
														paramVals,
														logPartFuncAll[nextDataInst],
														likelihoodNumAll[nextDataInst],
														EfiAll[nextDataInst],
														EdAll[nextDataInst],
														msgs[nextDataInst],
														separateMarg));
					}
					++numThreads;
					++nextDataInst;
					break;
				}
			}
		}

		for(int t = 0; t < (int)curEval.size(); ++t){
			//if not empty
			if(curEval[t]){
				if(curEval[t]->hasFinished()){
					cout << "thread at curEval[" << t << "] has finished" << endl;

					curEval[t].reset();
					--numThreads;
				}
			}
		}

		this_thread::sleep_for (std::chrono::milliseconds(50));

	}while(numThreads > 0);

	double likelihoodNum = 0;
	double logPartFunc = 0;
	vector<double> Efi(paramVals.size(), 0.0);
	vector<double> Ed(paramVals.size(), 0.0);

	for(int d = 0; d < (int)varVals->size(); ++d){
		likelihoodNum += likelihoodNumAll[d];

		if(separateMarg){
			logPartFunc += logPartFuncAll[d];
			for(int p = 0; p < (int)paramVals.size(); ++p){
				Efi[p] += EfiAll[d][p];
			}
		}

		for(int p = 0; p < (int)paramVals.size(); ++p){
			Ed[p] += EdAll[d][p];
		}
	}


	likelihood = 0.0;

	//Compute likelihood

	likelihoodNum /= varVals->size();

	//for CRFs log part function is different for every data instance
	if(separateMarg){
		logPartFunc /=  varVals->size();
	}

	//Regularization
	static const double sigma = 4.0;
	double regPenalty = 0.0;
	for(int p = 0; p < (int)paramVals.size(); ++p){
		regPenalty += paramVals[p] * paramVals[p] / (2 * sigma * sigma);
	}


	cout << "likelihoodNum = " << likelihoodNum <<
			", logPartFunc = " << logPartFunc <<
			", regPenalty = " << regPenalty << endl;

	likelihood = likelihoodNum - logPartFunc - regPenalty;

	//Compute gradients

	for(int p = 0; p < (int)paramVals.size(); ++p){
		Ed[p] /= varVals->size();
	}

	if(!separateMarg){
		int pgmIdx = 0;
		Pgm& curPgm = (*pgm)[pgmIdx];

		//Model expectation
		cout << "Model expectation MRF" << endl;

		for(int c = 0; c < (int)curPgm.constClusters().size(); ++c){
			std::shared_ptr<Cluster> curCluster = curPgm.clusters()[c];
			const vector<vector<double> >& curClustMsgs = msgs.front()[curCluster->id()];
			vector<double> curEfi = curCluster->compSumModelExpectation(curClustMsgs, paramVals, vector<double>());

			for(int f = 0; f < (int)curCluster->feats().size(); ++f){
				int paramNum = curCluster->feats()[f]->paramNum();
				Efi[paramNum] += curEfi[f];
			}
		}
	}
	else{
		//for CRF devide by number of data instances
		for(int p = 0; p < (int)paramVals.size(); ++p){
			Efi[p] /= varVals->size();
		}
	}

	ofstream edFile("log/Ed.log");
//	cout << "Ed = {";
	for(int p = 0; p < (int)paramVals.size(); ++p){
		edFile << Ed[p] << endl;
	}
//	cout << "}" << endl;

	ofstream efiFile("log/Efi.log");
//	cout << "Efi = {";
	for(int p = 0; p < (int)paramVals.size(); ++p){
		efiFile << Efi[p] << endl;
	}
//	cout << "}" << endl;

	/*cout << "Ed - Efi = {";
	for(int p = 0; p < paramVals.size(); ++p){
		cout << Ed[p] - Efi[p] << "; ";
	}
	cout << "}" << endl;*/

	grad = vector<double>(paramValsMapped.size(), 0.0);

	//Regularization
	if(paramMap->empty()){
		for(int p = 0; p < (int)paramVals.size(); ++p){
			grad[p] = Ed[p] - Efi[p] - (paramVals[p] / (sigma * sigma));
		}
	}
	else{
		for(int p = 0; p < (int)paramValsMapped.size(); ++p){
			grad[p] = Ed[paramMap->at(p)] - Efi[paramMap->at(p)] - (paramVals[paramMap->at(p)] / (sigma * sigma));
		}
	}

	cout << "paramVals = {";
	for(int p = 0; p < (int)paramVals.size(); ++p){
		cout << paramVals[p] << ", ";
	}
	cout << "}" << endl;

	cout << "likelihood = " << likelihood << endl;

	cout << "grad = {";
	for(int p = 0; p < (int)grad.size(); ++p){
		cout << grad[p] << ", ";
	}
	cout << "}" << endl;

//	cout << "press char and enter" << endl;
//	char a;
//	cin >> a;
//	cout << "End evaluate()" << endl;
}

void ParamEst::estimateParams(std::vector<Pgm>& curPgm,
							const std::vector<std::vector<double> >& curVarVals,
							const std::vector<std::vector<double> >& curObsVec,
							const std::vector<int>& curParamMap)
{
	int n = 0;
	if(curParamMap.empty()){
		n = curPgm.front().params().size();
	}
	else{
		n = curParamMap.size();
	}

	lbfgsfloatval_t ftheta;
	lbfgsfloatval_t *theta = lbfgs_malloc(n);
	lbfgs_parameter_t param;
	if (theta == NULL) {
		printf("ERROR: Failed to allocate a memory block for variables.\n");
		exit(-1);
	}
	/* Initialize the variables. */
	if(curParamMap.empty()){
		for(int p = 0; p < (int)curPgm.front().constParams().size(); ++p){
			theta[p] = curPgm.front().constParams()[p];
		}
	}
	else{
		for(int p = 0; p < (int)curParamMap.size(); ++p){
			theta[p] = curPgm.front().constParams()[curParamMap[p]];
		}
	}

	/* Initialize the parameters for the L-BFGS optimization. */
	lbfgs_parameter_init(&param);
	/*param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;*/
//	param.ftol = 1e-2;
//	param.xtol = 1e-5;
	param.epsilon = 1e-3;

	/* Save data for evaluate() function. */
	pgm = &curPgm;
	varVals = &curVarVals;
	obsVec = &curObsVec;
	paramMap = &curParamMap;
	params = curPgm.front().constParams();

	/*
	Start the L-BFGS optimization; this will invoke the callback functions
	evaluate() and progress() when necessary.
	*/
	int ret = lbfgs(n, theta, &ftheta, ::evaluate, ::progress, (void*)this, &param);

	/* Report the result. */
	printf("L-BFGS optimization terminated with status code = %d\n", ret);
	printf(" ftheta = %f\n", ftheta);
	for(int p = 0; p < (int)curPgm.front().params().size(); ++p){
		printf("allPgm[i].params()[%d] = %f;\n", p, theta[p]);
	}
	lbfgs_free(theta);
	if(ret == LBFGSERR_ROUNDING_ERROR){
		cout << "A rounding error occurred; alternatively, no line-search step " <<
				"satisfies the sufficient decrease and curvature conditions." << endl;
	}

	//Copy params to pgm structure
	for(int g = 0; g < (int)curPgm.size(); ++g){
		for(int p = 0; p < (int)curPgm[g].params().size(); ++p){
			curPgm[g].params()[p] = theta[p];
		}
	}
}

