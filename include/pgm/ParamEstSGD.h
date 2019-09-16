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

#ifndef PARAMESTSGD_H_
#define PARAMESTSGD_H_

#include <vector>

#include "pgm/Pgm.h"

class PgmCreator{
public:
	PgmCreator(int im) : m(im) {}

	virtual ~PgmCreator() {}

	int getM(){
		return m;
	}

	virtual void create(int idx,
						Pgm& pgm,
						std::vector<double>& iobsVec,
						std::vector<double>& ivarVals,
						std::vector<int>& ivarIds) = 0;
protected:
	int m;
};

class ParamEstSGD{
public:
	struct Params{
		double valSetFrac;

		double learnRateM0;

		double momentum;
		// number of graphs taken for one iteration
		int n;
		// sigma used in regularization
		double sigma;
	};

//	typedef void (*createFuncPtr)(int idx,
//									Pgm& pgm,
//									const std::vector<double>& iobsVec,
//									const std::vector<double>& ivarVals,
//									const std::vector<double>& ivarIds,
//									void* arg);

	ParamEstSGD();

	static std::vector<double> estimate(std::shared_ptr<PgmCreator> pgmCreator,
										Params estParams,
										const std::vector<double>& initParams);
private:

	class EvaluateThread{
	public:
		EvaluateThread(std::shared_ptr<Pgm> ipgm,
					std::shared_ptr<const std::vector<double>> iobsVec,
					std::shared_ptr<const std::vector<double>> ivarVals,
					std::shared_ptr<const std::vector<int>> ivarIds,
					std::shared_ptr<const std::vector<double>> iparamVals,
					double& ilhoodNumer,
					double& ilhoodDenom,
					std::vector<double>& iEfi,
					std::vector<double>& iEd);

		~EvaluateThread();

		bool hasFinished();
	private:
		std::shared_ptr<Pgm> pgm;
		std::shared_ptr<const std::vector<double>> obsVec;
		std::shared_ptr<const std::vector<double>> varVals;
		std::shared_ptr<const std::vector<int>> varIds;
		std::shared_ptr<const std::vector<double>> paramVals;
		double& lhoodNumer;
		double& lhoodDenom;
		std::vector<double>& Efi;
		std::vector<double>& Ed;

		std::thread runThread;

		bool finishedFlag;

		void run();
	};

	static void evaluateSamples(const std::vector<int>& sampIdxs,
								int maxThreads,
								std::shared_ptr<PgmCreator> pgmCreator,
								const std::vector<double>& curParams,
								std::vector<double>& lhoodNumers,
								std::vector<double>& lhoodDenoms,
								std::vector<std::vector<double>>& Efis,
								std::vector<std::vector<double>>& Eds);

	static void endThreadIfAny(int& threadCnt,
							std::vector<std::shared_ptr<ParamEstSGD::EvaluateThread>>& threads,
							std::vector<std::shared_ptr<Pgm>>& pgms,
							std::vector<std::shared_ptr<std::vector<double>>>& obsVecs,
							std::vector<std::shared_ptr<std::vector<double>>>& varVals,
							std::vector<std::shared_ptr<std::vector<int>>>& varIds,
							std::vector<std::shared_ptr<std::vector<double>>>& paramVals);
};


#endif /* PARAMESTSGD_H_ */
