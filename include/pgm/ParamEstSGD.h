/*
 * ParamEstSGD.h
 *
 *  Created on: 10 cze 2016
 *      Author: jachu
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
