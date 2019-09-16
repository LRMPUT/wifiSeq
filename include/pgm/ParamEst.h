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

#ifndef PARAMEST_H_
#define PARAMEST_H_

#include <vector>
#include <thread>

#include "pgm/Pgm.h"

class ParamEst{
	std::vector<Pgm>* pgm;
	const std::vector<std::vector<double> >* varVals;
	const std::vector<std::vector<double> >* obsVec;
	const std::vector<int>* paramMap;
	std::vector<double> params;

public:
	ParamEst();

	void evaluate(const std::vector<double>& paramValsMapped,
					double& likelihood,
					std::vector<double>& grad);

	void estimateParams(std::vector<Pgm>& curPgm,
						const std::vector<std::vector<double> >& curVarVals,
						const std::vector<std::vector<double> >& curObsVec = std::vector<std::vector<double> >(),
						const std::vector<int>& curParamMap = std::vector<int>());
};

class EvaluateThread{
	Pgm& curPgm;
	const std::vector<double>& curObsVec;
	const std::vector<double>& curVarVals;
	const std::vector<double>& paramVals;
	double& logPartFunc;
	double& likelihoodNum;
	std::vector<double>& Efi;
	std::vector<double>& Ed;
	std::vector<std::vector<std::vector<double> > >& msgs;
	bool separateMarg;

	std::thread runThread;
	bool finished;

	void run();

public:
	EvaluateThread(Pgm& icurPgm,
					const std::vector<double>& icurObsVec,
					const std::vector<double>& icurVarVals,
					const std::vector<double>& iparamVals,
					double& ilogPartFunc,
					double& ilikelihoodNum,
					std::vector<double>& iEfi,
					std::vector<double>& iEd,
					std::vector<std::vector<std::vector<double> > >& imsgs,
					bool iseparateMarg);

	~EvaluateThread();

	bool hasFinished();

};

#endif /* PARAMEST_H_ */
