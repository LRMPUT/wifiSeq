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

#ifndef PGM_H_
#define PGM_H_

#include <vector>
#include <ostream>
#include <memory>
#include <map>

extern std::vector<double> featScales;

template<class T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec){
	out << "[";
	for(int v = 0; v < (int)vec.size(); ++v){
		out << vec[v];
		if(v < vec.size() - 1){
			out << ", ";
		}
	}
	out << "]";

	return out;
}

enum MargType{
	SumProduct,
	MaxSum
};


class RandVar{
protected:
	int idData;
	bool isObserved;
	//double observedVal;
	std::vector<double> valsData;
public:
	RandVar(int iid, std::vector<double> ivals);

	RandVar(int iid, double observedVal);

	virtual ~RandVar();

	inline int id() const {
		return idData;
	}

	inline bool isObs() const {
		return isObserved;
	}

	void makeObserved(double val);

	void makeNotObserved(std::vector<double> vals);

	inline const std::vector<double>& vals() const{
		return valsData;
	}

	void setVals(const std::vector<double>& newVals);
};

class Feature{
	int paramNumData;
	int idData;
	std::vector<std::shared_ptr<RandVar>> randVarsOrderedData;
	std::vector<int> obsNumsData;
public:
	Feature(int iid,
			int iparamNum,
			const std::vector<std::shared_ptr<RandVar>>& irandVarsOrdered,
			const std::vector<int>& iobsNums = std::vector<int>());
	virtual ~Feature();

	inline int paramNum() const {
		return paramNumData;
	}

	inline const std::vector<std::shared_ptr<RandVar>>& randVarsOrdered(){
		return randVarsOrderedData;
	}

	inline const std::vector<int>& obsNums(){
		return obsNumsData;
	}

	inline int id() const {
		return idData;
	}

	virtual double comp(const std::vector<double>& vals,
						const std::vector<double>& obsVec = std::vector<double>()) = 0;

	virtual double compParam(const std::vector<double>& vals,
							const std::vector<double>& params,
							const std::vector<double>& obsVec = std::vector<double>()) = 0;
};

class Cluster{
protected:
	int idData;
	//weak pointers to prevent from making reference cycles
	std::vector<std::weak_ptr<Cluster>> nhData;	//id sorted
	std::vector<std::vector<std::shared_ptr<RandVar>>> sepsetsData;	//id sorted
	std::vector<std::shared_ptr<Feature>> featuresData;
	std::vector<std::shared_ptr<RandVar>> randVarsData;	//id sorted
	//const std::vector<double>* paramsData;
	std::vector<std::vector<int>> randVarsOrderData; // for every feature
	std::vector<std::vector<int>> obsVecIdxsData; // for every feature

	void normalizeMarg(std::vector<double>& marg,
						MargType type) const;
public:
	Cluster(int iid,
			const std::vector<std::shared_ptr<Feature>>& ifeats,
			const std::vector<std::shared_ptr<RandVar>>& irandVars /*id sorted*/);

	virtual ~Cluster();

	void setNh(std::vector<std::weak_ptr<Cluster>> newNh /*id sorted*/);

	void setSepsets(std::vector<std::vector<std::shared_ptr<RandVar>>> newSepsets /*id sorted*/);

	inline int id() const {
		return idData;
	}

	virtual double compFactorsVal(const std::vector<double>& varVals,
								MargType type,
								const std::vector<double>& params,
								const std::vector<double>& obsVec = std::vector<double>()) const;

	virtual double compSumHcdiv(const std::vector<std::vector<double> >& inMsgs,
								const std::vector<double>& params,
								const std::vector<double>& obsVec);

	virtual std::vector<double> compSumModelExpectation(const std::vector<std::vector<double> >& inMsgs,
														const std::vector<double>& params,
														const std::vector<double>& obsVec);

	virtual std::vector<double> compSumEmpiricalExpectation(const std::vector<double>& varVals,
														const std::vector<double>& obsVec) const;

	virtual std::vector<std::vector<int> > getBestValsIdxs(const std::vector<std::vector<double> >& inMsgs,
															const std::vector<double>& params,
															const std::vector<double>& obsVec,
															double eps);

	virtual std::vector<double> marginalize(const std::vector<std::shared_ptr<RandVar>>& margVars /*id sorted*/,
									MargType type,
									const std::vector<std::vector<double>>& inMsgs,
									const std::vector<double>& params,
									const std::vector<double>& obsVec = std::vector<double>(),
									const std::shared_ptr<Cluster> excluded = 0);

	inline const std::vector<std::weak_ptr<Cluster>>& nh() const{
		return nhData;
	}

	inline const std::vector<std::shared_ptr<Feature>>& feats() const{
		return featuresData;
	}

	inline const std::vector<std::shared_ptr<RandVar>>& randVars() const{
		return randVarsData;
	}

	inline const std::vector<std::shared_ptr<RandVar>>& sepset(int nhc /*nh index*/) const{
		return sepsetsData[nhc];
	}

	inline const std::vector<std::vector<std::shared_ptr<RandVar>>>& sepsets() const{
		return sepsetsData;
	}

	inline const std::vector<std::vector<int>>& obsVecIdxs() const{
		return obsVecIdxsData;
	}
	inline std::vector<std::vector<double>> getCurObsVecs(const std::vector<double>& obsVec) const {
		std::vector<std::vector<double>> curObsVec(obsVecIdxsData.size());
		if(!obsVec.empty()){
			for(int f = 0; f < obsVecIdxsData.size(); ++f){
				curObsVec[f] = std::vector<double>(obsVecIdxsData[f].size());
				for(int o = 0; o < (int)obsVecIdxsData[f].size(); ++o){
					curObsVec[f][o] = obsVec[obsVecIdxsData[f][o]];
				}
			}
		}
		return curObsVec;
	}
};


class VECluster : public Cluster
{
public:
	VECluster(int iid,
				const std::vector<std::shared_ptr<Feature>>& ifeats,
				const std::vector<std::shared_ptr<RandVar>>& irandVars /*id sorted*/);

	virtual ~VECluster(){}

	virtual double compFactorsVal(const std::vector<double>& varVals,
								MargType type,
								const std::vector<double>& params,
								const std::vector<double>& obsVec = std::vector<double>()) const;

	double compFactorsValOrdered(const std::vector<double>& orderedVarVals,
										MargType type,
										const std::vector<double>& params,
										const std::vector<double>& obsVec) const;

	virtual double compSumHcdiv(const std::vector<std::vector<double> >& inMsgs,
								const std::vector<double>& params,
								const std::vector<double>& obsVec);

	virtual std::vector<double> compSumModelExpectation(const std::vector<std::vector<double> >& inMsgs,
														const std::vector<double>& params,
														const std::vector<double>& obsVec);

	virtual std::vector<double> compSumEmpiricalExpectation(const std::vector<double>& varVals,
														const std::vector<double>& obsVec) const;

	virtual std::vector<std::vector<int> > getBestValsIdxs(const std::vector<std::vector<double> >& inMsgs,
															const std::vector<double>& params,
															const std::vector<double>& obsVec,
															double eps);

	virtual std::vector<double> marginalize(const std::vector<std::shared_ptr<RandVar>>& margVars /*id sorted*/,
										MargType type,
										const std::vector<std::vector<double> >& inMsgs,
										const std::vector<double>& params,
										const std::vector<double>& obsVec = std::vector<double>(),
										const std::shared_ptr<Cluster> excluded = 0);

protected:
	std::vector<std::shared_ptr<Cluster>> intraClusters;
	std::vector<std::shared_ptr<Cluster>> intraRandVarClusters;

	double compNormConstMarg(const std::vector<std::vector<double> >& inMsgs,
							const std::vector<double>& params,
							const std::vector<double>& obsVec) const;
};



class VariableElimination{
public:
	static void buildCliqueTree(const std::vector<std::shared_ptr<Feature>>& feats,
							const std::vector<std::shared_ptr<RandVar>>& randVars /*id sorted*/,
							std::vector<std::shared_ptr<Cluster>>& clusters,
							std::vector<std::shared_ptr<Cluster>>& randVarClusters);
protected:
	struct VEFactor{
		int id;
		std::vector<int> randVarIds;

		VEFactor() {}

		VEFactor(int iid, const std::vector<int>& irandVarIds) :
			id(iid),
			randVarIds(irandVarIds)
		{}
	};

	struct VERandVar{
		int randVarId;
		std::vector<int> factorClustIds;

		VERandVar() {}

		VERandVar(int irandVarId, const std::vector<int>& ifactorClustIds) :
			randVarId(irandVarId),
			factorClustIds(ifactorClustIds)
		{}
	};

	static void makeRandVars(std::vector<VEFactor>& factors,
					std::vector<VERandVar>& randVars,
					std::map<int, int>& factIdToIdx);

	static int eliminateRandVar(const std::vector<std::shared_ptr<RandVar>>& randVars,
								const std::vector<VEFactor>& factorsVE,
								const std::vector<VERandVar>& randVarsVE,
								const std::map<int, int>& factIdToIdx);
};




class Pgm{
	std::vector<std::shared_ptr<RandVar>> randVarsData;	//id (0, .., n-1) sorted
	std::vector<std::shared_ptr<Cluster>> clustersData;	//id (0, .., m-1) sorted
	std::vector<std::shared_ptr<Feature>> featsData;
	std::vector<double> paramsData;
public:
	Pgm() {}

	Pgm(std::vector<std::shared_ptr<RandVar>> irandVars,
		std::vector<std::shared_ptr<Cluster>> iclusters,
		std::vector<std::shared_ptr<Feature>> ifeats);

	~Pgm();

	static bool incVarValIdxs(std::vector<int>& varValIdxs,
						const std::vector<std::shared_ptr<RandVar>>& vars);

	template<class T>
	static bool incVarValIdxs(std::vector<int>& varValIdxs,
						const std::vector<std::vector<T> >& vals)
	{
		//iterate through all combinations of variables
		int pos = 0;
		bool carry = true;
		while(pos < vals.size() && carry == true){
			carry = false;
			varValIdxs[pos]++;
			if(varValIdxs[pos] >= vals[pos].size()){
				varValIdxs[pos] = 0;
				pos++;
				carry = true;
			}
		}
		if(pos == vals.size()){
			return false;
		}
		else{
			return true;
		}
	}

	static inline int roundToInt(double val){
		return (int)(val + (val >= 0.0 ? 0.5 : -0.5));	//round to nearest int
	}

	static void addEdgeToPgm(std::shared_ptr<Cluster> a,
			std::shared_ptr<Cluster> b,
			std::vector<std::shared_ptr<RandVar>> sepset /* id sorted */);

//	void deleteContents();

	inline const std::vector<std::shared_ptr<RandVar>>& constRandVars() const{
		return randVarsData;
	}

	inline std::vector<std::shared_ptr<RandVar>>& randVars(){
		return randVarsData;
	}

	inline const std::vector<std::shared_ptr<Cluster>>& constClusters() const{
		return clustersData;
	}

	inline std::vector<std::shared_ptr<Cluster>>& clusters(){
		return clustersData;
	}

	inline std::vector<std::shared_ptr<Feature>>& feats(){
		return featsData;
	}

	inline const std::vector<double>& constParams() const{
		return paramsData;
	}

	inline std::vector<double>& params(){
		return paramsData;
	}

    inline const std::vector<double>& params() const {
        return paramsData;
    }
};

inline bool compIdRandVars(const std::shared_ptr<RandVar> lh, const std::shared_ptr<RandVar> rh){
	return lh->id() < rh->id();
}

inline bool compIdFeats(const std::shared_ptr<Feature> lh, const std::shared_ptr<Feature> rh){
	return lh->id() < rh->id();
}

inline bool compIdClust(const std::weak_ptr<Cluster> lh, const std::weak_ptr<Cluster> rh){
	return lh.lock()->id() < rh.lock()->id();
}

inline bool compIdPairClustPInt(const std::pair<std::weak_ptr<Cluster>, int>& lh, const std::pair<std::weak_ptr<Cluster>, int>& rh){
	return lh.first.lock()->id() < rh.first.lock()->id();
}

#endif /* PGM_H_ */
