#include <iostream>
#include <cmath>
#include <fstream>

#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>
#include <Stepometer.hpp>

#include "pgm/Pgm.h"
#include "pgm/Inference.h"
#include "pgm/ParamEst.h"

#include "LocationWiFi.hpp"
#include "WiFiSeqFeatures.hpp"

using namespace std;

static constexpr int ssThreshold = -100;
static constexpr double sharedPercentThreshold = 0.6;

// in meters
static constexpr double mapGrid = 1.0;
static constexpr double mapMinX = 0.0;
static constexpr double mapMaxX = 130.0;
static constexpr double mapMinY = 0.0;
static constexpr double mapMaxY = 40.0;

static constexpr int mapGridSizeX = ceil((mapMaxX - mapMinX) / mapGrid);
static constexpr int mapGridSizeY = ceil((mapMaxY - mapMinY) / mapGrid);

static constexpr double wifiSigma = 1.0/8.0;
static constexpr double errorSigma = 0.8;

static constexpr double probVisScale = 25;

std::vector<LocationWiFi> readMap(boost::filesystem::path dirPath,
                             cv::Mat &mapImage,
                             double &mapImageScale)
{
    cout << "Reading map" << endl;
    
    vector<LocationWiFi> mapLocations;
    
    ifstream wifiFile((dirPath / "wifi.map").c_str());
    if(!wifiFile.is_open()){
        cout << "Error! Could not open " << (dirPath / "wifi.map").c_str() << " file" << endl;
    }
    while(!wifiFile.eof() && !wifiFile.fail()){
        LocationWiFi curLoc;
        
        uint64_t timestamp = 0;
        int locId = 0;
        int nscans = 0;
        wifiFile >> timestamp >> locId >> nscans;
//        cout << "nscans = " << nscans << endl;
        if(!wifiFile.fail()){
            for(int i = 0; i < nscans; ++i){
                string bssid;
                string ssid;
                double level = 0;
                int freq = 0;
                uint64_t localTimestamp = 0;
    
                {
                    string tmp;
                    getline(wifiFile, tmp, '\n');
                }
                getline(wifiFile, bssid, '\t');
                getline(wifiFile, ssid, '\t');
                wifiFile >> level >> freq >> localTimestamp;
//                cout << "localTimestamp = " << localTimestamp << endl;
                if(!wifiFile.fail()){
                    curLoc.wifiScans.emplace_back(bssid, ssid, level, freq, localTimestamp);
                }
            }
            
//            cout << "adding " << mapLocations.size() << endl;
            mapLocations.push_back(curLoc);
        }
    }
    cout << "read " << mapLocations.size() << " locations" << endl;
    
    ifstream positionsFile((dirPath / "positions.map").c_str());
    if(!positionsFile.is_open()){
        cout << "Error! Could not open " << (dirPath / "positions.map").c_str() << " file" << endl;
    }
    while(!positionsFile.eof() && !positionsFile.fail()){
        int id;
        double x, y;
        positionsFile >> id >> x >> y;
//        cout << "id = " << id << endl;
        if(!positionsFile.fail()){
            if(id >= 0 && id < mapLocations.size()) {
                mapLocations[id].locationXY = LocationXY(x, y);
            }
        }
    }
    
    ifstream scaleFile((dirPath / "scale.map").c_str());
    if(!scaleFile.is_open()){
        cout << "Error! Could not open " << (dirPath / "scale.map").c_str() << " file" << endl;
    }
    scaleFile >> mapImageScale;
    
    mapImage = cv::imread((dirPath / "map.png").c_str());
    if(mapImage.empty()){
        cout << "Error! Could not open " << (dirPath / "map.png").c_str() << " file" << endl;
    }
    
    cout << "Map read" << endl;
    
    return mapLocations;
}

void readTrajectory(boost::filesystem::path dirPath,
                    std::vector<LocationWiFi> &wifiLocations,
                    std::vector<double> &stepDists)
{
    cout << "Reading trajectory" << endl;
    
    ifstream wifiFile((dirPath / "wifi.map").c_str());
    if(!wifiFile.is_open()){
        cout << "Error! Could not open " << (dirPath / "wifi.raw").c_str() << " file" << endl;
    }
    while(!wifiFile.eof() && !wifiFile.fail()){
        LocationWiFi curLoc;
        
        uint64_t timestamp = 0;
        int locId = 0;
        int nscans = 0;
        wifiFile >> curLoc.timestamp >> locId >> nscans;
//        cout << "nscans = " << nscans << endl;
        if(!wifiFile.fail()){
            for(int i = 0; i < nscans; ++i){
                string bssid;
                string ssid;
                double level = 0;
                int freq = 0;
                uint64_t localTimestamp = 0;
                
                {
                    string tmp;
                    getline(wifiFile, tmp, '\n');
                }
                getline(wifiFile, bssid, '\t');
                getline(wifiFile, ssid, '\t');
                wifiFile >> level >> freq >> localTimestamp;
//                cout << "localTimestamp = " << localTimestamp << endl;
                if(!wifiFile.fail()){
                    curLoc.wifiScans.emplace_back(bssid, ssid, level, freq, localTimestamp);
                }
            }

//            cout << "adding " << wifiLocations.size() << endl;
            wifiLocations.push_back(curLoc);
        }
    }
    
    ifstream positionsFile((dirPath / "positions.map").c_str());
    if(!positionsFile.is_open()){
        cout << "Error! Could not open " << (dirPath / "positions.map").c_str() << " file" << endl;
    }
    while(!positionsFile.eof() && !positionsFile.fail()){
        int id;
        double x, y;
        positionsFile >> id >> x >> y;
//        cout << "id = " << id << endl;
        if(!positionsFile.fail()){
            if(id >= 0 && id < wifiLocations.size()) {
                wifiLocations[id].locationXY = LocationXY(x, y);
            }
        }
    }
    
    // distances from stepometer
    stepDists.clear();
    
    ifstream accFile((dirPath / "acc.map").c_str());
    if(!accFile.is_open()){
        cout << "Error! Could not open " << (dirPath / "acc.raw").c_str() << " file" << endl;
    }
    vector<double> accSamp;
    vector<uint64_t> accSampTs;
    while(!accFile.eof() && !accFile.fail()){
        uint64_t timestamp;
        double accX, accY, accZ;
        accFile >> timestamp >> accX >> accY >> accZ;
        if(!accFile.fail()){
            accSamp.push_back(sqrt(accX*accX + accY*accY + accZ*accZ));
            accSampTs.push_back(timestamp);
        }
    }
    
    {
        static constexpr int winLen = 32;
        static constexpr double stepLen = 0.7;
        static constexpr double accSampFreq = 25;
        static constexpr double freqMin = 1.1;
        static constexpr double freqMax = 2.6;
        static constexpr double fftMagThresh = 0.2;
        
        int curAccSampIdx = 0;
        for(int s = 0; s < wifiLocations.size(); ++s){
            uint64_t curScanTs = wifiLocations[s].timestamp;
            double curDist = 0;
         
            while(curAccSampIdx < accSamp.size() && accSampTs[curAccSampIdx] < curScanTs){
                if(curAccSampIdx >= winLen - 1){
                    vector<double> curAccSamp(accSamp.begin() + curAccSampIdx - winLen + 1,
                                              accSamp.begin() + curAccSampIdx + 1);
                    
//                    cout << "curAccSamp = " << curAccSamp << endl;
                    curDist += Stepometer::computeDist(curAccSamp,
                                                       (accSampTs[curAccSampIdx] - accSampTs[curAccSampIdx - 1]) * 1.0e-9,
                                                       accSampFreq,
                                                       stepLen,
                                                       freqMin,
                                                       freqMax,
                                                       fftMagThresh);
                }
                
                ++curAccSampIdx;
            }
            
//            cout << "curDist = " << curDist << endl;
            stepDists.push_back(curDist);
        }
    }
    
    cout << "Trajectory read" << endl;
}

pair<double, int> errorL2(const LocationWiFi &lhs, const LocationWiFi &rhs){
    int cnt = 0;
    double diff = 0.0;
    
    for(const ScanResult &a : lhs.wifiScans){
        for(const ScanResult &b : rhs.wifiScans){
            if(a.level > ssThreshold && b.level > ssThreshold){
                if(a.bssid == b.bssid){
                    diff += (a.level - b.level) * (a.level - b.level);
                    ++cnt;
                }
            }
        }
    }
    
    if(cnt < 1){
        diff = 1e9;
    }
    else{
        diff = sqrt(diff/cnt);
    };
    return make_pair(diff, cnt);
}

pair<int, int> mapCoordToGrid(double x, double y){
    return make_pair((x - mapMinX)/mapGrid, (y - mapMinY)/mapGrid);
};

LocationXY mapGridToCoord(int x, int y){
    return LocationXY(mapMinX + x * mapGrid, mapMinY + y * mapGrid);
};

std::vector<std::vector<double>> locationProb(const LocationWiFi &loc,
                            const std::vector<LocationWiFi> &database)
{

//    int mapGridSizeX = ceil((mapMaxX - mapMinX) / mapGrid);
//    int mapGridSizeY = ceil((mapMaxY - mapMinY) / mapGrid);
    vector<vector<double>> prob(mapGridSizeY, vector<double>(mapGridSizeX, 0));
    
    int nMatchedScans = 0;
    for (const LocationWiFi &databaseLoc : database) {
        pair<double, int> error = errorL2(loc, databaseLoc);
        double sharedPercentA = (double) error.second / loc.wifiScans.size();
        double sharedPercentB = (double) error.second / databaseLoc.wifiScans.size();
        
        if (sharedPercentA > sharedPercentThreshold &&
            sharedPercentB > sharedPercentThreshold)
        {
//            cout << "matched scan" << endl;
            // adding Gaussian kernel placed at databaseLoc and weighted with error
            for (int mapYIdx = 0; mapYIdx < prob.size(); ++mapYIdx) {
                for (int mapXIdx = 0; mapXIdx < prob[mapYIdx].size(); ++mapXIdx) {
                    LocationXY mapCoord = mapGridToCoord(mapXIdx, mapYIdx);
                    
//                    cout << "mapCoord = (" << mapCoord.x << ", " << mapCoord.y << ")" << endl;
//                    cout << "databaseLoc.locationXY = (" << databaseLoc.locationXY.x << ", " << databaseLoc.locationXY.y << ")" << endl;
                    double dx = mapCoord.x - databaseLoc.locationXY.x;
                    double dy = mapCoord.y - databaseLoc.locationXY.y;
                    double expVal = -(dx * dx * wifiSigma * wifiSigma +
                                      dy * dy * wifiSigma * wifiSigma);
                    expVal += -errorSigma * errorSigma * error.first;
                    
//                    cout << "expVal = " << expVal << endl;
                    prob[mapYIdx][mapXIdx] += exp(expVal);
                }
            }
            ++nMatchedScans;
        }
    }
    cout << "nMatchedScans = " << nMatchedScans << endl;
    return prob;
}

void visualizeMapProb(const std::vector<LocationWiFi> &database,
                      const std::vector<std::vector<double>> &prob,
                      const cv::Mat &mapImage,
                      const double &mapScale)
{
    cv::Mat mapVis = mapImage.clone();
    for(const LocationWiFi &curLoc : database){
        cv::Point2d pt(curLoc.locationXY.x, curLoc.locationXY.y);
        cv::circle(mapVis, pt * mapScale, 5, cv::Scalar(0, 0, 255), CV_FILLED);
    }
    cv::Mat probVal(mapImage.rows, mapImage.cols, CV_32FC1, cv::Scalar(0));
    cv::Mat probVis(mapImage.rows, mapImage.cols, CV_8UC3, cv::Scalar(0));
    double maxVal = 0.0;
    for (int mapYIdx = 0; mapYIdx < prob.size(); ++mapYIdx) {
        for (int mapXIdx = 0; mapXIdx < prob[mapYIdx].size(); ++mapXIdx) {
            double val = prob[mapYIdx][mapXIdx];
//            cout << "val = " << val << endl;
            maxVal = max(val, maxVal);
            
//            LocationXY mapCoord = mapGridToCoord(mapXIdx, mapYIdx);
//
//            cv::Point2d pt1(mapCoord.x - mapGrid/2, mapCoord.y - mapGrid/2);
//            cv::Point2d pt2(mapCoord.x + mapGrid/2, mapCoord.y + mapGrid/2);
//
//            cv::rectangle(probVal, pt1 * mapScale, pt2 * mapScale, cv::Scalar(val * probVisScale), CV_FILLED);
        }
    }
    double valThresh = min(0.1, maxVal/2.0);
    for (int mapYIdx = 0; mapYIdx < prob.size(); ++mapYIdx) {
        for (int mapXIdx = 0; mapXIdx < prob[mapYIdx].size(); ++mapXIdx) {
            double val = prob[mapYIdx][mapXIdx];
//            cout << "val = " << val << endl;
            if(val > valThresh) {
                LocationXY mapCoord = mapGridToCoord(mapXIdx, mapYIdx);
    
                cv::Point2d pt1(mapCoord.x - mapGrid / 2, mapCoord.y - mapGrid / 2);
                cv::Point2d pt2(mapCoord.x + mapGrid / 2, mapCoord.y + mapGrid / 2);
    
                cv::rectangle(probVal, pt1 * mapScale, pt2 * mapScale, cv::Scalar(0.5), CV_FILLED);
            }
        }
    }
    
    cout << "maxVal = " << maxVal << endl;
    
    probVal.convertTo(probVal, CV_8U, 256);
    cv::applyColorMap(probVal, probVis, cv::COLORMAP_JET);
    
    cv::Mat vis = mapVis * 0.75 + probVis * 0.25;
    cv::resize(vis, vis, cv::Size(0, 0), 0.5, 0.5);
    
    cv::imshow("map", vis);
    
    cv::waitKey(10);
}

void visualizeMapInfer(const std::vector<LocationWiFi> &database,
                       const std::vector<LocationWiFi> &trajLocations,
                       const cv::Mat &mapImage,
                       const double &mapScale)
{
    cv::Mat mapVis = mapImage.clone();
    for(const LocationWiFi &curLoc : database){
        cv::Point2d pt(curLoc.locationXY.x, curLoc.locationXY.y);
        cv::circle(mapVis, pt * mapScale, 5, cv::Scalar(0, 0, 255), CV_FILLED);
    }
//    cv::Mat probVal(mapImage.rows, mapImage.cols, CV_32FC1, cv::Scalar(0));
//    cv::Mat probVis(mapImage.rows, mapImage.cols, CV_8UC3, cv::Scalar(0));
//    double maxVal = 0.0;
//    for (int mapYIdx = 0; mapYIdx < prob.size(); ++mapYIdx) {
//        for (int mapXIdx = 0; mapXIdx < prob[mapYIdx].size(); ++mapXIdx) {
//            double val = prob[mapYIdx][mapXIdx] * probVisScale;
////            cout << "val = " << val << endl;
//            maxVal = max(val, maxVal);
//
//            LocationXY mapCoord = mapGridToCoord(mapXIdx, mapYIdx);
//
//            cv::Point2d pt1(mapCoord.x - mapGrid/2, mapCoord.y - mapGrid/2);
//            cv::Point2d pt2(mapCoord.x + mapGrid/2, mapCoord.y + mapGrid/2);
//
//            cv::rectangle(probVal, pt1 * mapScale, pt2 * mapScale, cv::Scalar(val), CV_FILLED);
//        }
//    }
//    cout << "maxVal = " << maxVal << endl;
//
//    probVal.convertTo(probVal, CV_8U, 256);
//    cv::applyColorMap(probVal, probVis, cv::COLORMAP_JET);
//
//    cv::Mat vis = mapVis * 0.75 + probVis * 0.25;
//    cv::resize(vis, vis, cv::Size(0, 0), 0.5, 0.5);
    
    
    for(int i = 0; i < trajLocations.size(); ++i){
        const LocationWiFi &curLoc = trajLocations[i];
        cv::Point2d pt(curLoc.locationXY.x, curLoc.locationXY.y);
        cv::circle(mapVis, pt * mapScale, 5, cv::Scalar(255, 0, 0), CV_FILLED);
        cv::putText(mapVis, to_string(i), pt * mapScale, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0));
    }
    
    cv::resize(mapVis, mapVis, cv::Size(0, 0), 0.5, 0.5);
    
    cv::imshow("map", mapVis);
    
    cv::waitKey();
}

Pgm buildPgm(const std::vector<LocationWiFi> &wifiLocations,
             const std::vector<std::vector<std::vector<double>>> &probs,
             const std::vector<double> &stepDists,
             std::vector<double> &obsVec,
             std::map<int, int> &locIdxToVarClusterId,
             std::vector<double> &varVals)
{
    std::vector<std::shared_ptr<RandVar>> randVars;
    std::vector<std::shared_ptr<Feature>> feats;
    std::vector<std::shared_ptr<Cluster>> clusters;
//    vector<double> obsVec;
    obsVec.clear();
    
    // observation vector
    // location coordinates
    vector<double> xCoords, yCoords;
//    vector<double> rvVals;
    int mapSize = 0;
    for (int mapYIdx = 0; mapYIdx < probs[0].size(); ++mapYIdx) {
        for (int mapXIdx = 0; mapXIdx < probs[0][mapYIdx].size(); ++mapXIdx) {
            LocationXY mapCoord = mapGridToCoord(mapXIdx, mapYIdx);
            xCoords.push_back(mapCoord.x);
            yCoords.push_back(mapCoord.y);
//            rvVals.push_back(mapSize++);
            ++mapSize;
        }
    }
    int obsVecStartLoc = 0;
    obsVec.insert(obsVec.end(), xCoords.begin(), xCoords.end());
    obsVec.insert(obsVec.end(), yCoords.begin(), yCoords.end());
    
    vector<int> obsVecStartProb;
    for(int i = 0; i < probs.size(); ++i){
        obsVecStartProb.push_back(obsVec.size());
        for (int mapYIdx = 0; mapYIdx < probs[i].size(); ++mapYIdx) {
            for (int mapXIdx = 0; mapXIdx < probs[i][mapYIdx].size(); ++mapXIdx) {
                obsVec.push_back(probs[i][mapYIdx][mapXIdx]);
            }
        }
    }
    int obsVecStartDist = obsVec.size();
    for(int i = 0; i < probs.size(); ++i){
        obsVec.push_back(stepDists[i]);
    }
    
    // random variables
    int nextRandVarId = 0;
    for(int i = 0; i < probs.size(); ++i){
        locIdxToVarClusterId[i] = nextRandVarId;
    
        double maxProb = 0;
        for (int mapYIdx = 0; mapYIdx < probs[i].size(); ++mapYIdx) {
            for (int mapXIdx = 0; mapXIdx < probs[i][mapYIdx].size(); ++mapXIdx) {
                maxProb = max(maxProb, probs[i][mapYIdx][mapXIdx]);
            }
        }
    
        int locX = std::round(wifiLocations[i].locationXY.x);
        int locY = std::round(wifiLocations[i].locationXY.y);
        double varVal = locX + mapGridSizeX * locY;
        double closestDist = std::numeric_limits<double>::max();
        double closestVarVal = varVal;
        
        double probThresh = min(0.1, maxProb/2.0);
        vector<double> rvVals;
        int mapIdx = 0;
        for (int mapYIdx = 0; mapYIdx < probs[i].size(); ++mapYIdx) {
            for (int mapXIdx = 0; mapXIdx < probs[i][mapYIdx].size(); ++mapXIdx) {
                if(probs[i][mapYIdx][mapXIdx] > probThresh){
                    rvVals.push_back(mapIdx);
                    
                    LocationXY loc = mapGridToCoord(mapXIdx, mapYIdx);
                    double dist = sqrt((loc.x - locX)*(loc.x - locX) + (loc.y - locY)*(loc.y - locY));
                    if(dist < closestDist){
                        closestDist = dist;
                        closestVarVal = mapIdx;
                    }
                }
                ++mapIdx;
            }
        }
        cout << "rvVals.size() = " << rvVals.size() << endl;
        
        
        varVals.push_back(varVal);
        if(find(rvVals.begin(), rvVals.end(), varVal) == rvVals.end()){
            cout << "Warning - varVal not in rvVals, substituting with the closest" << endl;
            cout << "Closest distance = " << closestDist << endl;
            
            varVal = closestVarVal;
        }
        
        
        shared_ptr<RandVar> curRandVar(new RandVar(nextRandVarId++, rvVals));
        
        randVars.push_back(curRandVar);
    }
    
    // features
    int nextFeatId = 0;
    int nextParamId = 0;
    // location features
    vector<shared_ptr<Feature>> locFeats;
    for(int i = 0; i < probs.size(); ++i){
        vector<int> curObsVecIdxs(mapSize);
        iota(curObsVecIdxs.begin(), curObsVecIdxs.end(), obsVecStartProb[i]);
        
        shared_ptr<Feature> curFeat(new LocFeature(nextFeatId++,
                                                   nextParamId,
                                                   vector<shared_ptr<RandVar>>{randVars[i]},
                                                   curObsVecIdxs));
        
        locFeats.push_back(curFeat);
        feats.push_back(curFeat);
    }
    ++nextParamId;
    
    // move features
    vector<shared_ptr<Feature>> moveFeats;
    // skip first one, as there is no distance from previous location
    for(int i = 1; i < probs.size(); ++i){
        vector<int> curObsVecIdxs(1 + 2*mapSize);
        curObsVecIdxs[0] = obsVecStartDist + i;
        // x coordinates
        iota(curObsVecIdxs.begin() + 1, curObsVecIdxs.begin() + 1 + mapSize, obsVecStartLoc);
        // y coordinates
        iota(curObsVecIdxs.begin() + 1 + mapSize, curObsVecIdxs.begin() + 1 + 2*mapSize, obsVecStartLoc + mapSize);
        
        shared_ptr<Feature> curFeat(new MoveFeature(nextFeatId++,
                                                    nextParamId,
                                                    vector<shared_ptr<RandVar>>{randVars[i-1], randVars[i]},
                                                    curObsVecIdxs,
                                                    mapSize,
                                                    1.0/2.0));
        
        moveFeats.push_back(curFeat);
        feats.push_back(curFeat);
    }
    ++nextParamId;
    
    //clusters
    int nextClusterId = 0;
    vector<shared_ptr<Cluster>> rvClusters;
    for(int i = 0; i < probs.size(); ++i){
        shared_ptr<Cluster> curCluster(new Cluster(nextClusterId++,
                                                   vector<shared_ptr<Feature>>{},
                                                   vector<shared_ptr<RandVar>>{randVars[i]}));
        
        rvClusters.push_back(curCluster);
        clusters.push_back(curCluster);
    }
    
    vector<shared_ptr<Cluster>> locFeatClusters;
    for(int i = 0; i < probs.size(); ++i){
        shared_ptr<Cluster> curCluster(new Cluster(nextClusterId++,
                                                   vector<shared_ptr<Feature>>{locFeats[i]},
                                                   vector<shared_ptr<RandVar>>{randVars[i]}));
    
        locFeatClusters.push_back(curCluster);
        clusters.push_back(curCluster);
    }
    
    vector<shared_ptr<Cluster>> moveFeatClusters;
    for(int i = 1; i < probs.size(); ++i){
        shared_ptr<Cluster> curCluster(new Cluster(nextClusterId++,
                                                   vector<shared_ptr<Feature>>{moveFeats[i - 1]},
                                                   vector<shared_ptr<RandVar>>{randVars[i - 1], randVars[i]}));
    
        moveFeatClusters.push_back(curCluster);
        clusters.push_back(curCluster);
    }
    
    // create pgm
    Pgm pgm(randVars, clusters, feats);
    
    // edges from random variable clusters to location feature clusters
    for(int i = 0; i < probs.size(); ++i){
        pgm.addEdgeToPgm(rvClusters[i], locFeatClusters[i], vector<shared_ptr<RandVar>>{randVars[i]});
    }
    
    for(int i = 1; i < probs.size(); ++i){
        pgm.addEdgeToPgm(rvClusters[i - 1], moveFeatClusters[i - 1], vector<shared_ptr<RandVar>>{randVars[i - 1]});
        pgm.addEdgeToPgm(rvClusters[i], moveFeatClusters[i - 1], vector<shared_ptr<RandVar>>{randVars[i]});
    }
    
    pgm.params() = vector<double>(nextParamId, 1.0);
    
    return pgm;
}

vector<LocationXY> inferLocations(int nloc,
                                  const Pgm &pgm,
                                  const vector<double> &obsVec,
                                  const std::map<int, int> &locIdxToVarClusterId)
{
    vector<LocationXY> retLoc;
    
    vector<vector<double>> marg;
    vector<vector<vector<double>>> msgs;
    vector<double> params{1.0, 1.0};
    
    bool calibrated = Inference::compMAPParam(pgm,
                                            marg,
                                            msgs,
                                            params,
                                            obsVec);
    
    cout << "calibrated = " << calibrated << endl;
    
    vector<vector<double>> retVals = Inference::decodeMAP(pgm,
                                                         marg,
                                                         msgs,
                                                         params,
                                                         obsVec);
    
    for(int i = 0; i < nloc; ++i){
        int varClusterId = locIdxToVarClusterId.at(i);
        const vector<double> &curVals = retVals[varClusterId];
        
        int curLoc = curVals.front();
        
        int mapXIdx = curLoc % mapGridSizeX;
        int mapYIdx = curLoc / mapGridSizeX;
        
        LocationXY curLocXY = mapGridToCoord(mapXIdx, mapYIdx);
        
        retLoc.push_back(curLocXY);
    }
    
    return retLoc;
}

void removeNotMatchedLocations(std::vector<LocationWiFi> &wifiLocations,
                               std::vector<double> &stepDists,
                               std::vector<std::vector<std::vector<double>>> &probs)
{
    for(int i = 0; i < probs.size(); ++i){
        double maxVal = 0.0;
        for (int mapYIdx = 0; mapYIdx < probs[i].size(); ++mapYIdx) {
            for (int mapXIdx = 0; mapXIdx < probs[i][mapYIdx].size(); ++mapXIdx) {
                double val = probs[i][mapYIdx][mapXIdx];

                maxVal = max(val, maxVal);
                
            }
        }
        
        if(maxVal < 1e-5){
            if(i < probs.size() - 1){
                // adding distance to next location
                stepDists[i + 1] += stepDists[i];
            }
            
            wifiLocations.erase(wifiLocations.begin() + i);
            stepDists.erase(stepDists.begin() + i);
            probs.erase(probs.begin() + i);
            
            --i;
        }
    }
}

int main() {
    static constexpr bool estimateParams = true;
    static constexpr bool infer = false;
    
    boost::filesystem::path mapDirPath("../res/Maps/PUTMC_Lenovo_18_05_21");
    
    cv::Mat mapImage;
    double mapScale;
    vector<LocationWiFi> mapLocations = readMap(mapDirPath, mapImage, mapScale);
    
    vector<boost::filesystem::path> trajDirPaths{"../res/Trajectories/traj1",
                                                 "../res/Trajectories/traj2",
                                                 "../res/Trajectories/traj3"};
    
    vector<Pgm> pgms;
    vector<vector<double>> obsVecs;
    vector<vector<double>> varVals;
    
    for(int t = 0; t < trajDirPaths.size(); ++t) {
    
        vector<LocationWiFi> curTrajLocations;
        vector<double> curStepDists;
        readTrajectory(trajDirPaths[t], curTrajLocations, curStepDists);
    
        vector<vector<vector<double>>> curProbs;
        for (int i = 0; i < curTrajLocations.size(); ++i) {
            vector<vector<double>> curProb = locationProb(curTrajLocations[i], mapLocations);
            visualizeMapProb(mapLocations, curProb, mapImage, mapScale);
        
            curProbs.push_back(curProb);
        }
    
        removeNotMatchedLocations(curTrajLocations,
                                  curStepDists,
                                  curProbs);
    
        vector<double> curObsVec;
        map<int, int> curLocIdxToRandVarClusterId;
        vector<double> curVarVals;
        Pgm curPgm = buildPgm(curTrajLocations,
                           curProbs,
                           curStepDists,
                           curObsVec,
                           curLocIdxToRandVarClusterId,
                           curVarVals);
    
        if(infer) {
            vector<LocationXY> infLoc = inferLocations(curProbs.size(),
                                                       curPgm,
                                                       curObsVec,
                                                       curLocIdxToRandVarClusterId);
    
            for (int i = 0; i < infLoc.size(); ++i) {
                curTrajLocations[i].locationXY = infLoc[i];
            }
    
            visualizeMapInfer(mapLocations,
                              curTrajLocations,
                              mapImage,
                              mapScale);
        }
        
        pgms.push_back(curPgm);
        varVals.push_back(curVarVals);
        obsVecs.push_back(curObsVec);
    }
    if(estimateParams){
        cout << "estimating parameters" << endl;
        ParamEst paramEst;
        paramEst.estimateParams(pgms,
                                 varVals,
                                 obsVecs);
    }
}