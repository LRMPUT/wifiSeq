#include <iostream>
#include <cmath>
#include <fstream>

#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>

#include <tinyxml.h>

#include "pgm/Pgm.h"
#include "pgm/Inference.h"
#include "pgm/ParamEst.h"

#include "LocationWiFi.hpp"
#include "WiFiSeqFeatures.hpp"
#include "Stepometer.hpp"

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

static constexpr double wifiSigma = 8.0;
static constexpr double errorSigma = 2;
static constexpr double distSigma = 8.0;

// parameter estimation
static constexpr double probThresh = 0.03;
// inference
//static constexpr double probThresh = 0.1;

static constexpr double probRatio = 2.0;

static constexpr double probScale = 1.0/5.0;

static constexpr double probVisScale = 0.5;


void selectPolygonPixels(const std::vector<cv::Point2i> &polygon,
                         int regionId,
                         cv::Mat& regionsOnImage)
{
    int polyCnt[] = {(int)polygon.size()};
    const cv::Point2i* points[] = {polygon.data()};
    //Point2i array
    fillPoly(regionsOnImage, points, polyCnt, 1, cv::Scalar(regionId));
//    int count = regionsOnImage.rows * regionsOnImage.cols - cv::countNonZero(regionsOnImage == cv::Scalar(regionId));
//
//    return count;
}

cv::Mat readAnnotation(const boost::filesystem::path &annotationFile,
                       const cv::Size &imageSize,
                       const map<string, int> &labelNameToId)
{
    TiXmlDocument data(annotationFile.string());
    if(!data.LoadFile()){
        throw "Bad data file";
    }
    TiXmlElement* pAnnotation = data.FirstChildElement("annotation");
    if(!pAnnotation){
        throw "Bad data file - no annotation entry";
    }
    TiXmlElement* pFile = pAnnotation->FirstChildElement("filename");
    if(!pFile){
        throw "Bad data file - no filename entry";
    }
    
    cv::Mat annotations(imageSize, CV_32SC1, cv::Scalar(0));
    
    TiXmlElement* pObject = pAnnotation->FirstChildElement("object");
    while(pObject){
        
        TiXmlElement* pPolygon = pObject->FirstChildElement("polygon");
        if(!pPolygon){
            throw "Bad data file - no polygon inside object";
        }
        vector<cv::Point2i> poly;
        
        TiXmlElement* pPt = pPolygon->FirstChildElement("pt");
        while(pPt){
            int x = stoi(pPt->FirstChildElement("x")->GetText());
            int y = stoi(pPt->FirstChildElement("y")->GetText());
            poly.push_back(cv::Point2i(x, y));
            pPt = pPt->NextSiblingElement("pt");
        }
        
        TiXmlElement* pAttributes = pObject->FirstChildElement("attributes");
        if(!pAttributes){
            throw "Bad data file - no object attributes";
        }
        string labelText = pAttributes->GetText();
        int label = -1;
        if(labelNameToId.count(labelText) > 0){
            label = labelNameToId.at(labelText);
        }
        else{
            throw "No such label found";
        }
        
        //cout << "Selecting polygon pixels for label " << labels[label] <<  endl;
        selectPolygonPixels(poly, label, annotations);
        //cout << "End selecting" << endl;
        
        pObject = pObject->NextSiblingElement("object");
    }
    
    return annotations;
}



std::vector<LocationWiFi> readMap(const boost::filesystem::path &dirPath,
                                  cv::Mat &mapImage,
                                  cv::Mat &obstacles,
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
    
    
    obstacles = readAnnotation(dirPath / "map.xml",
                                mapImage.size(),
                                map<string, int>{make_pair("obstacle", 255)});
    
    
    cout << "Map read" << endl;
    
    return mapLocations;
}

void readTrajectory(const boost::filesystem::path &dirPath,
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
    vector<vector<double>> prob(mapGridSizeY, vector<double>(mapGridSizeX, 0.01));
    
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
                    double expVal = -((dx * dx + dy * dy) / (wifiSigma * wifiSigma));
                    expVal += -error.first / (errorSigma * errorSigma);
                    
//                    cout << "expVal = " << expVal << endl;
                    prob[mapYIdx][mapXIdx] += exp(expVal) * probScale;
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
                      const double &varVal,
                      const cv::Mat &mapImage,
                      const cv::Mat &mapObstacle,
                      const double &mapScale)
{
    cv::Mat mapVis = mapImage.clone();
    for(const LocationWiFi &curLoc : database){
        cv::Point2d pt(curLoc.locationXY.x, curLoc.locationXY.y);
        cv::circle(mapVis, pt * mapScale, 5, cv::Scalar(0, 0, 255), CV_FILLED);
    }
    cv::Mat probVal(mapImage.rows, mapImage.cols, CV_32FC1, cv::Scalar(0));
    cv::Mat probVis(mapImage.rows, mapImage.cols, CV_8UC3, cv::Scalar(0));

    double maxProb = 0.0;
    for (int mapYIdx = 0; mapYIdx < prob.size(); ++mapYIdx) {
        for (int mapXIdx = 0; mapXIdx < prob[mapYIdx].size(); ++mapXIdx) {
            double val = prob[mapYIdx][mapXIdx];
//            cout << "val = " << val << endl;
            maxProb = max(val, maxProb);
            
            LocationXY mapCoord = mapGridToCoord(mapXIdx, mapYIdx);

            cv::Point2d pt1(mapCoord.x - mapGrid/2, mapCoord.y - mapGrid/2);
            cv::Point2d pt2(mapCoord.x + mapGrid/2, mapCoord.y + mapGrid/2);

            cv::rectangle(probVal, pt1 * mapScale, pt2 * mapScale, cv::Scalar(val * probVisScale), CV_FILLED);
        }
    }
//    double curProbThresh = min(probThresh, maxProb / probRatio);
//    for (int mapYIdx = 0; mapYIdx < prob.size(); ++mapYIdx) {
//        for (int mapXIdx = 0; mapXIdx < prob[mapYIdx].size(); ++mapXIdx) {
//            double val = prob[mapYIdx][mapXIdx];
////            cout << "val = " << val << endl;
//            if(val >= curProbThresh) {
//                LocationXY mapCoord = mapGridToCoord(mapXIdx, mapYIdx);
//
//                cv::Point2d pt1(mapCoord.x - mapGrid / 2, mapCoord.y - mapGrid / 2);
//                cv::Point2d pt2(mapCoord.x + mapGrid / 2, mapCoord.y + mapGrid / 2);
//
//                cv::rectangle(probVal, pt1 * mapScale, pt2 * mapScale, cv::Scalar(0.5), CV_FILLED);
//            }
//        }
//    }
    
    cout << "maxProb = " << maxProb << endl;
    
    probVal.convertTo(probVal, CV_8U, 256);
    cv::applyColorMap(probVal, probVis, cv::COLORMAP_JET);
    
    {
        int mapXIdx = ((int)varVal) % mapGridSizeX;
        int mapYIdx = ((int)varVal) / mapGridSizeX;
        LocationXY loc = mapGridToCoord(mapXIdx, mapYIdx);
        cv::Point2d pt(loc.x, loc.y);
        cv::circle(mapVis, pt * mapScale, 5, cv::Scalar(0, 255, 0), CV_FILLED);
    }
    
    {
        probVis.setTo(cv::Scalar(255, 255, 0), mapObstacle == 255);
    }
    
    cv::Mat vis = mapVis * 0.75 + probVis * 0.25;
    cv::resize(vis, vis, cv::Size(0, 0), 0.5, 0.5);
    
    cv::imshow("map", vis);
    
    cv::waitKey(10);
}

void visualizeMapInfer(const std::vector<LocationWiFi> &database,
                       const std::vector<LocationWiFi> &trajLocations,
                       const std::vector<LocationXY> &inferLocations,
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
        const LocationXY &infLoc = inferLocations[i];
        cv::Point2d pt(infLoc.x, infLoc.y);
        cv::circle(mapVis, pt * mapScale, 5, cv::Scalar(255, 0, 0), CV_FILLED);
        
        cv::putText(mapVis, to_string(i), pt * mapScale, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0));
        if(i > 0){
            const LocationXY &prevInfLoc = inferLocations[i-1];
            cv::Point2d prevPt(prevInfLoc.x, prevInfLoc.y);
            
            cv::line(mapVis, prevPt * mapScale, pt * mapScale, cv::Scalar(255, 0, 0), 2);
        }
        
        {
            const LocationXY &gtLoc = trajLocations[i].locationXY;
            cv::Point2d gtPt(gtLoc.x, gtLoc.y);
            cv::circle(mapVis, gtPt * mapScale, 5, cv::Scalar(0, 255, 0), CV_FILLED);
        }
    }
    
    cv::resize(mapVis, mapVis, cv::Size(0, 0), 0.5, 0.5);
    
    cv::imshow("map", mapVis);
    
    cv::waitKey();
}

Pgm buildPgm(const std::vector<LocationWiFi> &wifiLocations,
             const cv::Mat &obstacles,
             const double &mapScale,
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
    varVals.clear();
    
    set<int> forbiddenVals;
    {
//        cout << "obstacles.size() = " << obstacles.size() << endl;
        int val = 0;
        for (int mapYIdx = 0; mapYIdx < probs[0].size(); ++mapYIdx) {
            for (int mapXIdx = 0; mapXIdx < probs[0][mapYIdx].size(); ++mapXIdx) {
                LocationXY mapCoord = mapGridToCoord(mapXIdx, mapYIdx);
                
                cv::Point2d pt1(mapCoord.x - mapGrid / 2, mapCoord.y - mapGrid / 2);
                cv::Point2d pt2(mapCoord.x + mapGrid / 2, mapCoord.y + mapGrid / 2);
                cv::Point2i pt1i = pt1 * mapScale;
                cv::Point2i pt2i = pt2 * mapScale;
                pt1i.x = max(pt1i.x, 0);
                pt1i.y = max(pt1i.y, 0);
                pt2i.x = max(pt2i.x, 0);
                pt2i.y = max(pt2i.y, 0);
                pt1i.x = min(pt1i.x, obstacles.cols);
                pt1i.y = min(pt1i.y, obstacles.rows);
                pt2i.x = min(pt2i.x, obstacles.cols);
                pt2i.y = min(pt2i.y, obstacles.rows);
                
                cv::Rect roi = cv::Rect(pt1i, pt2i);
//                cout << "roi = " << roi << endl;
                cv::Mat curObstacle = obstacles(roi);
//                cout << "curObstacle = " << curObstacle << endl;
                int obstacleCnt = cv::countNonZero(curObstacle);
                
//                cout << "obstacleCnt = " << obstacleCnt << endl;
//                cout << "curObstacle.rows * curObstacle.cols / 2 = " << curObstacle.rows * curObstacle.cols / 2 << endl;
                if(obstacleCnt > curObstacle.rows * curObstacle.cols / 2){
                    forbiddenVals.insert(val);
                }
                
                ++val;
            }
        }
//        cout << "forbiddenVals.size() = " << forbiddenVals.size() << endl;
    }
    
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
        
        double curProbThresh = min(probThresh, maxProb / probRatio);
        vector<double> rvVals;
        int mapIdx = 0;
        for (int mapYIdx = 0; mapYIdx < probs[i].size(); ++mapYIdx) {
            for (int mapXIdx = 0; mapXIdx < probs[i][mapYIdx].size(); ++mapXIdx) {
                if(probs[i][mapYIdx][mapXIdx] >= curProbThresh){
                    
                    if(forbiddenVals.count(mapIdx) == 0) {
                        rvVals.push_back(mapIdx);
    
                        LocationXY loc = mapGridToCoord(mapXIdx, mapYIdx);
                        double dist = sqrt((loc.x - locX)*(loc.x - locX) + (loc.y - locY)*(loc.y - locY));
                        if(dist < closestDist){
                            closestDist = dist;
                            closestVarVal = mapIdx;
                        }
                    }
                }
                ++mapIdx;
            }
        }
        cout << "rvVals.size() = " << rvVals.size() << endl;
        
        if(find(rvVals.begin(), rvVals.end(), varVal) == rvVals.end()){
            cout << "Warning - varVal not in rvVals, substituting with the closest" << endl;
            cout << "Closest distance = " << closestDist << endl;
            
            varVal = closestVarVal;
        }
        varVals.push_back(varVal);
        
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
                                                    distSigma));
        
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
    
    pgm.params() = vector<double>{1.80407, 21.7148};
//    pgm.params() = vector<double>{1.0, 1.0};
    
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
    try{
//        static constexpr bool estimateParams = true;
//        static constexpr bool infer = false;
        static constexpr bool estimateParams = false;
        static constexpr bool infer = true;
        
        static constexpr int seqLen = 10;
        
        boost::filesystem::path mapDirPath("../res/Maps/PUTMC_Lenovo_18_05_21_full");
        
        cv::Mat mapImage;
        cv::Mat mapObstacles;
        double mapScale;
        vector<LocationWiFi> mapLocations = readMap(mapDirPath, mapImage, mapObstacles, mapScale);
        
//        vector<boost::filesystem::path> trajDirPaths{"../res/Trajectories/traj1",
//                                                     "../res/Trajectories/traj2",
//                                                     "../res/Trajectories/traj3"};
        vector<boost::filesystem::path> trajDirPaths{"../res/Trajectories/traj4",
                                                     "../res/Trajectories/traj5",
                                                     "../res/Trajectories/traj6"};
        
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
            
                curProbs.push_back(curProb);
            }
        
            removeNotMatchedLocations(curTrajLocations,
                                      curStepDists,
                                      curProbs);
        
            vector<double> curObsVec;
            map<int, int> curLocIdxToRandVarClusterId;
            vector<double> curVarVals;
            Pgm curPgm = buildPgm(curTrajLocations,
                               mapObstacles,
                               mapScale,
                               curProbs,
                               curStepDists,
                               curObsVec,
                               curLocIdxToRandVarClusterId,
                               curVarVals);
        
            for (int i = 0; i < curTrajLocations.size(); ++i) {
                visualizeMapProb(mapLocations,
                                 curProbs[i],
                                 curVarVals[i],
                                 mapImage,
                                 mapObstacles,
                                 mapScale);
            }
        
            if(infer) {
                vector<double> errors;
                double errorSum = 0;
                int errorCnt = 0;
                
                for (int i = seqLen; i < curTrajLocations.size(); ++i) {
                    vector<double> iObsVec;
                    map<int, int> iLocIdxToRandVarClusterId;
                    vector<double> iVarVals;
                    Pgm iPgm = buildPgm(vector<LocationWiFi>(curTrajLocations.begin() + i - seqLen, curTrajLocations.begin() + i),
                                          mapObstacles,
                                          mapScale,
                                          vector<vector<vector<double>>>(curProbs.begin() + i - seqLen, curProbs.begin() + i),
                                          vector<double>(curStepDists.begin() + i - seqLen, curStepDists.begin() + i),
                                          iObsVec,
                                          iLocIdxToRandVarClusterId,
                                          iVarVals);
                    
                    vector<LocationXY> infLoc = inferLocations(seqLen,
                                                               iPgm,
                                                               iObsVec,
                                                               iLocIdxToRandVarClusterId);
    
                    double dx = infLoc.back().x - curTrajLocations[i - 1].locationXY.x;
                    double dy = infLoc.back().y - curTrajLocations[i - 1].locationXY.y;
                    double curError = sqrt(dx*dx + dy*dy);
                    errorSum += curError;
                    errors.push_back(curError);
                    ++errorCnt;
                    
                    cout << "curError = " << curError << endl;
    
                    visualizeMapInfer(mapLocations,
                                      vector<LocationWiFi>(curTrajLocations.begin() + i - seqLen, curTrajLocations.begin() + i),
                                      infLoc,
                                      mapImage,
                                      mapScale);
                }
                if(errorCnt > 0) {
                    cout << "mean error = " << errorSum / errorCnt << endl;
                }
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
    catch(char const *str){
        cout << "Catch const char* in main(): " << str << endl;
        return -1;
    }
    catch(std::exception& e){
        cout << "Catch std exception in main(): " << e.what() << endl;
    }
    catch(...){
        cout << "Catch ... in main()" << endl;
        return -1;
    }
}