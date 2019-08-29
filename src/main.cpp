#include <iostream>
#include <cmath>
#include <fstream>
#include <numeric>

#include <boost/filesystem.hpp>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

#include <tinyxml.h>

#include "pgm/Pgm.h"
#include "pgm/Inference.h"
#include "pgm/ParamEst.h"

#include "Utils.hpp"
#include "LocationWiFi.hpp"
#include "WiFiSeqFeatures.hpp"
#include "Stepometer.hpp"

using namespace std;


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

double orientIdxToOrient(int oIdx){
    return orientSectorLen * oIdx;
}

int orientToOrientIdx(double o){
    int oIdx = (int)((o + 2 * M_PI + orientSectorLen / 2.0) / orientSectorLen);
    oIdx = ((oIdx % orientSectors) + orientSectors) % orientSectors;
    return oIdx;
}

int mapGridToVal(int x, int y, int o){
    return o + orientSectors * x + orientSectors * mapGridSizeX * y;
}

void valToMapGrid(double val, int &xIdx, int &yIdx, int &oIdx){
    int valInt = (int)round(val);
    oIdx = valInt % orientSectors;
    valInt /= orientSectors;

    xIdx = valInt % mapGridSizeX;
    valInt /= mapGridSizeX;

    yIdx = valInt;
}

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
                    std::vector<double> &stepDists,
                    std::vector<double> &orientMeas)
{
    cout << "Reading trajectory" << endl;
    
    ifstream wifiFile((dirPath / "wifi.map").c_str());
    if(!wifiFile.is_open()){
        cout << "Error! Could not open " << (dirPath / "wifi.map").c_str() << " file" << endl;
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
        cout << "Error! Could not open " << (dirPath / "acc.map").c_str() << " file" << endl;
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
        static constexpr int winLen = 256;
        static constexpr double stepLen = 0.7;
        static constexpr double accSampFreq = 200;
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

    orientMeas.clear();

    ifstream orientFile((dirPath / "orient.map").c_str());
    if(!orientFile.is_open()){
        cout << "Error! Could not open " << (dirPath / "orient.map").c_str() << " file" << endl;
    }
    vector<double> orientSamp;
    vector<uint64_t> orientSampTs;
    while(!orientFile.eof() && !orientFile.fail()){
        uint64_t timestamp;
        double qx, qy, qz, qw, accuracy;
        orientFile >> timestamp >> qx >> qy >> qz >> qw >> accuracy;
        if(!orientFile.fail()){
            if(accuracy < 0.3) {
                Eigen::Quaterniond q(qw, qx, qy, qz);
                Eigen::Matrix3d rotationMatrix = q.toRotationMatrix();

                // Getting the yaw angle from full orientation
                double heading = -atan2(rotationMatrix(1, 0), rotationMatrix(0, 0));
//            double heading = M_PI_2 - atan2(-rotationMatrix(1,2), rotationMatrix(0,2));

                orientSamp.push_back(heading);
                orientSampTs.push_back(timestamp);
            }
        }
    }

    {
        static constexpr int winLen = 20;

        int curOrientSampIdx = 0;
        double prevOrient = 0;
        for(int s = 0; s < wifiLocations.size(); ++s) {
            uint64_t curScanTs = wifiLocations[s].timestamp;

            while (curOrientSampIdx < orientSamp.size() && orientSampTs[curOrientSampIdx] < curScanTs) {
                ++curOrientSampIdx;
            }
            int startSamp = std::max(0, curOrientSampIdx - winLen/2);
            int endSamp = std::min((int)orientSamp.size(), curOrientSampIdx + winLen/2);

//            double curOrient = orientSamp[curOrientSampIdx];
            double curOrient = Utils::meanOrient(std::vector<double>(orientSamp.begin() + startSamp,
                                                                     orientSamp.begin() + endSamp));
//            orientMeas.push_back(Utils::toPiRange(curOrient - prevOrient));
            orientMeas.push_back(curOrient);

            prevOrient = curOrient;
        }

        double orientOffsetSin = 0;
        double orientOffsetCos = 0;
        for(int s = 0; s < orientMeas.size(); ++s) {
            double locX = wifiLocations[s].locationXY.x;
            double locY = wifiLocations[s].locationXY.y;
            double orientMap = 0.0;
            if(s < orientMeas.size() - 1){
                double nextLocX = wifiLocations[s + 1].locationXY.x;
                double nextLocY = wifiLocations[s + 1].locationXY.y;
                double dx = nextLocX - locX;
                double dy = nextLocY - locY;
                orientMap = atan2(dy, dx);
            }

            double curOrientOffset = Utils::angDiff(orientMeas[s], orientMap);
            orientOffsetSin += sin(curOrientOffset);
            orientOffsetCos += cos(curOrientOffset);
        }

        // mean using sin and cos
        orientOffsetSin /= orientMeas.size();
        orientOffsetCos /= orientMeas.size();

        double orientOffset = 0;
        // if not spread uniformly on the circle
        if(orientOffsetSin*orientOffsetSin + orientOffsetCos*orientOffsetCos > 0.01) {
            orientOffset = atan2(orientOffsetSin, orientOffsetCos);
        }

        // remove offset
        for(int s = 0; s < orientMeas.size(); ++s){
            orientMeas[s] = Utils::angDiff(orientMeas[s], orientOffset);
        }
    }

    cout << "Trajectory read" << endl;
}



LocationXY wknn(const std::vector<LocationWiFi> &database,
                const LocationWiFi &scan,
                int k,
                double& meanError)
{
    vector<pair<double, int>> errors;
    for(int d = 0; d < database.size(); ++d){
        pair<double, int> curError = errorL2(scan, database[d]);
        double sharedPercentA = (double) curError.second / scan.wifiScans.size();
        double sharedPercentB = (double) curError.second / database[d].wifiScans.size();
        
        if (sharedPercentA > sharedPercentThreshold &&
            sharedPercentB > sharedPercentThreshold)
        {
            errors.emplace_back(curError.first, d);
        }
    }
    sort(errors.begin(), errors.end());
    double wSum = 0.0;
    LocationXY retLoc(0.0, 0.0);
    meanError = 0.0;
    int meanErrorCnt = 0;
    for(int e = 0; e < min(k, (int)errors.size()); ++e){
        double w = 1.0 / (errors[e].first + 0.0001);
        int d = errors[e].second;
        retLoc.x += database[d].locationXY.x * w;
        retLoc.y += database[d].locationXY.y * w;
        wSum += w;
        
        meanError += errors[e].first;
        ++meanErrorCnt;
    }
    if(wSum > 0.0){
        retLoc.x /= wSum;
        retLoc.y /= wSum;
    }
    if(meanErrorCnt > 0){
        meanError /= meanErrorCnt;
    }
    else{
        meanError = 1e9;
    }
    
    return retLoc;
}

std::vector<std::vector<double>> locationProb(const LocationWiFi &loc,
                            const std::vector<LocationWiFi> &database)
{

//    int mapGridSizeX = ceil((mapMaxX - mapMinX) / mapGrid);
//    int mapGridSizeY = ceil((mapMaxY - mapMinY) / mapGrid);
    vector<vector<double>> prob(mapGridSizeY, vector<double>(mapGridSizeX, 0.01));
    
    double meanErrorWknn = 0.0;
    LocationXY locWknn = wknn(database, loc, wknnk, meanErrorWknn);
    if(meanErrorWknn < 100) {
        for (int mapYIdx = 0; mapYIdx < prob.size(); ++mapYIdx) {
            for (int mapXIdx = 0; mapXIdx < prob[mapYIdx].size(); ++mapXIdx) {
                LocationXY mapCoord = mapGridToCoord(mapXIdx, mapYIdx);

//                    cout << "mapCoord = (" << mapCoord.x << ", " << mapCoord.y << ")" << endl;
//                    cout << "databaseLoc.locationXY = (" << databaseLoc.locationXY.x << ", " << databaseLoc.locationXY.y << ")" << endl;
                double dx = mapCoord.x - locWknn.x;
                double dy = mapCoord.y - locWknn.y;
                double expVal = -((dx * dx + dy * dy) / (wifiSigma * wifiSigma));
                expVal += -meanErrorWknn / (errorSigma * errorSigma);

//                    cout << "expVal = " << expVal << endl;
                prob[mapYIdx][mapXIdx] += exp(expVal) * probScale;
            }
        }
    }
    
//    int nMatchedScans = 0;
//    for (const LocationWiFi &databaseLoc : database) {
//        pair<double, int> error = errorL2(loc, databaseLoc);
//        double sharedPercentA = (double) error.second / loc.wifiScans.size();
//        double sharedPercentB = (double) error.second / databaseLoc.wifiScans.size();
//
//        if (sharedPercentA > sharedPercentThreshold &&
//            sharedPercentB > sharedPercentThreshold)
//        {
////            cout << "matched scan" << endl;
//            // adding Gaussian kernel placed at databaseLoc and weighted with error
//            for (int mapYIdx = 0; mapYIdx < prob.size(); ++mapYIdx) {
//                for (int mapXIdx = 0; mapXIdx < prob[mapYIdx].size(); ++mapXIdx) {
//                    LocationXY mapCoord = mapGridToCoord(mapXIdx, mapYIdx);
//
////                    cout << "mapCoord = (" << mapCoord.x << ", " << mapCoord.y << ")" << endl;
////                    cout << "databaseLoc.locationXY = (" << databaseLoc.locationXY.x << ", " << databaseLoc.locationXY.y << ")" << endl;
//                    double dx = mapCoord.x - databaseLoc.locationXY.x;
//                    double dy = mapCoord.y - databaseLoc.locationXY.y;
//                    double expVal = -((dx * dx + dy * dy) / (wifiSigma * wifiSigma));
//                    expVal += -error.first / (errorSigma * errorSigma);
//
////                    cout << "expVal = " << expVal << endl;
//                    prob[mapYIdx][mapXIdx] += exp(expVal) * probScale;
//                }
//            }
//            ++nMatchedScans;
//        }
//    }
//    cout << "nMatchedScans = " << nMatchedScans << endl;
    return prob;
}

void visualizeMapProb(const std::vector<LocationWiFi> &database,
                      const std::vector<std::vector<double>> &prob,
                      const double &orient,
                      const double &varVal,
                      const cv::Mat &mapImage,
                      const cv::Mat &mapObstacle,
                      const double &mapScale)
{
    cv::Mat mapVis = mapImage.clone();
    for(const LocationWiFi &curLoc : database){
        cv::Point2d pt(curLoc.locationXY.x, curLoc.locationXY.y);
//        cv::circle(mapVis, pt * mapScale, 5, cv::Scalar(0, 0, 255), CV_FILLED);
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
    
    cv::Mat probValScaled;
    probVal.convertTo(probValScaled, CV_8U, 255);
    cv::applyColorMap(probValScaled, probVis, cv::COLORMAP_JET);
    
    {
        int mapXIdx, mapYIdx, oIdx;
        valToMapGrid(varVal, mapXIdx, mapYIdx, oIdx);
        LocationXY loc = mapGridToCoord(mapXIdx, mapYIdx);
        double orientVal = orientIdxToOrient(oIdx);
        cv::Point2d pt(loc.x, loc.y);
        static constexpr double radius = 2.0;
        cv::Point2d ptDir(loc.x + radius * cos(orientVal), loc.y + radius * sin(orientVal));
//        cv::Point2d ptDir(loc.x + radius * cos(orient), loc.y + radius * sin(orient));

        cv::circle(mapVis, pt * mapScale, radius * mapScale, cv::Scalar(0, 255, 0));
        cv::line(mapVis, pt * mapScale, ptDir * mapScale, cv::Scalar(0, 255, 0), 2);
    }
    
    {
        probVal.setTo(cv::Scalar(0.25, 0.25, 0.25));
        probVis.setTo(cv::Scalar(255, 255, 255));
//        probVis.setTo(cv::Scalar(0, 255, 0), mapObstacle == 0);
    }
    
    cv::Mat probVisBlended(probVis.clone());
    cv::Mat mapVisBlended(mapVis.clone());
    for(int r = 0; r < probVisBlended.rows; ++r){
        for(int c = 0; c < probVisBlended.cols; ++c){
            probVisBlended.at<cv::Vec3b>(r, c) *= probVal.at<float>(r, c);
            mapVisBlended.at<cv::Vec3b>(r, c) *= 1.0 - probVal.at<float>(r, c);
        }
    }
    cv::Mat vis = mapVisBlended + probVisBlended;
    cv::resize(vis, vis, cv::Size(0, 0), 0.5, 0.5);
    
    cv::imshow("map", vis);
    
//    cv::waitKey(100);
    cv::waitKey(-1);
}

void visualizeMapInfer(const std::vector<LocationWiFi> &database,
                       const std::vector<LocationWiFi> &trajLocations,
                       const std::vector<LocationXY> &inferLocations,
                       const std::vector<LocationXY> &compLocations,
                       const cv::Mat &mapImage,
                       const double &mapScale,
                       bool stop = false)
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
        {
            const LocationXY &infLoc = inferLocations[i];
            cv::Point2d pt(infLoc.x, infLoc.y);
            cv::circle(mapVis, pt * mapScale, 5, cv::Scalar(255, 0, 0), CV_FILLED);
        
//            cv::putText(mapVis,
//                        to_string(i),
//                        pt * mapScale,
//                        cv::FONT_HERSHEY_PLAIN,
//                        1,
//                        cv::Scalar(0, 0, 0));
            if (i > 0) {
                const LocationXY &prevInfLoc = inferLocations[i - 1];
                cv::Point2d prevPt(prevInfLoc.x, prevInfLoc.y);
            
                cv::line(mapVis, prevPt * mapScale, pt * mapScale, cv::Scalar(255, 0, 0), 2);
            }
        }
        {
            const LocationXY &compLoc = compLocations[i];
            cv::Point2d pt(compLoc.x, compLoc.y);
            cv::circle(mapVis, pt * mapScale, 5, cv::Scalar(255, 0, 255), CV_FILLED);
    
//            cv::putText(mapVis,
//                        to_string(i),
//                        pt * mapScale,
//                        cv::FONT_HERSHEY_PLAIN,
//                        1,
//                        cv::Scalar(0, 0, 0));
            if (i > 0) {
                const LocationXY &prevCompLoc = compLocations[i - 1];
                cv::Point2d prevPt(prevCompLoc.x, prevCompLoc.y);
        
                cv::line(mapVis, prevPt * mapScale, pt * mapScale, cv::Scalar(255, 0, 255), 2);
            }
        }
        
        {
            const LocationXY &gtLoc = trajLocations[i].locationXY;
            cv::Point2d gtPt(gtLoc.x, gtLoc.y);
            cv::circle(mapVis, gtPt * mapScale, 5, cv::Scalar(0, 255, 0), CV_FILLED);
    
            if (i > 0) {
                const LocationXY &prevGtLoc = trajLocations[i - 1].locationXY;
                cv::Point2d prevGtPt(prevGtLoc.x, prevGtLoc.y);
        
                cv::line(mapVis, prevGtPt * mapScale, gtPt * mapScale, cv::Scalar(0, 255, 0), 2);
            }
        }
    }
    
    cv::resize(mapVis, mapVis, cv::Size(0, 0), 0.5, 0.5);
    
    cv::imshow("map", mapVis);
    
    if(stop){
        cv::waitKey();
    }
    else {
        cv::waitKey(100);
    }
}

Pgm buildPgm(const std::vector<LocationWiFi> &wifiLocations,
             const cv::Mat &obstacles,
             const double &mapScale,
             const std::vector<std::vector<std::vector<double>>> &probs,
             const std::vector<double> &stepDists,
             const std::vector<double> &orientMap,
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
                    for(int oIdx = 0; oIdx < orientSectors; ++oIdx) {
                        int val = mapGridToVal(mapXIdx, mapYIdx, oIdx);
                        forbiddenVals.insert(val);
                    }
                }
            }
        }
//        cout << "forbiddenVals.size() = " << forbiddenVals.size() << endl;
    }
    
    // observation vector
    // location coordinates
    vector<double> xCoords, yCoords, oCoords;
//    vector<double> rvVals;
    int mapSize = 0;
    for (int mapYIdx = 0; mapYIdx < probs[0].size(); ++mapYIdx) {
        for (int mapXIdx = 0; mapXIdx < probs[0][mapYIdx].size(); ++mapXIdx) {
            LocationXY mapCoord = mapGridToCoord(mapXIdx, mapYIdx);
            for(int oIdx = 0; oIdx < orientSectors; ++oIdx) {
                double curOrient = orientIdxToOrient(oIdx);

                xCoords.push_back(mapCoord.x);
                yCoords.push_back(mapCoord.y);
                oCoords.push_back(curOrient);
                ++mapSize;
            }
        }
    }
    int obsVecStartLoc = 0;
    obsVec.insert(obsVec.end(), xCoords.begin(), xCoords.end());
    obsVec.insert(obsVec.end(), yCoords.begin(), yCoords.end());
    obsVec.insert(obsVec.end(), oCoords.begin(), oCoords.end());
    
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

    int obsVecStartOrientMeas = obsVec.size();
    for(int i = 0; i < probs.size(); ++i){
        obsVec.push_back(orientMap[i]);
    }

    int obsVecStartOrient = obsVec.size();
    for(int oIdx = 0; oIdx < orientSectors; ++oIdx) {
        double curOrient = orientIdxToOrient(oIdx);
        obsVec.push_back(curOrient);
    }


    // random variables
    int nextRandVarId = 0;
    // last position has the same orientation assigned as the previous position
    int prevLocOIdx = 0;
    for(int i = 0; i < probs.size(); ++i){
        locIdxToVarClusterId[i] = nextRandVarId;
    
        double maxProb = 0;
        for (int mapYIdx = 0; mapYIdx < probs[i].size(); ++mapYIdx) {
            for (int mapXIdx = 0; mapXIdx < probs[i][mapYIdx].size(); ++mapXIdx) {
                maxProb = max(maxProb, probs[i][mapYIdx][mapXIdx]);
            }
        }

        double locX = wifiLocations[i].locationXY.x;
        double locY = wifiLocations[i].locationXY.y;
        int locOIdx = prevLocOIdx;
        if( i < probs.size() - 1){
            double nextLocX = wifiLocations[i + 1].locationXY.x;
            double nextLocY = wifiLocations[i + 1].locationXY.y;
            double dx = nextLocX - locX;
            double dy = nextLocY - locY;
            double orient = atan2(dy, dx);
            locOIdx = orientToOrientIdx(orient);

            {
                double curDist = stepDists[i + 1];
                double curOrient = orientIdxToOrient(locOIdx);
                double nextLocXPred = locX + curDist * cos(curOrient);
                double nextLocYPred = locY + curDist * sin(curOrient);

                double error = sqrt((nextLocX - nextLocXPred)*(nextLocX - nextLocXPred) +
                        (nextLocY - nextLocYPred)*(nextLocY - nextLocYPred));
                double orientError = Utils::angDiff(curOrient, orientMap[i]);

                cout << "step = " << i << ", error = " << error << ", orientError  = " << orientError << endl;
//                cout << "loc = (" << locX << ", " << locY << "), nextLoc = (" << nextLocX << ", " << nextLocY << ")" <<
//                        ", d = (" << curDist * cos(curOrient) << ", " << curDist * sin(curOrient) << ")" << endl;
//                cout << "step = " << i << ", dist = " << curDist << ", error = " << error << endl;
            }

            prevLocOIdx = locOIdx;
        }

        pair<int, int> mapIdx = mapCoordToGrid(locX, locY);
        double varVal = mapGridToVal(mapIdx.first, mapIdx.second, locOIdx);
        double closestDist = std::numeric_limits<double>::max();
        int closestXIdx = mapIdx.first;
        int closestYIdx = mapIdx.second;
        
        double curProbThresh = min(probThresh, maxProb / probRatio);
        vector<double> rvVals;
        for (int mapYIdx = 0; mapYIdx < probs[i].size(); ++mapYIdx) {
            for (int mapXIdx = 0; mapXIdx < probs[i][mapYIdx].size(); ++mapXIdx) {
                // check any orientation
                int valCheck = mapGridToVal(mapXIdx, mapYIdx, 0);

                bool validLoc = (probs[i][mapYIdx][mapXIdx] >= curProbThresh) && (forbiddenVals.count(valCheck) == 0);
                for(int oIdx = 0; oIdx < orientSectors; ++oIdx) {
                    if (validLoc){
                        int val = mapGridToVal(mapXIdx, mapYIdx, oIdx);
                        rvVals.push_back(val);
                    }
                }
                if (validLoc){
                    LocationXY loc = mapGridToCoord(mapXIdx, mapYIdx);
                    double dist = sqrt((loc.x - locX) * (loc.x - locX) + (loc.y - locY) * (loc.y - locY));
                    if (dist < closestDist) {
                        closestDist = dist;
                        closestXIdx = mapXIdx;
                        closestYIdx = mapYIdx;
                    }
                }
            }
        }
        cout << "rvVals.size() = " << rvVals.size() << endl;
        
        if(find(rvVals.begin(), rvVals.end(), varVal) == rvVals.end()){
            cout << "Warning - varVal not in rvVals, substituting with the closest" << endl;
            cout << "Closest distance = " << closestDist << endl;
            
            varVal = mapGridToVal(closestXIdx, closestYIdx, locOIdx);
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

    // orient features
    vector<shared_ptr<Feature>> orientFeats;
    for(int i = 0; i < probs.size(); ++i){
        vector<int> curObsVecIdxs(1 + mapSize);
        // orientation measurement
        curObsVecIdxs[0] = obsVecStartOrientMeas + i;
        // orientations for each location
        iota(curObsVecIdxs.begin() + 1,
             curObsVecIdxs.begin() + 1 + mapSize,
             obsVecStartLoc + 2 * mapSize);

        shared_ptr<Feature> curFeat(new OrientFeature(nextFeatId++,
                                                      nextParamId,
                                                      vector<shared_ptr<RandVar>>{randVars[i]},
                                                      curObsVecIdxs,
                                                      orientSigma));

        orientFeats.push_back(curFeat);
        feats.push_back(curFeat);
    }
    ++nextParamId;

    // move features
    vector<shared_ptr<Feature>> moveFeats;
    // skip first one, as there is no distance from previous location
    for(int i = 1; i < probs.size(); ++i){
        vector<int> curObsVecIdxs(1 + 3 * mapSize);
        curObsVecIdxs[0] = obsVecStartDist + i;
        // x coordinates
        iota(curObsVecIdxs.begin() + 1, curObsVecIdxs.begin() + 1 + mapSize, obsVecStartLoc);
        // y coordinates
        iota(curObsVecIdxs.begin() + 1 + mapSize, curObsVecIdxs.begin() + 1 + 2 * mapSize, obsVecStartLoc + mapSize);
        // orientation
        iota(curObsVecIdxs.begin() + 1 + 2 * mapSize, curObsVecIdxs.begin() + 1 + 3 * mapSize, obsVecStartLoc + 2 * mapSize);
        
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

//    // orient features
//    vector<shared_ptr<Feature>> orientFeats;
//    // skip first one, as there is no orientation difference from previous location
//    for(int i = 1; i < probs.size(); ++i){
//        vector<int> curObsVecIdxs(1 + 2*mapSize);
//        curObsVecIdxs[0] = obsVecStartDist + i;
//        // x coordinates
//        iota(curObsVecIdxs.begin() + 1, curObsVecIdxs.begin() + 1 + mapSize, obsVecStartLoc);
//        // y coordinates
//        iota(curObsVecIdxs.begin() + 1 + mapSize, curObsVecIdxs.begin() + 1 + 2*mapSize, obsVecStartLoc + mapSize);
//
//        shared_ptr<Feature> curFeat(new MoveFeature(nextFeatId++,
//                                                    nextParamId,
//                                                    vector<shared_ptr<RandVar>>{randVars[i-1], randVars[i]},
//                                                    curObsVecIdxs,
//                                                    mapSize,
//                                                    distSigma));
//
//        moveFeats.push_back(curFeat);
//        feats.push_back(curFeat);
//    }
//    ++nextParamId;
    
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

    vector<shared_ptr<Cluster>> orientFeatClusters;
    for(int i = 0; i < probs.size(); ++i){
        shared_ptr<Cluster> curCluster(new Cluster(nextClusterId++,
                                                   vector<shared_ptr<Feature>>{orientFeats[i]},
                                                   vector<shared_ptr<RandVar>>{randVars[i]}));

        orientFeatClusters.push_back(curCluster);
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

    // edges from random variable clusters to orientation feature clusters
    for(int i = 0; i < probs.size(); ++i){
        pgm.addEdgeToPgm(rvClusters[i], orientFeatClusters[i], vector<shared_ptr<RandVar>>{randVars[i]});
    }
    
    for(int i = 1; i < probs.size(); ++i){
        pgm.addEdgeToPgm(rvClusters[i - 1], moveFeatClusters[i - 1], vector<shared_ptr<RandVar>>{randVars[i - 1]});
        pgm.addEdgeToPgm(rvClusters[i], moveFeatClusters[i - 1], vector<shared_ptr<RandVar>>{randVars[i]});
    }
    
    // prob map by wknn
//    pgm.params() = vector<double>{0.885548, 19.7255};
    // prob map by MoG
//    pgm.params() = vector<double>{1.64752, 21.8728};
//    pgm.params() = vector<double>{1.0, 1.0};
    
    pgm.params() = vector<double>{5.06612, 0.0721242, 1.09273};

    
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
    
//    cout << "calibrated = " << calibrated << endl;
    
    vector<vector<double>> retVals = Inference::decodeMAP(pgm,
                                                         marg,
                                                         msgs,
                                                         params,
                                                         obsVec);
    
    for(int i = 0; i < nloc; ++i){
        int varClusterId = locIdxToVarClusterId.at(i);
        const vector<double> &curVals = retVals[varClusterId];
        
        int curLoc = curVals.front();
        
        int mapXIdx, mapYIdx, oIdx;
        valToMapGrid(curLoc, mapXIdx, mapYIdx, oIdx);
        
        LocationXY curLocXY = mapGridToCoord(mapXIdx, mapYIdx);
        
        retLoc.push_back(curLocXY);
    }
    
    return retLoc;
}

void removeNotMatchedLocations(std::vector<LocationWiFi> &wifiLocations,
                               std::vector<double> &stepDists,
                               std::vector<double> &orients,
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
            orients.erase(orients.begin() + i);
            probs.erase(probs.begin() + i);
            
            --i;
        }
    }
}



int main() {
    try{
        static constexpr bool estimateParams = true;
        static constexpr bool infer = false;
//        static constexpr bool estimateParams = false;
//        static constexpr bool infer = true;
        
        static constexpr int seqLen = 10;
        
//        boost::filesystem::path mapDirPath("../res/Maps/PUTMC_Lenovo_18_05_21_full");
        boost::filesystem::path mapDirPath("../res/IGL/PUTMC_Floor3_Xperia_map");
        
        cv::Mat mapImage;
        cv::Mat mapObstacles;
        double mapScale;
        vector<LocationWiFi> mapLocations = readMap(mapDirPath, mapImage, mapObstacles, mapScale);
        
//        vector<boost::filesystem::path> trajDirPaths{"../res/Trajectories/traj1",
//                                                     "../res/Trajectories/traj2",
//                                                     "../res/Trajectories/traj3"};
//        vector<boost::filesystem::path> trajDirPaths{"../res/Trajectories/traj4",
//                                                     "../res/Trajectories/traj5",
//                                                     "../res/Trajectories/traj6"};
        vector<boost::filesystem::path> trajDirPaths{"../res/IGL/PUTMC_Floor3_Xperia_trajs/xperia_traj1",
                                                     "../res/IGL/PUTMC_Floor3_Xperia_trajs/xperia_traj2",
                                                     "../res/IGL/PUTMC_Floor3_Xperia_trajs/xperia_traj3"};
//        vector<boost::filesystem::path> trajDirPaths{"../res/IGL/PUTMC_Floor3_Xperia_trajs/xperia_traj4",
//                                                     "../res/IGL/PUTMC_Floor3_Xperia_trajs/xperia_traj5"};
        
        vector<Pgm> pgms;
        vector<vector<double>> obsVecs;
        vector<vector<double>> varVals;
    
    
        ofstream errorsFile("../log/errors");
        ofstream errorsAllFile("../log/errors_all");
    
        chrono::nanoseconds infDur(0);
        int infTimeCnt = 0;
        
        for(int t = 0; t < trajDirPaths.size(); ++t) {
        
            vector<LocationWiFi> curTrajLocations;
            vector<double> curStepDists;
            vector<double> curOrients;
            readTrajectory(trajDirPaths[t], curTrajLocations, curStepDists, curOrients);
        
            vector<vector<vector<double>>> curProbs;
            for (int i = 0; i < curTrajLocations.size(); ++i) {
                vector<vector<double>> curProb = locationProb(curTrajLocations[i], mapLocations);
            
                curProbs.push_back(curProb);
            }
        
            removeNotMatchedLocations(curTrajLocations,
                                      curStepDists,
                                      curOrients,
                                      curProbs);
        
            vector<double> curObsVec;
            map<int, int> curLocIdxToRandVarClusterId;
            vector<double> curVarVals;
            Pgm curPgm = buildPgm(curTrajLocations,
                               mapObstacles,
                               mapScale,
                               curProbs,
                               curStepDists,
                               curOrients,
                               curObsVec,
                               curLocIdxToRandVarClusterId,
                               curVarVals);
        
            for (int i = 0; i < curTrajLocations.size(); ++i) {
                visualizeMapProb(mapLocations,
                                 curProbs[i],
                                 curOrients[i],
                                 curVarVals[i],
                                 mapImage,
                                 mapObstacles,
                                 mapScale);
            }
        
            if(infer) {
    
                {
                    vector<LocationXY> infLocAll;
                    vector<LocationXY> wknnLocAll;
                    
                    vector<double> errors;
                    double errorSum = 0;
                    int errorCnt = 0;

                    vector<double> errorsComp;
                    double errorSumComp = 0;
                    int errorCntComp = 0;

                    for (int i = seqLen - 1; i < curTrajLocations.size(); ++i) {
                        
                        chrono::steady_clock::time_point startTime = chrono::steady_clock::now();
                        vector<double> iObsVec;
                        map<int, int> iLocIdxToRandVarClusterId;
                        vector<double> iVarVals;
                        Pgm iPgm = buildPgm(vector<LocationWiFi>(
                                curTrajLocations.begin() + i - seqLen + 1,
                                curTrajLocations.begin() + i + 1),
                                            mapObstacles,
                                            mapScale,
                                            vector<vector<vector<double>>>(
                                                    curProbs.begin() + i - seqLen + 1,
                                                    curProbs.begin() + i + 1),
                                            vector<double>(curStepDists.begin() + i - seqLen + 1,
                                                           curStepDists.begin() + i + 1),
                                            vector<double>(curOrients.begin() + i - seqLen + 1,
                                                           curOrients.begin() + i + 1),
                                            iObsVec,
                                            iLocIdxToRandVarClusterId,
                                            iVarVals);

                        vector<LocationXY> infLoc = inferLocations(seqLen,
                                                                   iPgm,
                                                                   iObsVec,
                                                                   iLocIdxToRandVarClusterId);
                        
                        chrono::steady_clock::time_point endTime = chrono::steady_clock::now();
                        infDur += endTime - startTime;
                        ++infTimeCnt;
                        
                        infLocAll.push_back(infLoc.back());
                        {
                            double dx = infLoc.back().x - curTrajLocations[i].locationXY.x;
                            double dy = infLoc.back().y - curTrajLocations[i].locationXY.y;
                            double curError = sqrt(dx * dx + dy * dy);
                            errorSum += curError;
                            errors.push_back(curError);
                            ++errorCnt;

                            cout << "curError = " << curError << endl;
                        }

                        vector<LocationXY> wknnLoc;
                        for (int t = i - seqLen + 1; t <= i; ++t) {
                            double meanErrorWknn = 0.0;
                            LocationXY curLoc = wknn(mapLocations,
                                                     curTrajLocations[t],
                                                     wknnk,
                                                     meanErrorWknn);
//                        cout << "curLoc = (" << curLoc.x << ", " << curLoc.y << ")" << endl;
                            wknnLoc.push_back(curLoc);

                        }
    
                        wknnLocAll.push_back(wknnLoc.back());
                        {
                            double dx = wknnLoc.back().x - curTrajLocations[i].locationXY.x;
                            double dy = wknnLoc.back().y - curTrajLocations[i].locationXY.y;
                            double curErrorComp = sqrt(dx * dx + dy * dy);
                            errorSumComp += curErrorComp;
                            errorsComp.push_back(curErrorComp);
                            ++errorCntComp;

                            cout << "curErrorComp = " << curErrorComp << endl;
                        }

                        visualizeMapInfer(mapLocations,
                                          vector<LocationWiFi>(
                                                  curTrajLocations.begin() + i - seqLen + 1,
                                                  curTrajLocations.begin() + i + 1),
                                          infLoc,
                                          wknnLoc,
                                          mapImage,
                                          mapScale);
                    }
                    cout << endl;
                    if (errorCnt > 0) {
                        cout << "mean error = " << errorSum / errorCnt << endl;
                    }
                    if (errorCntComp > 0) {
                        cout << "mean comp error = " << errorSumComp / errorCntComp << endl;
                    }
                    cout << endl;
    
                    visualizeMapInfer(mapLocations,
                                      vector<LocationWiFi>(
                                              curTrajLocations.begin() + seqLen - 1,
                                              curTrajLocations.end()),
                                      infLocAll,
                                      wknnLocAll,
                                      mapImage,
                                      mapScale,
                                      true);
                    
                    for (int e = 0; e < errors.size(); ++e) {
                        errorsFile << errors[e] << " " << errorsComp[e] << endl;
                    }

                }
                {
                    vector<double> errors;
                    double errorSum = 0;
                    int errorCnt = 0;
    
                    vector<double> errorsComp;
                    double errorSumComp = 0;
                    int errorCntComp = 0;
    
                    
                    vector<LocationXY> infLoc = inferLocations(curTrajLocations.size(),
                                                               curPgm,
                                                               curObsVec,
                                                               curLocIdxToRandVarClusterId);
                    for (int t = 0; t < curTrajLocations.size(); ++t) {
                        double dx = infLoc[t].x - curTrajLocations[t].locationXY.x;
                        double dy = infLoc[t].y - curTrajLocations[t].locationXY.y;
                        double curError = sqrt(dx * dx + dy * dy);
                        errorSum += curError;
                        errors.push_back(curError);
                        ++errorCnt;
        
//                        cout << "curError = " << curError << endl;
                    }
    
                    vector<LocationXY> wknnLoc;
                    for (int t = 0; t < curTrajLocations.size(); ++t) {
                        double meanErrorWknn = 0.0;
                        LocationXY curLoc = wknn(mapLocations,
                                                 curTrajLocations[t],
                                                 wknnk,
                                                 meanErrorWknn);
//                        cout << "curLoc = (" << curLoc.x << ", " << curLoc.y << ")" << endl;
                        wknnLoc.push_back(curLoc);
        
                    }
    
                    for (int t = 0; t < curTrajLocations.size(); ++t) {
                        double dx = wknnLoc[t].x - curTrajLocations[t].locationXY.x;
                        double dy = wknnLoc[t].y - curTrajLocations[t].locationXY.y;
                        double curErrorComp = sqrt(dx * dx + dy * dy);
                        errorSumComp += curErrorComp;
                        errorsComp.push_back(curErrorComp);
                        ++errorCntComp;
        
//                        cout << "curErrorComp = " << curErrorComp << endl;
                    }
    
                    cout << endl;
                    if (errorCnt > 0) {
                        cout << "mean error = " << errorSum / errorCnt << endl;
                    }
                    if (errorCntComp > 0) {
                        cout << "mean comp error = " << errorSumComp / errorCntComp << endl;
                    }
                    cout << endl;
                    
                    visualizeMapInfer(mapLocations,
                                      curTrajLocations,
                                      infLoc,
                                      wknnLoc,
                                      mapImage,
                                      mapScale,
                                      true);
                    
                    for (int e = 0; e < errors.size(); ++e) {
                        errorsAllFile << errors[e] << " " << errorsComp[e] << endl;
                    }
                }
            }
            
            pgms.push_back(curPgm);
            varVals.push_back(curVarVals);
            obsVecs.push_back(curObsVec);
        }
        
        cout << "mean inference time: " << (double)infDur.count() / infTimeCnt / 1e6 << " ms" << endl;
        
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