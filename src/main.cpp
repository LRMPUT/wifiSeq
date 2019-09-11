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
#include "LocationWifi.hpp"
#include "WiFiSeqFeatures.hpp"
#include "Stepometer.hpp"
#include "Graph.hpp"
#include "fastable/FastABLE.h"


using namespace std;


pair<double, int> errorL2(const LocationWifi &lhs, const LocationWifi &rhs){
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

void selectPolygonPixels(const std::vector<cv::Point2i> &polygon,
                         int regionId,
                         cv::Mat& regionsOnImage)
{
    int polyCnt[] = {(int)polygon.size()};
    const cv::Point2i* points[] = {polygon.data()};
    fillPoly(regionsOnImage, points, polyCnt, 1, cv::Scalar(regionId));
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

        selectPolygonPixels(poly, label, annotations);
        
        pObject = pObject->NextSiblingElement("object");
    }
    
    return annotations;
}



LocationXY wknn(const std::vector<LocationWifi> &database,
                const LocationWifi &scan,
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
    LocationXY retLoc(0.0, 0.0, -1);
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



std::vector<std::vector<double>> locationWifiProb(const LocationWifi &loc,
                                                  const std::vector<LocationWifi> &database) {

    vector<vector<double>> prob(mapGridSizeY, vector<double>(mapGridSizeX, minProb));

    if(useWknn) {
        double meanErrorWknn = 0.0;
        LocationXY locWknn = wknn(database, loc, wknnk, meanErrorWknn);
        if(meanErrorWknn < 100) {
            for (int mapYIdx = 0; mapYIdx < prob.size(); ++mapYIdx) {
                for (int mapXIdx = 0; mapXIdx < prob[mapYIdx].size(); ++mapXIdx) {
                    LocationXY mapCoord = Utils::mapGridToCoord(mapXIdx, mapYIdx);

                    double dx = mapCoord.x - locWknn.x;
                    double dy = mapCoord.y - locWknn.y;
                    double expVal = -((dx * dx + dy * dy) / (wifiSigma * wifiSigma));
                    expVal += -meanErrorWknn / (errorWifiSigma * errorWifiSigma);

                    prob[mapYIdx][mapXIdx] += exp(expVal) * probScale;
                }
            }
        }
    }
    else {
        int nMatchedScans = 0;
        for (const LocationWifi &databaseLoc : database) {
            pair<double, int> error = errorL2(loc, databaseLoc);
            double sharedPercentA = (double) error.second / loc.wifiScans.size();
            double sharedPercentB = (double) error.second / databaseLoc.wifiScans.size();

            if (sharedPercentA > sharedPercentThreshold &&
                sharedPercentB > sharedPercentThreshold) {
                // adding Gaussian kernel placed at databaseLoc and weighted with error
                for (int mapYIdx = 0; mapYIdx < prob.size(); ++mapYIdx) {
                    for (int mapXIdx = 0; mapXIdx < prob[mapYIdx].size(); ++mapXIdx) {
                        LocationXY mapCoord = Utils::mapGridToCoord(mapXIdx, mapYIdx);

                        double dx = mapCoord.x - databaseLoc.locationXY.x;
                        double dy = mapCoord.y - databaseLoc.locationXY.y;
                        double expVal = -((dx * dx + dy * dy) / (wifiSigma * wifiSigma));
                        expVal += -error.first / (errorWifiSigma * errorWifiSigma);

                        prob[mapYIdx][mapXIdx] += exp(expVal) * probScale;
                    }
                }
                ++nMatchedScans;
            }
        }
        cout << "nMatchedScans = " << nMatchedScans << endl;
    }
    return prob;
}

std::vector<std::vector<double>> locationImageProb(const LocationGeneral &loc,
                                                  const ImageRecognitionResult &imgRecogRes) {

    vector<vector<double>> prob(mapGridSizeY, vector<double>(mapGridSizeX, minProb));
    vector<pair<double, int>> weightIdx;
    for(int l = 0; l < imgRecogRes.matchingLocations.size(); ++l) {
        static constexpr double minVal = 0.2;
        static constexpr double maxVal = 1.0;
        double error = (imgRecogRes.matchingWeights[l] - minVal) / (maxVal - minVal);
        error = min(max(error, 0.0), 1.0);

        double curW = exp(-error / (errorImageSigma * errorImageSigma));
        weightIdx.emplace_back(curW, l);
    }
    sort(weightIdx.begin(), weightIdx.end(), std::greater<>());
    double maxW = 0.0;
    if(!weightIdx.empty()){
        maxW = weightIdx.front().first;
    }
//    cout << "maxW = " << maxW << endl;
    for(int l = 0; l < (int)weightIdx.size(); ++l) {
        double curW = weightIdx[l].first;
        int curIdx = weightIdx[l].second;

//        if(curW > maxW / 2.0) {
        {
            // adding Gaussian kernel placed at databaseLoc and weighted with error
            for (int mapYIdx = 0; mapYIdx < prob.size(); ++mapYIdx) {
                for (int mapXIdx = 0; mapXIdx < prob[mapYIdx].size(); ++mapXIdx) {
                    LocationXY mapCoord = Utils::mapGridToCoord(mapXIdx, mapYIdx);

                    double dx = mapCoord.x - imgRecogRes.matchingLocations[curIdx].x;
                    double dy = mapCoord.y - imgRecogRes.matchingLocations[curIdx].y;
                    double expVal = -((dx * dx + dy * dy) / (wifiSigma * wifiSigma));

                    prob[mapYIdx][mapXIdx] += curW * exp(expVal) * probScale;
                }
            }
        }
    }
    return prob;
}

void readMap(const boost::filesystem::path &dirPath,
             std::vector<LocationWifi> &wifiLocations,
             std::vector<LocationImage> &imageLocations,
             cv::Mat &mapImage,
             cv::Mat &obstacles,
             double &mapImageScale,
             set<int> &forbiddenVals,
             set<int> &allowedVals,
             shared_ptr<Graph> &graph) {

    cout << "Reading map" << endl;

    map<int, int> locWifiIdToIdx;

    {
        ifstream wifiFile((dirPath / "wifi.map").c_str());
        if (!wifiFile.is_open()) {
            cout << "Error! Could not open " << (dirPath / "wifi.map").c_str() << " file" << endl;
        }
        while (!wifiFile.eof() && !wifiFile.fail()) {
            LocationWifi curLoc;

            uint64_t timestamp = 0;
            int locId = 0;
            int nscans = 0;
            wifiFile >> timestamp >> locId >> nscans;
            if (!wifiFile.fail()) {
                curLoc.locationXY.id = locId;
                for (int i = 0; i < nscans; ++i) {
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
                    if (!wifiFile.fail()) {
                        curLoc.wifiScans.emplace_back(bssid, ssid, level, freq, localTimestamp);
                    }
                }

                locWifiIdToIdx[locId] = wifiLocations.size();
                wifiLocations.push_back(curLoc);
            }
        }
        cout << "read " << wifiLocations.size() << " WiFi locations" << endl;
    }

    map<int, int> locImageIdToIdx;
    {
        ifstream imageFile((dirPath / "imgs/images.map").c_str());
        if (!imageFile.is_open()) {
            cout << "Error! Could not open " << (dirPath / "imgs/images.map").c_str() << " file" << endl;
        }
        while (!imageFile.eof() && !imageFile.fail()) {
            LocationImage curLoc;

            uint64_t timestamp = 0;
            int locId = 0;
            string imageFilename;
            int segId = 0;
            imageFile >> timestamp >> locId >> imageFilename >> segId;
            if (!imageFile.fail()) {
                curLoc.segmentId = segId;
                curLoc.locationXY.id = locId;

                curLoc.image = cv::imread((dirPath / "imgs" / imageFilename).string());

                locImageIdToIdx[locId] = imageLocations.size();
                imageLocations.push_back(curLoc);
            }
        }
        cout << "read " << imageLocations.size() << " image locations" << endl;
    }

    ifstream positionsFile((dirPath / "positions.map").c_str());
    if(!positionsFile.is_open()){
        cout << "Error! Could not open " << (dirPath / "positions.map").c_str() << " file" << endl;
    }
    while(!positionsFile.eof() && !positionsFile.fail()){
        int id;
        double x, y;
        positionsFile >> id >> x >> y;
        if(!positionsFile.fail()){
            if(locWifiIdToIdx.count(id) > 0) {
                int idx = locWifiIdToIdx[id];
                wifiLocations[idx].locationXY = LocationXY(x, y, id);
            }
            if(locImageIdToIdx.count(id) > 0) {
                int idx = locImageIdToIdx[id];
                imageLocations[idx].locationXY = LocationXY(x, y, id);
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


    {
        for (int mapYIdx = 0; mapYIdx < mapGridSizeY; ++mapYIdx) {
            for (int mapXIdx = 0; mapXIdx < mapGridSizeX; ++mapXIdx) {
                LocationXY mapCoord = Utils::mapGridToCoord(mapXIdx, mapYIdx);

                cv::Point2d pt1(mapCoord.x - mapGrid / 2, mapCoord.y - mapGrid / 2);
                cv::Point2d pt2(mapCoord.x + mapGrid / 2, mapCoord.y + mapGrid / 2);
                cv::Point2i pt1i = pt1 * mapImageScale;
                cv::Point2i pt2i = pt2 * mapImageScale;
                pt1i.x = max(pt1i.x, 0);
                pt1i.y = max(pt1i.y, 0);
                pt2i.x = max(pt2i.x, 0);
                pt2i.y = max(pt2i.y, 0);
                pt1i.x = min(pt1i.x, obstacles.cols);
                pt1i.y = min(pt1i.y, obstacles.rows);
                pt2i.x = min(pt2i.x, obstacles.cols);
                pt2i.y = min(pt2i.y, obstacles.rows);

                cv::Rect roi = cv::Rect(pt1i, pt2i);
                cv::Mat curObstacle = obstacles(roi);
                int obstacleCnt = cv::countNonZero(curObstacle);

                if(obstacleCnt > curObstacle.rows * curObstacle.cols / 2){
                    for(int oIdx = 0; oIdx < orientSectors; ++oIdx) {
                        int val = Utils::mapGridToVal(mapXIdx, mapYIdx, oIdx);
                        forbiddenVals.insert(val);
                    }
                }
                else{
                    for(int oIdx = 0; oIdx < orientSectors; ++oIdx) {
                        int val = Utils::mapGridToVal(mapXIdx, mapYIdx, oIdx);
                        allowedVals.insert(val);
                    }
                }
            }
        }
        cout << "forbiddenVals.size() = " << forbiddenVals.size() << endl;
        cout << "allowedVals.size() = " << allowedVals.size() << endl;
    }

    graph = shared_ptr<Graph>(new Graph(allowedVals));

    cout << "Map read" << endl;
}

void readTrajectory(const boost::filesystem::path &dirPath,
                    const vector<LocationWifi> &mapWifiLocations,
                    FastABLE &fastable,
                    std::vector<LocationGeneral> &locations,
                    vector<vector<vector<double>>> &probs,
                    std::vector<double> &stepDists,
                    std::vector<double> &orientMeas)
{
    cout << "Reading trajectory" << endl;

    {
        map<int, int> locIdToIdx;

        {
            ifstream wifiFile((dirPath / "wifi.map").c_str());
            if (!wifiFile.is_open()) {
                cout << "Error! Could not open " << (dirPath / "wifi.map").c_str() << " file" << endl;
            }
            while (!wifiFile.eof() && !wifiFile.fail()) {
                LocationGeneral curLoc;

                uint64_t timestamp = 0;
                int locId = 0;
                int nscans = 0;
                wifiFile >> timestamp >> locId >> nscans;
                if (!wifiFile.fail()) {
                    curLoc.timestamp = timestamp;
                    curLoc.locationXY.id = locId;
                    for (int i = 0; i < nscans; ++i) {
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
                        if (!wifiFile.fail()) {
                            curLoc.wifiScans.emplace_back(bssid, ssid, level, freq, localTimestamp);
                        }
                    }

                    vector<vector<double>> curProb = locationWifiProb(LocationWifi(curLoc.timestamp,
                                                                                   curLoc.locationXY,
                                                                                   curLoc.wifiScans),
                                                                      mapWifiLocations);

                    double maxVal = 0.0;
                    for (int mapYIdx = 0; mapYIdx < curProb.size(); ++mapYIdx) {
                        for (int mapXIdx = 0; mapXIdx < curProb[mapYIdx].size(); ++mapXIdx) {
                            double val = curProb[mapYIdx][mapXIdx];

                            maxVal = max(val, maxVal);
                        }
                    }

                    if(maxVal > minProb) {

                        locIdToIdx[locId] = locations.size();
                        locations.push_back(curLoc);
                        probs.push_back(curProb);
                    }
                }
            }
            cout << "read " << locations.size() << " WiFi locations" << endl;
        }

        vector<ImageRecognitionResult> imgRecogRes(locations.size());
        {
            uint64_t prevTimestamp = 0;
            int cnt = 0;

            ifstream imageFile((dirPath / "imgs/images.map").c_str());
            if (!imageFile.is_open()) {
                cout << "Error! Could not open " << (dirPath / "imgs/images.map").c_str() << " file" << endl;
            }
            while (!imageFile.eof() && !imageFile.fail()) {
                LocationGeneral curLoc;

                uint64_t timestamp = 0;
                int locId = 0;
                string imageFilename;
                int segId = 0;
                imageFile >> timestamp >> locId >> imageFilename >> segId;
                if (!imageFile.fail()) {
                    curLoc.timestamp = timestamp;
                    curLoc.segmentId = segId;
                    curLoc.locationXY.id = locId;

                    curLoc.image = cv::imread((dirPath / "imgs" / imageFilename).string());

                    // FastABLE recognition
                    ImageRecognitionResult curRes = fastable.addNewTestingImage(curLoc.image);

                    if(curRes.matchingLocations.size() > 0 && (timestamp - prevTimestamp) / 1.0e9 > 1.0) {
                        vector<vector<double>> curProb = locationImageProb(curLoc,
                                                                          curRes);

                        locIdToIdx[locId] = locations.size();
                        locations.push_back(curLoc);
                        probs.push_back(curProb);

                        imgRecogRes.push_back(curRes);

                        ++cnt;
                        prevTimestamp = timestamp;
                    }
                }
            }
            cout << "read " << cnt << " image locations" << endl;
        }

        ifstream positionsFile((dirPath / "positions.map").c_str());
        if (!positionsFile.is_open()) {
            cout << "Error! Could not open " << (dirPath / "positions.map").c_str() << " file" << endl;
        }
        while (!positionsFile.eof() && !positionsFile.fail()) {
            int id;
            double x, y;
            positionsFile >> id >> x >> y;
            if (!positionsFile.fail()) {
                if (locIdToIdx.count(id) > 0) {
                    int idx = locIdToIdx[id];
                    locations[idx].locationXY = LocationXY(x, y, id);

                    if(locations[idx].wifiScans.empty()){
                        cout << endl << "id = " << id << endl;
                        for(int i = 0; i < imgRecogRes[idx].matchingLocations.size(); ++i){
                            double dx = locations[idx].locationXY.x - imgRecogRes[idx].matchingLocations[i].x;
                            double dy = locations[idx].locationXY.y - imgRecogRes[idx].matchingLocations[i].y;

                            cout << "w = " << imgRecogRes[idx].matchingWeights[i] << ", dist = " << sqrt(dx*dx + dy*dy) << endl;
                        }
                    }
                }
            }
        }

    }

    // sort according to timestamp
    {
        vector<pair<uint64_t, int>> tsIdx;
        for(int l = 0; l < locations.size(); ++l){
            tsIdx.emplace_back(locations[l].timestamp, l);
        }
        sort(tsIdx.begin(), tsIdx.end());

        std::vector<LocationGeneral> newLocations;
        vector<vector<vector<double>>> newProbs;
        for(const auto &loc : tsIdx){
            newLocations.push_back(locations[loc.second]);
            newProbs.push_back(probs[loc.second]);
        }
        locations = std::move(newLocations);
        probs = std::move(newProbs);
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
        for(int s = 0; s < locations.size(); ++s){
            uint64_t curScanTs = locations[s].timestamp;
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
//                double heading = -atan2(rotationMatrix(1, 0), rotationMatrix(0, 0));
                double heading = atan2(-rotationMatrix(1,2), rotationMatrix(0,2));

                orientSamp.push_back(heading);
                orientSampTs.push_back(timestamp);
            }
        }
    }

    {
        int curOrientSampIdx = 0;
        double prevOrient = 0;
        int prevOrientSampIdx = 0;
        for(int s = 0; s < locations.size(); ++s) {
            uint64_t curScanTs = locations[s].timestamp;

            while (curOrientSampIdx < orientSamp.size() && orientSampTs[curOrientSampIdx] < curScanTs) {
                ++curOrientSampIdx;
            }

            double curOrient = Utils::meanOrient(orientSamp.begin() + prevOrientSampIdx,
                                                 orientSamp.begin() + curOrientSampIdx);
//            orientMeas.push_back(Utils::toPiRange(curOrient - prevOrient));
            // first is ignored, because orientation regards direction of the movement after a scan
            if(s > 0) {
                orientMeas.push_back(curOrient);
            }

            prevOrient = curOrient;
            prevOrientSampIdx = curOrientSampIdx;
        }
        // adding orientation for the last scan just for sake of completeness
        orientMeas.push_back(prevOrient);

        vector<double> orientOffsets;
        for(int s = 0; s < orientMeas.size(); ++s) {
            double locX = locations[s].locationXY.x;
            double locY = locations[s].locationXY.y;
            double orientMap = 0.0;
            if(s < orientMeas.size() - 1){
                double nextLocX = locations[s + 1].locationXY.x;
                double nextLocY = locations[s + 1].locationXY.y;
                double dx = nextLocX - locX;
                double dy = nextLocY - locY;
                orientMap = atan2(dy, dx);
            }

            double curOrientOffset = Utils::angDiff(orientMeas[s], orientMap);
            orientOffsets.push_back(curOrientOffset);
        }

        double orientOffset = Utils::meanOrient(orientOffsets.begin(), orientOffsets.end());

        // remove offset
        for(int s = 0; s < orientMeas.size(); ++s){
            orientMeas[s] = Utils::angDiff(orientMeas[s], orientOffset);
        }
    }

    cout << "Trajectory read" << endl;
}

void visualizeMapProb(const std::vector<LocationWifi> &database,
                      const std::vector<std::vector<double>> &prob,
                      const double &orient,
                      const double &varVal,
                      const cv::Mat &mapImage,
                      const cv::Mat &mapObstacle,
                      const double &mapScale)
{
    cv::Mat mapVis = mapImage.clone();

    cv::Mat probVal(mapImage.rows, mapImage.cols, CV_32FC1, cv::Scalar(0));
    cv::Mat probVis(mapImage.rows, mapImage.cols, CV_8UC3, cv::Scalar(0));

    double maxProb = 0.0;
    for (int mapYIdx = 0; mapYIdx < prob.size(); ++mapYIdx) {
        for (int mapXIdx = 0; mapXIdx < prob[mapYIdx].size(); ++mapXIdx) {
            double val = prob[mapYIdx][mapXIdx];
//            cout << "val = " << val << endl;
            maxProb = max(val, maxProb);
            
            LocationXY mapCoord = Utils::mapGridToCoord(mapXIdx, mapYIdx);

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
    
//    {
//        probVal.setTo(cv::Scalar(0.25, 0.25, 0.25));
//        probVis.setTo(cv::Scalar(255, 255, 255));
//        probVis.setTo(cv::Scalar(0, 255, 0), mapObstacle == 0);
//    }
    
    cv::Mat probVisBlended(probVis.clone());
    cv::Mat mapVisBlended(mapVis.clone());
    for(int r = 0; r < probVisBlended.rows; ++r){
        for(int c = 0; c < probVisBlended.cols; ++c){
            probVisBlended.at<cv::Vec3b>(r, c) *= probVal.at<float>(r, c);
            mapVisBlended.at<cv::Vec3b>(r, c) *= 1.0 - probVal.at<float>(r, c);
        }
    }
    cv::Mat vis = mapVisBlended + probVisBlended;

    // draw ground truth positions and orientation measurements
    {
        int mapXIdx, mapYIdx, oIdx;
        Utils::valToMapGrid(varVal, mapXIdx, mapYIdx, oIdx);
        LocationXY loc = Utils::mapGridToCoord(mapXIdx, mapYIdx);
        double orientVal = Utils::orientIdxToOrient(oIdx);
        cv::Point2d pt(loc.x, loc.y);
        static constexpr double radius = 2.0;
//        cv::Point2d ptDir(loc.x + radius * cos(orientVal), loc.y + radius * sin(orientVal));
        cv::Point2d ptDir(loc.x + radius * cos(orient), loc.y + radius * sin(orient));

        static const cv::Scalar color(0.1880 * 255, 0.6740 * 255, 0.4660 * 255);
        cv::circle(vis, pt * mapScale, radius * mapScale, color, 4);
        cv::line(vis, pt * mapScale, ptDir * mapScale, color, 4);
    }
//    for(const LocationWifi &curLoc : database){
//        cv::Point2d pt(curLoc.locationXY.x, curLoc.locationXY.y);
//        cv::circle(vis, pt * mapScale, 8, cv::Scalar(0.1840 * 255, 0.0780 * 255, 0.6350 * 255), CV_FILLED);
//    }

    cv::Mat visScaled;
    cv::resize(vis, visScaled, cv::Size(0, 0), 0.5, 0.5);

    cv::imshow("map", visScaled);
    
    cv::waitKey(100);
//    if((cv::waitKey() & 0xff) == 's'){
//        cv::imwrite("../log/prob.png", vis);
//    }
}

void visualizeMapInfer(const std::vector<LocationWifi> &database,
                       const std::vector<LocationGeneral> &trajLocations,
                       const std::vector<LocationXY> &inferLocations,
                       const std::vector<LocationXY> &compLocations,
                       const cv::Mat &mapImage,
                       const double &mapScale,
                       bool stop = false,
                       bool save = false)
{
    cv::Mat mapVis = mapImage.clone();
//    for(const LocationWifi &curLoc : database){
//        cv::Point2d pt(curLoc.locationXY.x, curLoc.locationXY.y);
//        cv::circle(mapVis, pt * mapScale, 5, cv::Scalar(0, 0, 255), CV_FILLED);
//    }

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
    
    static constexpr double radius = 8;
    static constexpr double thickness = 4;

    for(int i = 0; i < inferLocations.size(); ++i){
        static const cv::Scalar color(0.7410 * 255, 0.4470 * 255, 0.0 * 255);

        const LocationXY &infLoc = inferLocations[i];
        cv::Point2d pt(infLoc.x, infLoc.y);
        if(trajLocations[i].wifiScans.empty()){
            cv::circle(mapVis, pt * mapScale, radius, color, 2);
        }
        else {
            cv::circle(mapVis, pt * mapScale, radius, color, CV_FILLED);
        }

//            cv::putText(mapVis,
//                        to_string(i),
//                        pt * mapScale,
//                        cv::FONT_HERSHEY_PLAIN,
//                        1,
//                        cv::Scalar(0, 0, 0));
        if (i > 0) {
            const LocationXY &prevInfLoc = inferLocations[i - 1];
            cv::Point2d prevPt(prevInfLoc.x, prevInfLoc.y);

            cv::line(mapVis, prevPt * mapScale, pt * mapScale, color, thickness);
        }
    }
    for(int i = 0; i < compLocations.size(); ++i){
        static const cv::Scalar color(0.0980 * 255, 0.3250 * 255, 0.8500 * 255);

        const LocationXY &compLoc = compLocations[i];
        cv::Point2d pt(compLoc.x, compLoc.y);
        cv::circle(mapVis, pt * mapScale, radius, color, CV_FILLED);

//            cv::putText(mapVis,
//                        to_string(i),
//                        pt * mapScale,
//                        cv::FONT_HERSHEY_PLAIN,
//                        1,
//                        cv::Scalar(0, 0, 0));
        if (i > 0) {
            const LocationXY &prevCompLoc = compLocations[i - 1];
            cv::Point2d prevPt(prevCompLoc.x, prevCompLoc.y);

            cv::line(mapVis, prevPt * mapScale, pt * mapScale, color, thickness);
        }
    }

    for(int i = 0; i < trajLocations.size(); ++i){
        static const cv::Scalar color(0.1250 * 255, 0.6940 * 255, 0.9290 * 255);

        const LocationXY &gtLoc = trajLocations[i].locationXY;
        cv::Point2d pt(gtLoc.x, gtLoc.y);
        if(trajLocations[i].wifiScans.empty()){
            cv::circle(mapVis, pt * mapScale, radius, color, 2);
        }
        else {
            cv::circle(mapVis, pt * mapScale, radius, color, CV_FILLED);
        }

        if (i > 0) {
            const LocationXY &prevGtLoc = trajLocations[i - 1].locationXY;
            cv::Point2d prevGtPt(prevGtLoc.x, prevGtLoc.y);

            cv::line(mapVis, prevGtPt * mapScale, pt * mapScale, color, thickness);
        }
    }

    if(save) {
        static int cnt = 0;
        static bool online = true;

        char nameBuff[100];
        sprintf(nameBuff, "../log/res_%02d_%s.png", cnt, online ? "online" : "offline");
        cv::imwrite(nameBuff, mapVis);

        if(online){
            online = false;
        }
        else{
            ++cnt;
            online = true;
        }
    }
    
    cv::resize(mapVis, mapVis, cv::Size(0, 0), 0.5, 0.5);
    
    cv::imshow("map", mapVis);
    
    if(stop){
        cv::waitKey();
//        cv::imwrite("../log/res.png", mapVis);
    }
    else {
        cv::waitKey(100);
    }
}

Pgm buildPgm(const std::vector<LocationGeneral> &wifiLocations,
             const cv::Mat &obstacles,
             const double &mapScale,
             const std::vector<std::vector<std::vector<double>>> &probs,
             const std::vector<double> &stepDists,
             const std::vector<double> &orientMap,
             const set<int> &forbiddenVals,
             const shared_ptr<Graph> &graph,
             std::vector<double> &obsVec,
             std::map<int, int> &locIdxToVarClusterId,
             std::vector<double> &varVals)
{
    std::vector<std::shared_ptr<RandVar>> randVars;
    std::vector<std::shared_ptr<Feature>> feats;
    std::vector<std::shared_ptr<Cluster>> clusters;

    obsVec.clear();
    varVals.clear();
    
    // observation vector
    // location coordinates
    vector<double> xCoords, yCoords, oCoords;
//    vector<double> rvVals;
    int mapSize = 0;
    for (int mapYIdx = 0; mapYIdx < probs[0].size(); ++mapYIdx) {
        for (int mapXIdx = 0; mapXIdx < probs[0][mapYIdx].size(); ++mapXIdx) {
            LocationXY mapCoord = Utils::mapGridToCoord(mapXIdx, mapYIdx);
            for(int oIdx = 0; oIdx < orientSectors; ++oIdx) {
                double curOrient = Utils::orientIdxToOrient(oIdx);

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
        double curOrient = Utils::orientIdxToOrient(oIdx);
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
            double curOrient = atan2(dy, dx);
            locOIdx = Utils::orientToOrientIdx(curOrient);

//            {
//                double distMeas = stepDists[i + 1];
//                double curDist = sqrt(dx*dx + dy*dy);
////                double curOrient = Utils::orientIdxToOrient(locOIdx);
//                double orientMeas = orientMap[i + 1];
//
//                double error = curDist - distMeas;
//                double orientError = Utils::angDiff(curOrient, orientMeas);
//
////                double nextLocXPred = locX + curDist * cos(curOrient);
////                double nextLocYPred = locY + curDist * sin(curOrient);
////
////                double error = sqrt((nextLocX - nextLocXPred)*(nextLocX - nextLocXPred) +
////                        (nextLocY - nextLocYPred)*(nextLocY - nextLocYPred));
////                double orientError = Utils::angDiff(curOrient, orientMap[i]);
//
//                cout << "step = " << i << ", error = " << error << ", orientError  = " << orientError << endl;
////                cout << "loc = (" << locX << ", " << locY << "), nextLoc = (" << nextLocX << ", " << nextLocY << ")" <<
////                        ", d = (" << curDist * cos(curOrient) << ", " << curDist * sin(curOrient) << ")" << endl;
////                cout << "step = " << i << ", dist = " << curDist << ", error = " << error << endl;
//            }

            prevLocOIdx = locOIdx;
        }

        pair<int, int> mapIdx = Utils::mapCoordToGrid(locX, locY);
        double varVal = Utils::mapGridToVal(mapIdx.first, mapIdx.second, locOIdx);
        double closestDist = std::numeric_limits<double>::max();
        int closestXIdx = mapIdx.first;
        int closestYIdx = mapIdx.second;
        
        double curProbThresh = min(probThresh, maxProb / probRatio);
        vector<double> rvVals;
        for (int mapYIdx = 0; mapYIdx < probs[i].size(); ++mapYIdx) {
            for (int mapXIdx = 0; mapXIdx < probs[i][mapYIdx].size(); ++mapXIdx) {
                // check any orientation
                int valCheck = Utils::mapGridToVal(mapXIdx, mapYIdx, 0);

                bool validLoc = (probs[i][mapYIdx][mapXIdx] >= curProbThresh) && (forbiddenVals.count(valCheck) == 0);
                for(int oIdx = 0; oIdx < orientSectors; ++oIdx) {
                    if (validLoc){
                        int val = Utils::mapGridToVal(mapXIdx, mapYIdx, oIdx);
                        rvVals.push_back(val);
                    }
                }
                if (validLoc){
                    LocationXY loc = Utils::mapGridToCoord(mapXIdx, mapYIdx);
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

            if(closestDist > 3.0){
                throw "Closest varVal too far";
            }
            
            varVal = Utils::mapGridToVal(closestXIdx, closestYIdx, locOIdx);
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

        int curParamId = nextParamId;
        // separate parameter for image locations.
        if(wifiLocations[i].wifiScans.empty()){
           curParamId += 1;
        }
        shared_ptr<Feature> curFeat(new LocFeature(nextFeatId++,
                                                   curParamId,
                                                   vector<shared_ptr<RandVar>>{randVars[i]},
                                                   curObsVecIdxs));

//        {
//            vector<double> curObsVec;
//            for(int o = 0; o < curObsVecIdxs.size(); ++o){
//                curObsVec.push_back(obsVec[curObsVecIdxs[o]]);
//            }
//            cout << "step " << i << ", locFeat val = " << curFeat->comp(vector<double>{varVals[i - 1], varVals[i]},
//                                                                               curObsVec) << endl;
//        }
        
        locFeats.push_back(curFeat);
        feats.push_back(curFeat);
    }
    // for wifi and image locations
    nextParamId += 2;

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
                                                    distSigma,
                                                    graph));

//        {
//            vector<double> curObsVec;
//            for(int o = 0; o < curObsVecIdxs.size(); ++o){
//                curObsVec.push_back(obsVec[curObsVecIdxs[o]]);
//            }
//            cout << "step " << i << ", moveFeat val = " << curFeat->comp(vector<double>{varVals[i - 1], varVals[i]},
//                                                                         curObsVec) << endl;
//        }

        moveFeats.push_back(curFeat);
        feats.push_back(curFeat);
    }
    ++nextParamId;

    // orient move features
    vector<shared_ptr<Feature>> orientMoveFeats;
    // skip first one, as there is no distance from previous location
    for(int i = 1; i < probs.size(); ++i){
        vector<int> curObsVecIdxs(1 + 3 * mapSize);
        curObsVecIdxs[0] = obsVecStartOrientMeas + i;
        // x coordinates
        iota(curObsVecIdxs.begin() + 1, curObsVecIdxs.begin() + 1 + mapSize, obsVecStartLoc);
        // y coordinates
        iota(curObsVecIdxs.begin() + 1 + mapSize, curObsVecIdxs.begin() + 1 + 2 * mapSize, obsVecStartLoc + mapSize);
        // orientation
        iota(curObsVecIdxs.begin() + 1 + 2 * mapSize, curObsVecIdxs.begin() + 1 + 3 * mapSize, obsVecStartLoc + 2 * mapSize);

        shared_ptr<Feature> curFeat(new OrientMoveFeature(nextFeatId++,
                                                    nextParamId,
                                                    vector<shared_ptr<RandVar>>{randVars[i-1], randVars[i]},
                                                    curObsVecIdxs,
                                                    mapSize,
                                                    orientSigma));

//        {
//            vector<double> curObsVec;
//            for(int o = 0; o < curObsVecIdxs.size(); ++o){
//                curObsVec.push_back(obsVec[curObsVecIdxs[o]]);
//            }
//            cout << "step " << i << ", orientMoveFeat val = " << curFeat->comp(vector<double>{varVals[i - 1], varVals[i]},
//                                                                         curObsVec) << endl;
//        }

        orientMoveFeats.push_back(curFeat);
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
                                                   vector<shared_ptr<Feature>>{moveFeats[i - 1], orientMoveFeats[i - 1]},
                                                   vector<shared_ptr<RandVar>>{randVars[i - 1], randVars[i]}));
    
        moveFeatClusters.push_back(curCluster);
        clusters.push_back(curCluster);
    }


    // create pgm
    Pgm pgm(randVars, clusters, feats);
    
    // edges from random variable clusters to location feature clusters
    for(int i = 0; i < probs.size(); ++i){
        Pgm::addEdgeToPgm(rvClusters[i], locFeatClusters[i], vector<shared_ptr<RandVar>>{randVars[i]});
    }

   for(int i = 1; i < probs.size(); ++i){
       Pgm::addEdgeToPgm(rvClusters[i - 1], moveFeatClusters[i - 1], vector<shared_ptr<RandVar>>{randVars[i - 1]});
       Pgm::addEdgeToPgm(rvClusters[i], moveFeatClusters[i - 1], vector<shared_ptr<RandVar>>{randVars[i]});
    }
    
    // MoG, orient
//    pgm.params() = vector<double>{3.41153, 3.41153, 2.49436, 1.1856};
    pgm.params() = vector<double>{3.21782, 1.26879, 2.9435, 1.46455};

    // MoG, no orient
//    pgm.params() = vector<double>{4.16378, 4.16378, 2.51943, 0.0110781};

    // wknn, orient
//    pgm.params() = vector<double>{3.05433, 3.05433, 2.50476, 1.14605};

    // wknn, no orient
//    pgm.params() = vector<double>{3.59296, 3.59296, 2.54812, 0.00494331};

    return pgm;
}

vector<LocationXY> inferLocations(const Pgm &pgm,
                                  const vector<double> &obsVec,
                                  const std::map<int, int> &locIdxToVarClusterId)
{
    vector<LocationXY> retLoc;
    
    vector<vector<double>> marg;
    vector<vector<vector<double>>> msgs;
//    vector<double> params{1.0, 1.0};
//    vector<double> params = pgm.params();
    
    bool calibrated = Inference::compMAPParam(pgm,
                                            marg,
                                            msgs,
                                              pgm.params(),
                                            obsVec);
    
//    cout << "calibrated = " << calibrated << endl;
    
    vector<vector<double>> retVals = Inference::decodeMAP(pgm,
                                                         marg,
                                                         msgs,
                                                          pgm.params(),
                                                         obsVec);
    
    for(const auto& idxId : locIdxToVarClusterId){
        int varClusterId = idxId.second;

        const vector<double> &curVals = retVals[varClusterId];
        
        int curLoc = curVals.front();
        
        int mapXIdx, mapYIdx, oIdx;
        Utils::valToMapGrid(curLoc, mapXIdx, mapYIdx, oIdx);
        
        LocationXY curLocXY = Utils::mapGridToCoord(mapXIdx, mapYIdx);
        
        retLoc.push_back(curLocXY);
    }
    
    return retLoc;
}

int main() {
    static constexpr bool stopVis = false;
    static constexpr bool saveVis = false;

//    static constexpr bool estimateParams = true;
//    static constexpr bool infer = false;
    static constexpr bool estimateParams = false;
    static constexpr bool infer = true;

    static constexpr int seqLen = 5;

    boost::filesystem::path mapDirPath("/mnt/data/datasets/JW/indoor_localization/IndoorGraphLocalization/dataset/2019_04_02_PUTMC_Floor3_Experia_map");

    cv::Mat mapImage;
    cv::Mat mapObstacles;
    double mapScale;
    set<int> forbiddenVals;
    set<int> allowedVals;
    shared_ptr<Graph> graph;
    vector<LocationWifi> mapWifiLocations;
    vector<LocationImage> mapImageLocations;
    readMap(mapDirPath,
            mapWifiLocations,
            mapImageLocations,
            mapImage,
            mapObstacles,
            mapScale,
            forbiddenVals,
            allowedVals,
            graph);

    boost::filesystem::path trajRoot("/mnt/data/datasets/JW/indoor_localization/IndoorGraphLocalization/dataset/2019_04_02_PUTMC_Floor3_Experia_trajs");
    vector<boost::filesystem::path> trajDirPaths;
    if(estimateParams) {
        trajDirPaths = {"kc_1",
                        "mn_1",
                        "mn_4",
                        "mn_sick_2"};
    }
    else {
        trajDirPaths = {"kc_2",
                        "kc_3",
                        "kc_4",
                        "mn_2",
                        "mn_3",
                        "mn_5",
                        "mn_sick_1",
                        "mn_sick_3",
                        "ps_1"};
    }

    setFastABLE setFastable{64, 40, 0.8, 7, 3, 5, 0.5};

    FastABLE fastable(setFastable);
    fastable.addImageMap(mapImageLocations);

    vector<Pgm> pgms;
    vector<vector<double>> obsVecs;
    vector<vector<double>> varVals;


    ofstream errorsFile("../log/errors");
    ofstream errorsAllFile("../log/errors_all");

    chrono::nanoseconds infDur(0);
    int infTimeCnt = 0;

    for(int t = 0; t < trajDirPaths.size(); ++t) {
        cout << "Trajectory: " << trajDirPaths[t].string() << endl;

        fastable.resetTestingImages();

        vector<LocationGeneral> curTrajLocations;
        vector<vector<vector<double>>> curProbs;
        vector<double> curStepDists;
        vector<double> curOrients;
        readTrajectory(trajRoot / trajDirPaths[t],
                       mapWifiLocations,
                       fastable,
                       curTrajLocations,
                       curProbs,
                       curStepDists,
                       curOrients);

//        for (int i = 0; i < curTrajLocations.size(); ++i) {
//            vector<vector<double>> curProb = locationWifiProb(curTrajLocations[i], mapWifiLocations, useWknn);
//
//            curProbs.push_back(curProb);
//        }
//
//        removeNotMatchedLocations(curTrajLocations,
//                                  curStepDists,
//                                  curOrients,
//                                  curProbs);

        vector<double> curObsVec;
        map<int, int> curLocIdxToRandVarClusterId;
        vector<double> curVarVals;
        Pgm curPgm = buildPgm(curTrajLocations,
                           mapObstacles,
                           mapScale,
                           curProbs,
                           curStepDists,
                           curOrients,
                           forbiddenVals,
                           graph,
                           curObsVec,
                           curLocIdxToRandVarClusterId,
                           curVarVals);

        for (int i = 0; i < curTrajLocations.size(); ++i) {
            visualizeMapProb(mapWifiLocations,
                             curProbs[i],
                             curOrients[i],
                             curVarVals[i],
                             mapImage,
                             mapObstacles,
                             mapScale);
        }

        {

            {
                vector<LocationXY> infLocAll;
                vector<LocationXY> wknnLocAll;

                vector<double> errors;
                double errorSum = 0;
                int errorCnt = 0;

                vector<double> errorsComp;
                double errorSumComp = 0;
                int errorCntComp = 0;

                int curIdxStart = 0;
                int curIdxEnd = 0;
                int scanCnt = 0;
                int firstIdxEnd = 0;
                while (curIdxEnd < curTrajLocations.size()) {
                    bool stopVisSeq = false;

                    // find next scan
                    while (curIdxStart < curTrajLocations.size() && scanCnt >= seqLen) {
                        if(!curTrajLocations[curIdxStart].wifiScans.empty()){
                            --scanCnt;
                        }
                        ++curIdxStart;
                    }

                    int newLocCnt = 0;
                    // iterate until seqLen scans found
                    while (curIdxEnd < curTrajLocations.size() && scanCnt < seqLen) {
                        if (!curTrajLocations[curIdxEnd].wifiScans.empty()) {
                            ++scanCnt;
                        }
                        ++curIdxEnd;
                        ++newLocCnt;
                    }

                    // if first sequence
                    if(curIdxStart == 0){
                        firstIdxEnd = curIdxEnd;
                    }

                    if(scanCnt == seqLen) {
                        vector<LocationGeneral> seqTrajLocationsGeneral = vector<LocationGeneral>(
                                curTrajLocations.begin() + curIdxStart,
                                curTrajLocations.begin() + curIdxEnd);
                        vector<LocationWifi> seqTrajLocationsWifi;
                        for(const auto &loc : seqTrajLocationsGeneral){
                            if(!loc.wifiScans.empty()){
                                seqTrajLocationsWifi.push_back(LocationWifi(loc));
                            }
                        }

                        chrono::steady_clock::time_point startTime = chrono::steady_clock::now();
                        vector<double> iObsVec;
                        map<int, int> iLocIdxToRandVarClusterId;
                        vector<double> iVarVals;
                        Pgm iPgm = buildPgm(seqTrajLocationsGeneral,
                                            mapObstacles,
                                            mapScale,
                                            vector<vector<vector<double>>>(
                                                    curProbs.begin() + curIdxStart,
                                                    curProbs.begin() + curIdxEnd),
                                            vector<double>(curStepDists.begin() + curIdxStart,
                                                           curStepDists.begin() + curIdxEnd),
                                            vector<double>(curOrients.begin() + curIdxStart,
                                                           curOrients.begin() + curIdxEnd),
                                            forbiddenVals,
                                            graph,
                                            iObsVec,
                                            iLocIdxToRandVarClusterId,
                                            iVarVals);

                        if (infer) {
                            vector<LocationXY> infLoc = inferLocations(iPgm,
                                                                       iObsVec,
                                                                       iLocIdxToRandVarClusterId);

                            chrono::steady_clock::time_point endTime = chrono::steady_clock::now();
                            infDur += endTime - startTime;
                            ++infTimeCnt;

                            // in first iteration add only last location - first wifi scan localized
                            if(curIdxStart == 0){
                                infLocAll.push_back(infLoc.back());
                            }
                            else {
                                infLocAll.insert(infLocAll.end(), infLoc.end() - newLocCnt, infLoc.end());
                            }
                            {
                                double dx = infLoc.back().x - seqTrajLocationsGeneral.back().locationXY.x;
                                double dy = infLoc.back().y - seqTrajLocationsGeneral.back().locationXY.y;
                                double curError = sqrt(dx * dx + dy * dy);
                                errorSum += curError;
                                errors.push_back(curError);
                                ++errorCnt;

//                            if(curError > 3.0){
//                                stopVisSeq = true;
//                            }
                                cout << "curError = " << curError << endl;
                            }

                            vector<LocationXY> wknnLoc;
                            for (int t =  0; t < seqTrajLocationsWifi.size(); ++t) {
                                double meanErrorWknn = 0.0;

                                LocationXY curLoc = wknn(mapWifiLocations,
                                                         seqTrajLocationsWifi[t],
                                                         wknnk,
                                                         meanErrorWknn);
                                //                        cout << "curLoc = (" << curLoc.x << ", " << curLoc.y << ")" << endl;
                                wknnLoc.push_back(curLoc);
                            }

                            wknnLocAll.push_back(wknnLoc.back());
                            {
                                double dx = wknnLoc.back().x - seqTrajLocationsWifi.back().locationXY.x;
                                double dy = wknnLoc.back().y - seqTrajLocationsWifi.back().locationXY.y;
                                double curErrorComp = sqrt(dx * dx + dy * dy);
                                errorSumComp += curErrorComp;
                                errorsComp.push_back(curErrorComp);
                                ++errorCntComp;

                                cout << "curErrorComp = " << curErrorComp << endl;
                            }

                            visualizeMapInfer(mapWifiLocations,
                                              seqTrajLocationsGeneral,
                                              infLoc,
                                              wknnLoc,
                                              mapImage,
                                              mapScale,
                                              stopVisSeq);
                        }

                        pgms.push_back(iPgm);
                        varVals.push_back(iVarVals);
                        obsVecs.push_back(iObsVec);
                    }
                }
                if (infer && !infLocAll.empty()) {
                    cout << endl;
                    if (errorCnt > 0) {
                        cout << "mean error = " << errorSum / errorCnt << endl;
                    }
                    if (errorCntComp > 0) {
                        cout << "mean comp error = " << errorSumComp / errorCntComp << endl;
                    }
                    cout << endl;

                    visualizeMapInfer(mapWifiLocations,
                                      vector<LocationGeneral>(
                                              curTrajLocations.begin() + firstIdxEnd - 1,
                                              curTrajLocations.end()),
                                      infLocAll,
                                      wknnLocAll,
                                      mapImage,
                                      mapScale,
                                      stopVis,
                                      saveVis);

                    for (int e = 0; e < errors.size(); ++e) {
                        errorsFile << errors[e] << " " << errorsComp[e] << endl;
                    }
                }
            }
            if(infer){
                vector<double> errors;
                double errorSum = 0;
                int errorCnt = 0;

                vector<double> errorsComp;
                double errorSumComp = 0;
                int errorCntComp = 0;


                vector<LocationXY> infLoc = inferLocations(curPgm,
                                                           curObsVec,
                                                           curLocIdxToRandVarClusterId);
                for (int t = 0; t < curTrajLocations.size(); ++t) {
                    if(!curTrajLocations[t].wifiScans.empty()) {
                        double dx = infLoc[t].x - curTrajLocations[t].locationXY.x;
                        double dy = infLoc[t].y - curTrajLocations[t].locationXY.y;
                        double curError = sqrt(dx * dx + dy * dy);
                        errorSum += curError;
                        errors.push_back(curError);
                        ++errorCnt;

//                        cout << "curError = " << curError << endl;
                    }
                }

                vector<LocationXY> wknnLoc;
                for (int t = 0; t < curTrajLocations.size(); ++t) {
                    if(!curTrajLocations[t].wifiScans.empty()) {
                        double meanErrorWknn = 0.0;
                        LocationXY curLoc = wknn(mapWifiLocations,
                                                 LocationWifi(curTrajLocations[t]),
                                                 wknnk,
                                                 meanErrorWknn);
//                        cout << "curLoc = (" << curLoc.x << ", " << curLoc.y << ")" << endl;
                        wknnLoc.push_back(curLoc);

                        double dx = curLoc.x - curTrajLocations[t].locationXY.x;
                        double dy = curLoc.y - curTrajLocations[t].locationXY.y;
                        double curErrorComp = sqrt(dx * dx + dy * dy);
                        errorSumComp += curErrorComp;
                        errorsComp.push_back(curErrorComp);
                        ++errorCntComp;
                    }
                }

                cout << endl;
                if (errorCnt > 0) {
                    cout << "mean error = " << errorSum / errorCnt << endl;
                }
                if (errorCntComp > 0) {
                    cout << "mean comp error = " << errorSumComp / errorCntComp << endl;
                }
                cout << endl;

                visualizeMapInfer(mapWifiLocations,
                                  curTrajLocations,
                                  infLoc,
                                  wknnLoc,
                                  mapImage,
                                  mapScale,
                                  stopVis,
                                  saveVis);

                for (int e = 0; e < errors.size(); ++e) {
                    errorsAllFile << errors[e] << " " << errorsComp[e] << endl;
                }
            }
        }

//            pgms.push_back(curPgm);
//            varVals.push_back(curVarVals);
//            obsVecs.push_back(curObsVec);
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