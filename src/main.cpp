#include <iostream>
#include <cmath>
#include <fstream>

#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>

#include "LocationWiFi.hpp"

using namespace std;

static constexpr int ssThreshold = -100;
static constexpr double sharedPercentThreshold = 0.6;

// in meters
static constexpr double mapGrid = 1.0;
static constexpr double mapMinX = 0.0;
static constexpr double mapMaxX = 140.0;
static constexpr double mapMinY = 0.0;
static constexpr double mapMaxY = 45.0;

static constexpr double wifiSigma = 1.0/8.0;

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
                    std::vector<double> stepDists)
{
    cout << "Reading trajectory" << endl;
    
    ifstream wifiFile((dirPath / "wifi.raw").c_str());
    if(!wifiFile.is_open()){
        cout << "Error! Could not open " << (dirPath / "wifi.raw").c_str() << " file" << endl;
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

//            cout << "adding " << wifiLocations.size() << endl;
            wifiLocations.push_back(curLoc);
        }
    }
    
    // TODO Add distances from stepometer
    stepDists.resize(wifiLocations.size(), 0);
    
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
    int mapGridSizeX = ceil((mapMaxX - mapMinX) / mapGrid);
    int mapGridSizeY = ceil((mapMaxY - mapMinY) / mapGrid);
    vector<vector<double>> prob(mapGridSizeY, vector<double>(mapGridSizeX, 0.0));
    
    for (const LocationWiFi &databaseLoc : database) {
        pair<double, int> error = errorL2(loc, databaseLoc);
        double sharedPercentA = (double) error.second / loc.wifiScans.size();
        double sharedPercentB = (double) error.second / databaseLoc.wifiScans.size();
        
        if (sharedPercentA > sharedPercentThreshold &&
            sharedPercentB > sharedPercentThreshold)
        {
            cout << "matched scan" << endl;
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
                    expVal += -error.first;
                    
//                    cout << "expVal = " << expVal << endl;
                    prob[mapYIdx][mapXIdx] += exp(expVal);
                }
            }
        }
    }
    return prob;
}

void visualizeMap(const std::vector<LocationWiFi> &database,
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
            double val = prob[mapYIdx][mapXIdx] * probVisScale;
//            cout << "val = " << val << endl;
            maxVal = max(val, maxVal);
            
            LocationXY mapCoord = mapGridToCoord(mapXIdx, mapYIdx);
            
            cv::Point2d pt1(mapCoord.x - mapGrid/2, mapCoord.y - mapGrid/2);
            cv::Point2d pt2(mapCoord.x + mapGrid/2, mapCoord.y + mapGrid/2);
            
            cv::rectangle(probVal, pt1 * mapScale, pt2 * mapScale, cv::Scalar(val), CV_FILLED);
        }
    }
    cout << "maxVal = " << maxVal << endl;
    
    probVal.convertTo(probVal, CV_8U, 256);
    cv::applyColorMap(probVal, probVis, cv::COLORMAP_JET);
    
    cv::Mat vis = mapVis * 0.75 + probVis * 0.25;
    cv::resize(vis, vis, cv::Size(0, 0), 0.5, 0.5);
    
    cv::imshow("map", vis);
    
    cv::waitKey();
}

int main() {
    boost::filesystem::path mapDirPath("../res/Maps/PUTMC_Lenovo_17_05_25");
    
    cv::Mat mapImage;
    double mapScale;
    vector<LocationWiFi> mapLocations = readMap(mapDirPath, mapImage, mapScale);
    
    boost::filesystem::path trajDirPath("../res/Trajectories/PUTMC_Lenovo_17_05_10/PUTMC_Lenovo_013_17_05_10");
    
    vector<LocationWiFi> trajLocations;
    vector<double> stepDists;
    readTrajectory(trajDirPath, trajLocations, stepDists);
    
    for(int i = 0; i < trajLocations.size(); ++i){
        vector<vector<double>> prob = locationProb(trajLocations[i], mapLocations);
        visualizeMap(mapLocations, prob, mapImage, mapScale);
    }
}