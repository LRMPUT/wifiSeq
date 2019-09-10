//
// Created by jachu on 14.05.18.
//

#ifndef WIFISEQ_LOCATIONWIFI_HPP
#define WIFISEQ_LOCATIONWIFI_HPP

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

struct LocationXY{
    LocationXY() {}
    
    LocationXY(double ix, double iy, int iid) : x(ix), y(iy), id(iid) {}
    
    double x, y;
    int id;
};

struct ScanResult{
    ScanResult() {}
    
    ScanResult(const std::string &bssid,
               const std::string &ssid,
               double level,
               int freq,
               int localTimestamp)
            : bssid(bssid), ssid(ssid), level(level), freq(freq), localTimestamp(localTimestamp) {}
    
    std::string bssid, ssid;
    double level;
    int freq;
    uint64_t localTimestamp;
};

class LocationGeneral {
public:
    LocationGeneral() {}

    uint64_t timestamp;
    LocationXY locationXY;

    std::vector<ScanResult> wifiScans;

    int segmentId;
    cv::Mat image;
};

class LocationWiFi {
public:
    LocationWiFi() {}

    explicit LocationWiFi(const LocationGeneral &lg)
            : timestamp(lg.timestamp), locationXY(lg.locationXY), wifiScans(lg.wifiScans) {

    }

    LocationWiFi(uint64_t timestamp, const LocationXY &locationXy, const std::vector<ScanResult> &wifiScans)
            : timestamp(timestamp), locationXY(locationXy), wifiScans(wifiScans) {}

    uint64_t timestamp;
    LocationXY locationXY;
    std::vector<ScanResult> wifiScans;
private:

};

class LocationImage {
public:
    LocationImage() {}

    explicit LocationImage(const LocationGeneral &lg)
            : timestamp(lg.timestamp), locationXY(lg.locationXY), segmentId(lg.segmentId), image(lg.image.clone()) {
    }

    uint64_t timestamp;
    LocationXY locationXY;
    int segmentId;
    cv::Mat image;
};



class CompResWiFi{
public:
    CompResWiFi() {}
    
    LocationWiFi locA, locB;
    double error;
    double sharedPercentA, sharedPercentB;
private:

};


#endif //WIFISEQ_LOCATIONWIFI_HPP
