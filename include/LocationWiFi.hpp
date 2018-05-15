//
// Created by jachu on 14.05.18.
//

#ifndef WIFISEQ_LOCATIONWIFI_HPP
#define WIFISEQ_LOCATIONWIFI_HPP

#include <string>
#include <vector>

struct LocationXY{
    LocationXY() {}
    
    LocationXY(double ix, double iy) : x(ix), y(iy) {}
    
    double x, y;
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

class LocationWiFi {
public:
    LocationWiFi() {}
    
    uint64_t timestamp;
    LocationXY locationXY;
    std::vector<ScanResult> wifiScans;
private:

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
