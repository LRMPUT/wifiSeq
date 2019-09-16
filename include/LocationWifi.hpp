/*
	Copyright (c) 2019,	Mobile Robots Laboratory:
    -Michal Nowicki (michal.nowicki@put.poznan.pl),
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

class LocationWifi {
public:
    LocationWifi() {}

    explicit LocationWifi(const LocationGeneral &lg)
            : timestamp(lg.timestamp), locationXY(lg.locationXY), wifiScans(lg.wifiScans) {

    }

    LocationWifi(uint64_t timestamp, const LocationXY &locationXy, const std::vector<ScanResult> &wifiScans)
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
    
    LocationWifi locA, locB;
    double error;
    double sharedPercentA, sharedPercentB;
private:

};


#endif //WIFISEQ_LOCATIONWIFI_HPP
