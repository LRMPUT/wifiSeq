//
// Created by jachu on 15.05.18.
//

#ifndef WIFISEQ_STEPOMETER_HPP
#define WIFISEQ_STEPOMETER_HPP

#include <vector>

class Stepometer {
public:
    static double computeDist(const std::vector<double> &accMagSamples,
                              double dt,
                              double accSampFreq,
                              double stepLen,
                              double freqMin,
                              double freqMax,
                              double fftMagThresh);
};


#endif //WIFISEQ_STEPOMETER_HPP
