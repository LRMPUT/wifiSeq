//
// Created by jachu on 15.05.18.
//

#include <string>
#include <vector>
#include <complex>
#include <algorithm>

#include "kissfft/kissfft.hh"

#include "Stepometer.hpp"

double Stepometer::computeDist(const std::vector<double> &accMagSamples,
                               double dt,
                               double accSampFreq,
                               double stepLen,
                               double freqMin,
                               double freqMax,
                               double fftMagThresh)
{
    double dist = 0.0;
    
    int winLen = accMagSamples.size();

    // winLen samples for transform_real and forward transform (inverse = false)
    kissfft<double> kissfftInst(winLen/2, false);
    
    // for transform_real output is winLen/2 complex numbers
    std::vector<std::complex<double>> fftOut(winLen / 2);
    kissfftInst.transform_real(accMagSamples.data(), fftOut.data());
    
    // do not consider bin for k = 0;
    int lowerBound = 1;
    // do not consider bin for k >= winLen/2
    int upperBound = winLen / 2 - 1;
    
    int kMin = std::max((int) std::floor(freqMin * winLen / accSampFreq), lowerBound);
    int kMax = std::min((int) std::ceil(freqMax * winLen / accSampFreq), upperBound);
    
    int bestK = kMin;
    double bestScore = 0.0;
    for (int k = kMin; k <= kMax; ++k) {
        // scale using winLen, so the value is independent from winLen
        double mag = std::abs(fftOut[k]) / winLen;
        if (mag > bestScore) {
            bestK = k;
            bestScore = mag;
        }
    }
    
    // if fft magnitude is above specified threshold
    if (bestScore > fftMagThresh) {
        // check for fft results for neighboring frequencies
        double nhScore = 0.0;
        int nhK = 0;
        if(bestK - 1 >= lowerBound){
            double nhMag = std::abs(fftOut[bestK - 1]) / winLen;
            if(nhScore < nhMag){
                nhK = bestK - 1;
                nhScore = nhMag;
            }
        }
        if(bestK + 1 <= upperBound){
            double nhMag = std::abs(fftOut[bestK + 1]) / winLen;
            if(nhScore < nhMag){
                nhK = bestK + 1;
                nhScore = nhMag;
            }
        }
        // use weighted mean to compute frequency - beneficial when the resolution of FFT is poor (when sampling frequency is low)
        double meanK = (bestK * bestScore + nhK * nhScore) / (bestScore + nhScore);
//            double meanK = bestK;
        // compute user's step frequency
        double bestFreq = meanK * accSampFreq / winLen;
        
        dist = bestFreq * stepLen * dt;
    }

    
    return dist;
}
