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

#ifndef GRAPHLOCALIZATION_FASTABLE_H
#define GRAPHLOCALIZATION_FASTABLE_H

#include <iostream>

#include "LocationWifi.hpp"
#include "ldb/ldb.h"
#include <vector>
#include <algorithm>

struct setFastABLE {
    double patchSize;
    int compareLength;
    double safetyThresholdRatio;
    double earlyAcceptedVicinity;
    double consistencyThreshold;
    double acceptedVicinityThreshold;
    double timeDiffThreshold;
};

struct ImageRecognitionResult {

    int correctRecognitions;
    std::vector<double> matchingWeights;
    std::vector<LocationXY> matchingLocations;
};


class FastABLE {


public:
    /**
     * Initializes the visual place recognition
     */
    FastABLE(setFastABLE settings);

    /**
     * Adds image map for recognition and computes the thresholds
     * @param imageMap
     */
    void addImageMap(const std::vector<LocationImage> &imageMap);

    /**
     * Adds new image for recognition
     * @param image
     */
//    LocationXY addNewTestingImage(cv::Mat image, LocationXY currentPose);


    ImageRecognitionResult addNewTestingImage(cv::Mat image);

    void resetTestingImages();

private:

    /*
     * Global_description from:
     *  * @file    OpenABLE.cpp
        * @brief   Core functions of the open place recognition method called ABLE
            (Able for Binary-appearance Loop-closure Evaluation)
        * @author  Roberto Arroyo
     */
    cv::Mat global_description(cv::Mat imageIn);

    /*
     * Computing the hamming distance between two descriptors
     */
    unsigned long long hamming_matching(cv::Mat desc1, cv::Mat desc2);

    /**
     * Computes the hamming distances between provided window and selected map sequence
     * @param imageCounter
     * @param testDescriptors
     * @param trainigDescriptors
     * @param previousDistances
     * @param onePriorToWindow
     * @return
     */
    std::vector<unsigned long long> matchWindowToSingleSequence(int imageCounter, const std::vector<cv::Mat> &trainingDescriptors,
                                                      const std::vector<cv::Mat> &testDescriptors,
                                                      cv::Mat onePriorToWindow,
                                                      std::vector<unsigned long long> &previousDistance);

    /**
     *
     * @param trainingDescriptors
     * @param testDescriptorsWindow
     * @param onePriorToWindow
     * @param previousDistances
     * @return
     */
    std::vector<std::vector<unsigned long long>> matchWindowToSequences(const std::vector<std::vector<cv::Mat> > &trainingDescriptors,
                                                                        const std::vector<cv::Mat> &testDescriptorsWindow,
                                                                        const cv::Mat onePriorToWindow,
                                                                        std::vector<std::vector<unsigned long long> > &previousDistances);


    ImageRecognitionResult performRecognition(const std::vector<cv::Mat> &testDescriptorsWindow, const cv::Mat onePriorToWindow);

    /**
     *
     * @param trainingDescriptors
     * @return
     */
    std::vector<unsigned long long> automaticThresholdEstimation(const std::vector<std::vector<Mat> > &trainingDescriptors);

    /**
     *
     * @param trainingDescriptors
     * @param testDescriptors
     * @return
     */
    unsigned long long determineMinimalHammingDistance(const std::vector<std::vector<cv::Mat> > &trainingDescriptors,
                                                                 const std::vector<cv::Mat> &testDescriptors);


//    LocationXY bestLocationGuess(ImageRecognitionResult imageRecognitionResult, LocationXY currentPose);


    // Parameters
    setFastABLE settings;

    // Map stored as a series of segments of patches
    std::vector<std::vector<cv::Mat>> mapImageSegments;
    std::vector<std::vector<LocationXY>> mapImageSegmentLocations;

    // Corresponding thresholds for segments
    std::vector<unsigned long long> mapImageSegThresholds;

    // Vector of previous distances
    std::vector<std::vector<unsigned long long> > previousDistances;

    // Current id of processed images
    int imageCounter;

    // Accumulated descriptors for lastimages
    std::vector<cv::Mat> imgDescWindow;
    cv::Mat lastImageNotInWindow;


};


#endif //GRAPHLOCALIZATION_FASTABLE_H
