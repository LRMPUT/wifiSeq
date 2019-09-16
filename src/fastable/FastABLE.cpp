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

#include "fastable/FastABLE.h"

FastABLE::FastABLE(setFastABLE settings) : settings(settings) {


}


void FastABLE::addImageMap(const std::vector<LocationImage> &imageMap) {

    // Divide into map segments
    std::vector<cv::Mat> imageSegment;
    std::vector<LocationXY> imageSegmentLocations;
    for (int i = 0; i < imageMap.size(); i++) {

        // Different segment
        if (!imageSegment.empty() && imageMap[i - 1].segmentId != imageMap[i].segmentId) {

            // Only adding if segment is larger than compareLength
            if (imageSegment.size() > settings.compareLength) {
                mapImageSegments.push_back(imageSegment);
                mapImageSegmentLocations.push_back(imageSegmentLocations);
            }
            imageSegment.clear();
            imageSegmentLocations.clear();
        }

        imageSegment.push_back(global_description(imageMap[i].image));
        imageSegmentLocations.push_back(imageMap[i].locationXY);

//        std::cout << "?? " << imageMap[i].locationXY.x << " " << imageMap[i].locationXY.y << " " << imageMap[i].locationXY.id << std::endl;
    }


    // Only adding if segment is larger than compareLength
    if (imageSegment.size() > settings.compareLength) {
        mapImageSegments.push_back(imageSegment);
        mapImageSegmentLocations.push_back(imageSegmentLocations);
    }

    imageCounter = 0;
    previousDistances = std::vector<std::vector<unsigned long long> >{mapImageSegments.size(), std::vector<unsigned long long>()};

    // Threshold estimation for segments
    mapImageSegThresholds = automaticThresholdEstimation(mapImageSegments);

    for (unsigned long long &v : mapImageSegThresholds)
        std::cout << "FastABLE estimated thresholds: " << v << std::endl;
}


//LocationXY FastABLE::addNewTestingImage(cv::Mat image, LocationXY currentPose) {
//
//    // Computing descriptor
//    cv::Mat imgDesc = global_description(image);
//    imgDescWindow.push_back(imgDesc);
//
//
//    // Moving window of images, also saving the last one not in the window
//    if (imgDescWindow.size() > settings.compareLength) {
//
//        lastImageNotInWindow = imgDescWindow.front();
//        imgDescWindow.erase(imgDescWindow.begin());
//    }
//
//    // Recognition
//    ImageRecognitionResult imageRecognitionResult;
//    if (imgDescWindow.size() == settings.compareLength) {
//        imageRecognitionResult = performRecognition(imgDescWindow, lastImageNotInWindow);
//
//        LocationXY bestGuess = bestLocationGuess(imageRecognitionResult, currentPose);
//        return bestGuess;
//    }
//
//    return LocationXY();
//}

ImageRecognitionResult FastABLE::addNewTestingImage(cv::Mat image) {

    // Computing descriptor
    cv::Mat imgDesc = global_description(image);
    imgDescWindow.push_back(imgDesc);


    // Moving window of images, also saving the last one not in the window
    if (imgDescWindow.size() > settings.compareLength) {

        lastImageNotInWindow = imgDescWindow.front();
        imgDescWindow.erase(imgDescWindow.begin());
    }

    // Recognition
    ImageRecognitionResult imageRecognitionResult{0};
    if (imgDescWindow.size() == settings.compareLength) {
        imageRecognitionResult = performRecognition(imgDescWindow, lastImageNotInWindow);
    }

    imageCounter++;

    return imageRecognitionResult;
}

void FastABLE::resetTestingImages() {
    imageCounter = 0;
    previousDistances = std::vector<std::vector<unsigned long long> >{mapImageSegments.size(), std::vector<unsigned long long>()};
    imgDescWindow.clear();
    lastImageNotInWindow = cv::Mat();
}

/*
 * Global_description from:
 *  * @file    OpenABLE.cpp
    * @brief   Core functions of the open place recognition method called ABLE
        (Able for Binary-appearance Loop-closure Evaluation)
    * @author  Roberto Arroyo
 */
cv::Mat FastABLE::global_description(cv::Mat imageIn) {

    // To grayscale
    cv::Mat image;
    cv::cvtColor(imageIn, image, COLOR_BGR2GRAY);

    // Resize the image
    cv::Mat image_resized;
    cv::resize(image, image_resized, cv::Size(settings.patchSize, settings.patchSize), 0, 0,
               INTER_LINEAR);

    // Select the central keypoint
    std::vector<cv::KeyPoint> kpts;
    cv::KeyPoint kpt;
    kpt.pt.x = settings.patchSize / 2 + 1;
    kpt.pt.y = settings.patchSize / 2 + 1;
    kpt.size = 1.0;
    kpt.angle = 0.0;
    kpts.push_back(kpt);

    // Computing the descriptor
    cv::Mat descriptor;
    LDB ldb;
    ldb.compute(image_resized, kpts, descriptor);

    return descriptor;

}

unsigned long long FastABLE::hamming_matching(cv::Mat desc1, cv::Mat desc2) {

    unsigned long long distance = 0;

    if (desc1.rows != desc2.rows || desc1.cols != desc2.cols || desc1.rows != 1
        || desc2.rows != 1) {

        cout << "The dimension of the descriptors is different." << desc1.rows << " " << desc1.cols << " " << desc2.rows
             << " " << desc2.cols << endl;
        return -1;

    }

    for (unsigned long long i = 0; i < desc1.cols; i++) {
        distance += __builtin_popcount((*(desc1.ptr<unsigned char>(0) + i))
                                       ^ (*(desc2.ptr<unsigned char>(0) + i)));
    }

    // MNowicki: Alternatively, it is possible to compute distance with OpenCV
    //double dist = cv::norm( desc1, desc2, NORM_HAMMING);

    return distance;
}

std::vector<unsigned long long> FastABLE::matchWindowToSingleSequence(int imageCounter, const std::vector<cv::Mat> &trainingDescriptors,
                                                                      const std::vector<cv::Mat> &testDescriptorsWindow,
                                                                      cv::Mat onePriorToWindow,
                                                                      std::vector<unsigned long long> &previousDistances) {
//    std::cout << "FastABLE::matchWindowToSingleSequence - begin" << std::endl;

    std::vector<unsigned long long> matchingResult;

    int trainingSize = trainingDescriptors.size();

    std::vector<unsigned long long> nextPreviousDistances;

    for (int trainingShift = 0;
         trainingShift < trainingSize - settings.compareLength + 1;
         trainingShift++) {


        unsigned long long distance = 0;

        // If it is possible to make it faster
        if (imageCounter != settings.compareLength - 1 // Not first iteration
            && previousDistances.empty() == false // We have previous distances
            && trainingShift != 0 // It has to be computed normally
            ) {

            // We compute current result as a result from previous iteration adding last and subtracting first
            distance = previousDistances[trainingShift - 1]
                       + hamming_matching(testDescriptorsWindow[settings.compareLength - 1],
                                          trainingDescriptors[trainingShift + settings.compareLength - 1])
                       - hamming_matching(onePriorToWindow,
                                          trainingDescriptors[trainingShift - 1]);

            // Let's save those precomputed distances for next iteration
            nextPreviousDistances.push_back(distance);
        }
        // Computing everything for 1st window and for every first comparison for a new frame
        else {
            // for each element in the sequence to be matched
            for (int k = 0; k < settings.compareLength; k++) {

                // Distance between:
                // 		testDescriptorsWindow
                // 		trainDescriptors shifted by (trainingShift)
                distance = distance
                           + hamming_matching(testDescriptorsWindow[k],
                                              trainingDescriptors[trainingShift + k]);
            }
            // if we just started, we initialize the previousDistances to later use modified recurrence
            if (imageCounter == settings.compareLength - 1)
                nextPreviousDistances.push_back(distance);

            // the first comparison when we already have previous distances
            else if (trainingShift == 0)
                nextPreviousDistances.push_back(distance);
        }

        // Saving the result
        matchingResult.push_back(distance);
    }

    // Copying the previous results
    previousDistances = nextPreviousDistances;

//    std::cout << "FastABLE::matchWindowToSingleSequence - end" << std::endl;
    return matchingResult;
}



std::vector<std::vector<unsigned long long>> FastABLE::matchWindowToSequences(const std::vector<std::vector<cv::Mat> > &trainingDescriptors,
                                                  const std::vector<cv::Mat> &testDescriptorsWindow,
                                                  const cv::Mat onePriorToWindow,
                                                  std::vector<std::vector<unsigned long long> > &previousDistances) {
//    std::cout << "FastABLE::matchWindowToSequences - start" << std::endl;

    std::vector<std::vector<unsigned long long>> results;

    // For each map segment
    for (uint i = 0; i < trainingDescriptors.size(); i++) {

        // Image matching against i-th training part
        std::vector<unsigned long long> matchingResult = matchWindowToSingleSequence(imageCounter, trainingDescriptors[i],
                testDescriptorsWindow, onePriorToWindow, previousDistances[i]);

        results.emplace_back(matchingResult);
    }

//    std::cout << "FastABLE::matchWindowToSequences - end" << std::endl;
    return results;
}



ImageRecognitionResult FastABLE::performRecognition(const std::vector<cv::Mat> &testDescriptorsWindow, const cv::Mat onePriorToWindow) {
//    std::cout << "FastABLE::performRecognition - start" << std::endl;

    // Match current window
    std::vector<std::vector<unsigned long long>> resultsForAll = matchWindowToSequences( this->mapImageSegments,
            testDescriptorsWindow, onePriorToWindow, this->previousDistances);

//    std::cout << "FastABLE::performRecognition - after recognition" << std::endl;

    ImageRecognitionResult imageRecognitionResult;

    // Go through results
    for (int i=0;i<this->mapImageSegments.size();i++) {

        // Consider matching for selected segment from map
        auto & resultForSegment = resultsForAll[i];
        auto & segmentThreshold = this->mapImageSegThresholds[i];

        // For all performed matches
        for (int j=0;j<resultForSegment.size();j++) {

            // Correct recognition
            if (resultForSegment[j] < segmentThreshold ){

//                std::cout << "Correct recognition! " << resultForSegment[j] << " < " << segmentThreshold << std::endl;
//                std::cout << "Location: " << mapImageSegmentLocations[i][j].x << " " << mapImageSegmentLocations[i][j].y << std::endl;

                double eps = 1e-6;
                imageRecognitionResult.matchingWeights.push_back((double)resultForSegment[j] / this->mapImageSegThresholds[i]);
                imageRecognitionResult.matchingLocations.push_back(mapImageSegmentLocations[i][j + settings.compareLength - 1]);
            }
        }
    }

    imageRecognitionResult.correctRecognitions = imageRecognitionResult.matchingLocations.size();

//    std::cout << "FastABLE::performRecognition - end" << std::endl;
    return imageRecognitionResult;
}


std::vector<unsigned long long> FastABLE::automaticThresholdEstimation(const std::vector<std::vector<Mat> > &trainingDescriptors) {
    std::cout << "FastABLE - automaticThresholdEstimation for " <<  trainingDescriptors.size() << " sequences" << std::endl;

//    for (uint i = 0; i < trainingDescriptors.size(); i++) {
//        std::cout << "Sizes : " << trainingDescriptors[i].size() << std::endl;
//    }

    // We look for separate threshold for each training sequence
    std::vector<unsigned long long> localThresholds;
    for (uint i = 0; i < trainingDescriptors.size(); i++) {

        // Selecting one training sequence as test
        std::vector<Mat> chosenDesc = trainingDescriptors[i];

        // The remaining sequences remain as training sequences
        std::vector<std::vector<Mat> > tmpGroundTruthDescriptors(trainingDescriptors);
        tmpGroundTruthDescriptors.erase(tmpGroundTruthDescriptors.begin() + i);

        // Let's compute recognitions
        std::cout << "FastABLE - sequence id = " << i << " length = " << chosenDesc.size() << std::endl;
        unsigned long long minThreshold = determineMinimalHammingDistance(tmpGroundTruthDescriptors, chosenDesc);

        localThresholds.push_back(minThreshold * settings.safetyThresholdRatio);

//        std::cout << "FastABLE - minimal threshold = " << minThreshold * 1.0 / compareLength << std::endl;
    }

    // Clear for new recognition
    imageCounter = 0;
    previousDistances = std::vector<std::vector<unsigned long long> >{trainingDescriptors.size(), std::vector<unsigned long long>()};

    return localThresholds;
}

unsigned long long FastABLE::determineMinimalHammingDistance(const std::vector<std::vector<cv::Mat> > &trainingDescriptors,
                                                             const std::vector<cv::Mat> &testDescriptors) {
//    std::cout << "FastABLE::determineMinimalHammingDistance - begin" << std::endl;

    // Clearing for automatic threshold
    imageCounter = 0;
    previousDistances = std::vector<std::vector<unsigned long long> >{trainingDescriptors.size(), std::vector<unsigned long long>()};

    // Let's determine the global minimum
    unsigned long long globalMin = std::numeric_limits<unsigned long long>::max();

    // For each new test descriptor
    for (int imageIndex = 0; imageIndex < testDescriptors.size(); imageIndex++) {

        // Selecting window
        int indexBegin = std::max(0, imageIndex - settings.compareLength + 1);
        int indexLast = imageIndex;
        std::vector<cv::Mat> testDescriptorsWindow(testDescriptors.begin() + indexBegin,
                                                   testDescriptors.begin() + indexLast + 1);

        // Last frame not in a window
        cv::Mat onePriorToWindow;
        if (indexBegin > 0)
            onePriorToWindow = testDescriptors[indexBegin - 1];

//        std::cout << indexBegin << " " << indexLast << " " << testDescriptorsWindow.size() << " " << imageCounter << std::endl;

        // Proper size for comparison
        if(testDescriptorsWindow.size() == settings.compareLength) {

            std::vector<std::vector<unsigned long long>> matchingResult = matchWindowToSequences(trainingDescriptors, testDescriptorsWindow, onePriorToWindow, previousDistances);

//            std::cout << "SIZE: " << matchingResult.size() << std::endl;
//            int aaa = 0;

            // Current minimum
            for (auto &sequenceResult : matchingResult) {
//                std::cout << "sequenceResult: " << sequenceResult.size() << " " << trainingDescriptors[aaa].size() << std::endl;
                unsigned long long localMin = *std::min_element(sequenceResult.begin(), sequenceResult.end());
//                std::cout << "localMin = " << localMin << " " << sequenceResult.size() <<  std::endl;

                globalMin = std::min(globalMin, localMin);
            }
        }

        imageCounter++;
    }

//    std::cout << "FastABLE::determineMinimalHammingDistance - end" << std::endl;
    return globalMin;
}

//LocationXY FastABLE::bestLocationGuess(ImageRecognitionResult imageRecognitionResult, LocationXY currentPose) {
//
//
//
//    LocationXY locationEstimate;
//    for(int j=0;j<3;j++)
//    {
//        double x = 0.0, y = 0.0, weightSum = 0.0;
//        locationEstimate.id = -1;
//        std::vector<int> indiciesToRemove;
//
//        // For all matches
//        for (int i = 0; i < imageRecognitionResult.matchingWeights.size(); i++) {
//
//            // Current recognition
//            LocationXY &guess = imageRecognitionResult.matchingLocations[i];
//
//            // Distant to the current estimate
//            double distance = distL2(guess, currentPose);
//
//            // Recognition has to be closer than 5 meters to the current position estimate
//            if (distance < settings.earlyAcceptedVicinity) {
//                x += imageRecognitionResult.matchingWeights[i] * imageRecognitionResult.matchingLocations[i].x;
//                y += imageRecognitionResult.matchingWeights[i] * imageRecognitionResult.matchingLocations[i].y;
//                weightSum += imageRecognitionResult.matchingWeights[i];
//                locationEstimate.id = 1;
//            }
//            else {
////                std::cout << "indiciesToRemove - add " << i << std::endl;
//                indiciesToRemove.push_back(i);
//            }
////        std::cout << "w=" <<  imageRecognitionResult.matchingWeights[i] << " x=" <<
////            imageRecognitionResult.matchingLocations[i].x << " y=" << imageRecognitionResult.matchingLocations[i].y << std::endl;
//        }
//
//        // No matches found
//        if (locationEstimate.id == -1)
//            return LocationXY();
//
//        // Computing current estimate
//        locationEstimate.x = x / weightSum;
//        locationEstimate.y = y / weightSum;
//
//        if (indiciesToRemove.size() == 0)
//            return locationEstimate;
//
//
////        std::cout << "id=" <<  locationEstimate.id << " x=" << locationEstimate.x << " y=" << locationEstimate.y << std::endl;
//
//        sort(indiciesToRemove.begin(), indiciesToRemove.end(), std::greater<int>());
//        for (int i=0;i<indiciesToRemove.size();i++) {
//            int idx = indiciesToRemove[i];
////            std::cout << "Removing 1 : " << idx << " " << imageRecognitionResult.matchingWeights.size() << std::endl;
//            imageRecognitionResult.matchingWeights.erase(imageRecognitionResult.matchingWeights.begin() + idx);
//            imageRecognitionResult.matchingLocations.erase(imageRecognitionResult.matchingLocations.begin() + idx);
//        }
//        indiciesToRemove.clear();
//
//        // Chceck consistency w.r.t the estimate
//        for (int i = 0; i < imageRecognitionResult.matchingWeights.size(); i++) {
//
//            // Computing the distance to average
//            double errDist = distL2(locationEstimate, imageRecognitionResult.matchingLocations[i]);
//
//            // Removing that match
//            if (errDist > settings.consistencyThreshold) {
////                std::cout << "indiciesToRemove -  consistency add " << i << std::endl;
////            std::cout << "Consistency issue! Recognition further than " << consistencyThreshold << " meters from average" << std::endl;
//                indiciesToRemove.push_back(i);
//            }
//        }
//
//        // We reach equilibrium
////        if (indiciesToRemove.size() == 0)
////            break;
//
////        std::cout << "Removing - start" << std::endl;
//
//        // We need to further remove some matches
//        sort(indiciesToRemove.begin(), indiciesToRemove.end(), std::greater<int>());
//
//        for (int i=0;i<indiciesToRemove.size();i++) {
//            int idx = indiciesToRemove[i];
////            std::cout << "Removing 2: " << idx << " " << imageRecognitionResult.matchingWeights.size() << std::endl;
//            imageRecognitionResult.matchingWeights.erase(imageRecognitionResult.matchingWeights.begin() + idx);
//            imageRecognitionResult.matchingLocations.erase(imageRecognitionResult.matchingLocations.begin() + idx);
//        }
//    }
//
//    std::cout << "id=" <<  locationEstimate.id << " x=" << locationEstimate.x << " y=" << locationEstimate.y << std::endl;
//
//    // TODO DEBUG
////    if (std::isnan(locationXY.x)) {
////        std::cin >> locationXY.x;
////    }
//
//    return locationEstimate;
//}


