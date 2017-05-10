#include "tree_utils.h"
#include <iostream>
#include <omp.h>
/**
 *   @brief  Generate a set of training samples for each color class
 *   by randomly extracting patches P_M of size 16 × 16 from each training
 *   image.
 *
 *   @param  trainingImgs input non-empty vector of training images
 *   @param  groundTruth input vector of ground truth images
 *   @param  samplePatchesPerClass output vector of samples.
 *   @param  nClasses number of classes
 *   @param  patchesPerClass min number of patches to be sample for each class
 *   @param  patchSize side length of extracted patches
 *   @return void
 */
void generateTrainingSamples(std::vector<cv::Mat>& trainingImgs,
                             std::vector<cv::Mat>& groundTruth,
                             std::vector<Sample>& samplePatchesPerClass, int nClasses,
                             int patchesPerClass , int patchSize ) {

    int64 state = time(NULL);
//    std::cerr << "TS State: " << state << std::endl;
    if (omp_in_parallel()) {
        state *= (1 + omp_get_thread_num());
    }
//    std::cerr << "TS State " << omp_get_thread_num() << ": " << state << std::endl;

    cv::RNG rng(state);

    int xLowerBound = 0, yLowerBound = 0;
    int xHigherBound = (trainingImgs[0].cols - patchSize + 1);
    int yHigherBound = (trainingImgs[0].rows - patchSize + 1);
    int halfLength = patchSize / 2.0f;
    std::vector<int> count(nClasses);

    // while /*not all classes are filled*/
    while (std::any_of(count.begin(), count.end(),
    [&patchesPerClass](int i) {
    return i < patchesPerClass;
})) {
        //get new x and y for the bbox
        int x = rng.uniform(xLowerBound, xHigherBound);
        int y = rng.uniform(yLowerBound, yHigherBound);

        // get random image
        int idx = rng.uniform(0, trainingImgs.size());
        cv::Mat sampleMat = trainingImgs[idx];

        // pixel to be tested
        int row = y + halfLength;
        int col = x + halfLength;

        // ground truth of the pixel
        int gt = groundTruth[idx].at<cv::Vec3b>(row, col)[0];

//        if (count[gt] < patchesPerClass) {
        samplePatchesPerClass.push_back(
            Sample(new cv::Mat(sampleMat), cv::Rect(x, y, patchSize, patchSize),
                   gt));
        count[gt]++;
//        }
//		std::cout << samplePatchesPerClass.size() << ": " << count[0] << ", "
//				<< count[1] << ", " << count[2] << ", " << count[3]
//				<< std::endl;
    }
}
