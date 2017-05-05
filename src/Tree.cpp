#include "Tree.h"
#include <iostream>
TreeParam Tree::getParam() const {
    return param;
}

void Tree::setParam(const TreeParam& value) {
    param = value;
}

std::vector<BinaryTest> Tree::getBinaryTests() {
    cv::RNG rng;

    const int channelTries = 10;
    const int PixelLocationTries = 100;
    const int threshTries = 50;

    std::vector<BinaryTest> binaryTests(channelTries * PixelLocationTries * threshTries);
    int index = 0;

    //10 random RGB
    for (int channel = 0; channel < channelTries; ++channel) {
        int c = rng.uniform((int)0, (int)4);
        // //100 random pixel locations
        for (int pixelLoc = 0; pixelLoc < PixelLocationTries; ++pixelLoc) {
            int x = rng.uniform((int)0, (int) param.patchSideLength);
            int y = rng.uniform((int)0, (int) param.patchSideLength);
            // // // 50 random threshs
            for (int thres = 0; thres < threshTries; ++thres) {
                BinaryTest test;
                test.c = c;
                test.x = x;
                test.y = y;
                test.thresh = rng.uniform((int)0, (int)255);
                // at this point we have a new binary test...
                // push it (off a cliff) into the vector
                binaryTests[index++] = test;
            }
        }
    }
    return binaryTests;
}

void Tree::Train(std::vector<cv::Mat> trainImgs, std::vector<cv::Mat> trainSegMaps, int numClasses) {
    int maxNumNodes = pow(2, param.depth + 1) - 1;//full tree size
    int maxNumLeaves = (maxNumNodes + 1) / 2;

    std::vector<Sample> samplePatchesPerClass;
    generateTrainingSamples(trainImgs, trainSegMaps, samplePatchesPerClass, numClasses);
    std::cout << samplePatchesPerClass.size() << " samples created" << std::endl;


    std::vector<BinaryTest> binaryTests = getBinaryTests();
    std::cout << binaryTests.size() << " binary tests created" << std::endl;

}
