#include "Tree.h"
#include "BinaryTest.h"
TreeParam Tree::getParam() const {
    return param;
}

void Tree::setParam(const TreeParam& value) {
    param = value;
}

void Tree::Train(std::vector<cv::Mat> trainImgs, std::vector<cv::Mat> trainSegMaps, int numClasses) {
    int maxNumNodes = pow(2, param.depth + 1) - 1;//full tree size
    cv::RNG rng;
    BinaryTest test;
    //10 random RGB
    for (int channel = 0; channel < 10; ++channel) {
        test.c = rng.uniform((int)0, (int)4);

        // //100 random pixel locations
        // // // 50 random threshs
    }
}
