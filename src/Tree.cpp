#include "Tree.h"
#include <iostream>
#include <omp.h>
Tree::Tree() {

}

Tree::Tree(TreeParam& params) {
    setParam(params);
}

TreeParam Tree::getParam() const {
    return param;
}

void Tree::setParam(const TreeParam& value) {
    param = value;
    if(root != nullptr)
        delete root;
    int rootDepth = 0;
    #pragma omp parallel
    root = new Node(rootDepth, param.depth, param.numClasses);

    colors.resize(param.numClasses);
    if(param.numClasses == 4) {
        colors[0] = 0; //black
        colors[1] = cv::Scalar(0, 0, 255); // red
        colors[2] = cv::Scalar(255, 0, 0); // blue
        colors[3] = cv::Scalar(0, 255, 0); // green

    }   else {
        int64 state = time(NULL);
        if (omp_in_parallel()) {
            state *= (1 + omp_get_thread_num());
        }
        cv::RNG rng(state);
        for (int i = 0; i < colors.size(); ++i) {
            cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            colors[i] = color;
        }
    }
}

std::vector<BinaryTest> Tree::getBinaryTests() {
    int64 state = time(NULL);
    if (omp_in_parallel()) {
        state *= (1 + omp_get_thread_num());
    }
    cv::RNG rng(state);

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
                // push it into the vector
                binaryTests[index++] = test;
            }
        }
    }
    return binaryTests;
}

bool Tree::isTrained() const {
    return trained;
}

double Tree::testImage(cv::Mat& testImg, cv::Mat& segMapOut) {
    segMapOut.create(testImg.rows, testImg.cols, testImg.type());
    for (int col = 0; col < (testImg.cols - param.patchSideLength); ++col) {
        for (int row = 0; row < (testImg.rows - param.patchSideLength); ++row) {
            cv::Rect bbox(col, row, param.patchSideLength, param.patchSideLength);
            Sample s(&testImg, bbox, -1);
            int classIdx = classifySample(s);
            int x =  col + param.patchSideLength / 2.0f;
            int y =  row + param.patchSideLength / 2.0f;
            segMapOut.at<cv::Vec3b>(y, x)[0] = colors[classIdx][0];
            segMapOut.at<cv::Vec3b>(y, x)[1] = colors[classIdx][1];
            segMapOut.at<cv::Vec3b>(y, x)[2] = colors[classIdx][2];

        }
    }
}

int Tree::classifySample(Sample& s) {
    std::vector<double> probs = root->classifySample(s);

    int classNum = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
    return classNum;
}

int Tree::classifySample(Sample& s, std::vector<double>& probs) {
    std::vector<double> probs_ = root->classifySample(s);
    for (int var = 0; var < probs.size(); ++var) {
        probs[var] += probs_[var];
    }
    int classNum = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
    return classNum;
}

void Tree::Train(std::vector<cv::Mat>& trainImgs, std::vector<cv::Mat>& trainSegMaps, int& numClasses) {
//    int maxNumNodes = pow(2, param.depth + 1) - 1;//full tree size
//    int maxNumLeaves = (maxNumNodes + 1, numClasses) / 2;

    std::vector<Sample> samplePatchesPerClass;
    samplePatchesPerClass.reserve(4500);
    generateTrainingSamples(trainImgs, trainSegMaps, samplePatchesPerClass, numClasses, param.minTrainPatchesPerClass, param.patchSideLength);
    std::cout << samplePatchesPerClass.size() << " samples created for tree " << omp_get_thread_num() + 1 << std::endl;

    std::vector<BinaryTest> binaryTests = getBinaryTests();

    root->trainRecursively(samplePatchesPerClass, binaryTests, param.depth, param.minPatchesAtLeaf);

    trained = true;

}
