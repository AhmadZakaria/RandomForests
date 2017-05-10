#include "Forest.h"

#include <iostream>
#include <omp.h>

Forest::Forest(int numTrees, TreeParam& params) : param(params) {
    forestTrees.reserve(numTrees);
    for (int i = 0; i < numTrees; ++i) {
        forestTrees.push_back(Tree(params));
    }
    initColors();
}

void Forest::Train(std::vector<cv::Mat>& trainImgs, std::vector<cv::Mat>& trainSegMaps, int& numClasses) {
    std::cerr << "Training " << forestTrees.size() << " trees." << std::endl;
    #pragma omp parallel for
    for (int i = 0; i < forestTrees.size(); ++i) {
        int64 t0 = cv::getTickCount();
        forestTrees[i].Train(trainImgs, trainSegMaps, numClasses);
        int64 t1 = cv::getTickCount();
        std::cerr << "Done training tree:\t" << i + 1  << ", in " << (t1 - t0) / cv::getTickFrequency() << " secs" << std::endl;
    }
}

double Forest::testImage(cv::Mat& testImg, cv::Mat& segMapOut) {
    segMapOut.create(testImg.rows, testImg.cols, testImg.type());
    #pragma omp parallel
    {

        #pragma omp  for collapse(2)
        for (int col = 0; col < (testImg.cols - param.patchSideLength); ++col) {
            for (int row = 0; row < (testImg.rows - param.patchSideLength); ++row) {
                std::vector<double> votes;
                votes.resize(param.numClasses);
                for (int i = 0; i < forestTrees.size(); ++i) {
                    cv::Rect bbox(col, row, param.patchSideLength, param.patchSideLength);
                    Sample s(&testImg, bbox, -1);
                    forestTrees[i].classifySample(s, votes);
                }
                int winnerClass = std::distance(votes.begin(), std::max_element(votes.begin(), votes.end()));

                int x =  col + param.patchSideLength / 2.0f;
                int y =  row + param.patchSideLength / 2.0f;

                segMapOut.at<cv::Vec3b>(y, x)[0] = colors[winnerClass][0];
                segMapOut.at<cv::Vec3b>(y, x)[1] = colors[winnerClass][1];
                segMapOut.at<cv::Vec3b>(y, x)[2] = colors[winnerClass][2];

            }
        }
    }
}
void Forest::initColors() {

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
