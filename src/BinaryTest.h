#ifndef BINARY_TEST_H
#define BINARY_TEST_H
#include <opencv2/core/core.hpp>

struct BinaryTest {
    int c;//channel
    int x;
    int y;
    cv::Mat patch;
    double thresh;
};

#endif
