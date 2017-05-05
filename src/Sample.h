#ifndef SAMPLE_H
#define SAMPLE_H

#include <algorithm>    // std::all_of

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

/// Structure to contain a training sample
struct Sample {
    cv::Mat* Image;
    cv::Rect bbox;
    int label;
    Sample(cv::Mat* Image_, cv::Rect bbox_, int label_) :
        Image(Image_), bbox(bbox_), label(label_) {
    }
};


#endif
