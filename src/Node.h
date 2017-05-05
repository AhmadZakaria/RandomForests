#pragma once
#ifndef NODE_H
#define NODE_H
#include <opencv2/core/core.hpp>
#include "Sample.h"
/// Node of a tree
class Node {

  public:
    Node(int row, int col, cv::Vec3b RGB, double thresh, int numClasses);
    double train(std::vector<Sample>& samples);
    double entropy();
    double infoGain(std::vector<Sample>& samples, int numClasses);
    double f(Sample& sample) const;
    int split(Sample& sample); // returns info gain

    std::vector<Sample> getPoints() const;
    void setPoints(const std::vector<Sample>& value);

  private:
    Node* right;
    Node* left;

    int row;
    int col;
    int numClasses;
    cv::Vec3b pixelRGB;
    double thresh;
    std::vector<Sample> points;
};

#endif // NODE_H
