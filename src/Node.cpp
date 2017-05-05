#include "Node.h"
Node::Node(int row, int col, cv::Vec3b RGB, double thresh, int numClasses) :
    row(row), col(col), pixelRGB(RGB), thresh(thresh), numClasses(numClasses) {
}

double Node::entropy() {
    double h = 0.0;
    std::vector<int> samplesPerClass(numClasses);
    for (int ind = 0; ind < points.size(); ++ind) {
        Sample s = points[ind];
        samplesPerClass[s.label]++;
    }
    for (int i = 0; i < samplesPerClass.size(); ++i) {
        int spc = samplesPerClass[i];

        double p_c = ((double)spc) / points.size();
        h += p_c * log2(p_c); // Shannon's entropy
    }
    return -h;
}

double Node::infoGain(std::vector<Sample>& samples, int numClasses) {
    double h_all = entropy();
    double h_right = (((float)(right->getPoints().size())) / getPoints().size()) * right->entropy();
    double h_left = (((float)(left->getPoints().size())) / getPoints().size()) * left->entropy();

    return h_all - (h_right + h_left);
}

/**
 * @brief f distance function
 * @param sample patch to be classified
 * @return distance from patch to node
 */
double Node::f(Sample& sample) const {
    cv::Vec3b patchValue = sample.Image->at<cv::Vec3b>(sample.bbox.y, sample.bbox.x);
    return cv::norm(patchValue, pixelRGB, CV_L2);
}

int Node::split(Sample& sample) {
    if (f(sample) < thresh)
        return 1;
    return 0;
}

std::vector<Sample> Node::getPoints() const {
    return points;
}

void Node::setPoints(const std::vector<Sample>& value) {
    points = value;
}

