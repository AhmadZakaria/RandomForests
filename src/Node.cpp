#include "Node.h"
#include <iostream>
Node::Node(int depth, int numClasses) : depth(depth),
    row(0), col(0), channel(0), thresh(125), numClasses(4) {
}

void Node::trainRecursively(std::vector<Sample>& samples, std::vector<BinaryTest>& tests, int maxDepth, int minLeaves) {
    patches = samples;

    if (depth >= maxDepth || samples.size() < (minLeaves) || hasOneClassOnly()) {
        //should be leaf node;
//        std::cout << "Leaf node at depth " << depth << ", with " << samples.size() << " patches." << std::endl;
        return;
    }
    BinaryTest best = tests[0];
    double bestInfoGain = -std::numeric_limits<float>::infinity();

    for (int tIdx = 0; tIdx < tests.size(); ++tIdx) {
        BinaryTest test  = tests[tIdx];
        setParams(test);

        //pew pew pew
        pushDownSamples(samples);

        double gain = getInfoGain();
        if(gain > bestInfoGain) {
            best = test;
            bestInfoGain = gain;
        }
    }
    setParams(best);
    pushDownSamples(samples);

//    std::cout << "Depth: " << depth << ", Best InfoGain: " << bestInfoGain << " for thresh: " << thresh << std::endl;


    auto rightPatches = right->getPatches();
    right->trainRecursively(rightPatches, tests, maxDepth, minLeaves);

    auto leftPatches = left->getPatches();
    left->trainRecursively(leftPatches, tests, maxDepth, minLeaves);


}
void Node::setParams(BinaryTest test) {
    channel = test.c;
    col = test.x;
    row = test.y;
    thresh = test.thresh;
}

double Node::entropy() {
    double h = 0.0;
    std::vector<double> samplesPerClass = getProbabilities();

    for (int i = 0; i < samplesPerClass.size(); ++i) {
        double spc = samplesPerClass[i];
        if(spc < 0.0000001) //zero (double comparison)
            continue;

        double tempH = spc * log2(spc);
        h += tempH; // Shannon's entropy
    }
    return -h;
}

double Node::getInfoGain() {
    if(right->patches.empty() || left->patches.empty())
        return -std::numeric_limits<float>::infinity();
    double h_all = entropy();
    double h_right = (((float)(right->getPatches().size())) / getPatches().size()) * right->entropy();
    double h_left = (((float)(left->getPatches().size())) / getPatches().size()) * left->entropy();

    return h_all - (h_right + h_left);
}

void Node::pushDownSamples(std::vector<Sample>& samples) {
    if(right)delete right;
    if(left)delete left;
    right = new Node(depth + 1, numClasses);
    left = new Node(depth + 1, numClasses);

    for (int sIdx = 0; sIdx < samples.size(); ++sIdx) {
        Sample s = samples[sIdx];
        if (split(s) == 1)
            right->pushSample(s);
        else
            left->pushSample(s);
    }
}

int Node::split(Sample& sample) {
    int patchValue = sample.Image->at<cv::Vec3b>(sample.bbox.y + row, sample.bbox.x + col)[channel];

    if (patchValue < thresh)
        return 1;
    return 0;
}

std::vector<Sample> Node::getPatches() {
    return patches;
}

void Node::setPatches(const std::vector<Sample>& value) {
    patches = value;
}

void Node::pushSample(Sample s) {
    patches.push_back(s);
}

std::vector<double> Node::classifySample(Sample s) {
    if (right == nullptr || left == nullptr) {
        return getProbabilities();
    }
    if (split(s) == 1)
        return right->classifySample(s);
    else
        return left->classifySample(s);
}

bool Node::hasOneClassOnly() {
    std::vector<int> classExists(numClasses);
    for (int ind = 0; ind < patches.size(); ++ind) {
        Sample s = patches[ind];
        classExists[s.label] = 1;
    }
    int sum = 0;
    for (auto& n : classExists)
        sum += n;
    return sum == 1;
}

std::vector<double> Node::getProbabilities() {
    std::vector<double> probs(numClasses);
    for (int ind = 0; ind < patches.size(); ++ind) {
        Sample s = patches[ind];
        probs[s.label]+= 1.0/patches.size();
    }
//    for (int i = 0; i < probs.size(); ++i) {
//        probs[i] = ((double)probs[i]) / patches.size();
//    }
    return probs;
}

