#include "Node.h"
#include <iostream>
Node::Node(int depth, int maxDepth, int numClasses, int initSize) : depth(depth), maxDepth(maxDepth),
    row(0), col(0), channel(0), thresh(125), numClasses(numClasses) {
    patches.reserve(initSize);
}

Node::Node(int depth, int maxDepth, int numClasses) : depth(depth), maxDepth(maxDepth),
    row(0), col(0), channel(0), thresh(125), numClasses(numClasses) {
    patches.reserve(30);
}

Node::~Node() {
    delete right;
    delete left;
}

void Node::trainRecursively(std::vector<Sample>& samples, std::vector<BinaryTest>& tests, int maxDepth, int minLeaves) {


    if (depth >= maxDepth || (samples.size() < (2 * minLeaves)) || hasOneClassOnly()) {
        //should be leaf node;
        patches = samples;
        return;
    }

    double bestInfoGain = -std::numeric_limits<float>::infinity();
    BinaryTest best = tests[0];
    for (int tIdx = 0; tIdx < tests.size(); ++tIdx) {
        BinaryTest test  = tests[tIdx];
        setParams(test);

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
    Node*l = left;
    Node*r = right;

    std::vector<Sample> rightPatches = r->getPatches();
    r->trainRecursively(rightPatches, tests, maxDepth, minLeaves);

    std::vector<Sample> leftPatches = l->getPatches();
    l->trainRecursively(leftPatches, tests, maxDepth, minLeaves);

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
    double h_right = (((float)(right->getPatches().size()))) * right->entropy();
    double h_left = (((float)(left->getPatches().size())) ) * left->entropy();

    return h_all - ((h_right + h_left) / getPatches().size());
}



void Node::pushDownSamples(std::vector<Sample>& samples) {
    if(right) delete right;
    if (left) delete left;
    right = new Node(depth + 1, maxDepth, numClasses, samples.size());
    left = new Node(depth + 1, maxDepth, numClasses, samples.size());
    for (int sIdx = 0; sIdx < samples.size(); ++sIdx) {
        Sample s = samples[sIdx];
        if (split(s))
            right->pushSample(s);
        else
            left->pushSample(s);
    }
}

bool Node::split(Sample& sample) {
    int patchValue = sample.Image->at<cv::Vec3b>(sample.bbox.y + row, sample.bbox.x + col)[channel];

    return (patchValue < thresh);
}

std::vector<Sample>& Node::getPatches() {
    return patches;
}

void Node::setPatches(const std::vector<Sample>& value) {
    patches = value;
}

void Node::pushSample(Sample& s) {
    patches.push_back(s);
}

std::vector<double> Node::classifySample(Sample s) {
    if (right == nullptr || left == nullptr) {
        return getProbabilities();
    }
    if (split(s))
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
    double increment = 1.0 / patches.size();
    std::vector<double> probs(numClasses);
    for (int ind = 0; ind < patches.size(); ++ind) {
        Sample s = patches[ind];
        probs[s.label] += increment;
    }
    return probs;
}

