#pragma once
#ifndef NODE_H
#define NODE_H
#include <opencv2/core/core.hpp>
#include "Sample.h"
#include "BinaryTest.h"
/// Node of a tree
class Node {

  public:
    Node(int depth, int maxDepth, int numClasses, int initSize);
    Node(int depth, int maxDepth, int numClasses);
    ~Node();
    void trainRecursively(std::vector<Sample>& samples, std::vector<BinaryTest>& tests, int maxDepth, int minLeaves);
    double entropy();
    double getInfoGain();
    void pushDownSamples(std::vector<Sample>& samples);
    bool split(Sample& sample);

    std::vector<Sample> &getPatches();
    void setPatches(const std::vector<Sample>& value);
    void pushSample(Sample &s);
    std::vector<double> classifySample(Sample s);
    bool hasOneClassOnly();
    std::vector<double> getProbabilities();

  private:
    Node* right = nullptr;
    Node* left = nullptr;

    int depth;
    int maxDepth;

    int row;
    int col;
    int numClasses;
    int channel;
    double thresh;
    std::vector<Sample> patches;
    void setParams(BinaryTest test);
};

#endif // NODE_H
