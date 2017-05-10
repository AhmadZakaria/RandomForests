#ifndef TREE_H
#define TREE_H
#include "tree_utils.h"
#include "Node.h"
#include "BinaryTest.h"
/// Implements the classification tree
class Tree {
  public:
    Tree();
    Tree(TreeParam& params);
    TreeParam getParam() const;
    void setParam(const TreeParam& value);
    void Train(std::vector<cv::Mat> &trainImgs, std::vector<cv::Mat> &trainSegMaps, int &numClasses);
    bool isTrained() const;
    double testImage(cv::Mat &testImg, cv::Mat &segMapOut);
    int classifySample(Sample &s, std::vector<double> &probs);
    int classifySample(Sample &s);
private:
    TreeParam param;
    Node* root = nullptr;
    std::vector<BinaryTest> getBinaryTests();
    bool trained = false;
    std::vector<cv::Scalar> colors;

};

#endif // TREE_H
