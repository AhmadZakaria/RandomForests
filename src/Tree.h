#ifndef TREE_H
#define TREE_H
#include "tree_utils.h"
#include "Node.h"
/// Implements the classification tree
class Tree {
  public:
    TreeParam getParam() const;
    void setParam(const TreeParam& value);
    void Train(std::vector<cv::Mat> trainImgs, std::vector<cv::Mat> trainSegMaps, int numClasses);
  private:
    TreeParam param;
    Node* root = new Node();
};

#endif // TREE_H
