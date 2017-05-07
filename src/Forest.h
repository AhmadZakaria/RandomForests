#ifndef FOREST_H
#define FOREST_H

#include "tree_utils.h"
#include "Tree.h"
/// implements a classification forest
class Forest {
  public:
    Forest(int numTrees, TreeParam& params);
    void Train(std::vector<cv::Mat>& trainImgs, std::vector<cv::Mat>& trainSegMaps, int& numClasses);
    double testImage(cv::Mat& testImg, cv::Mat& segMapOut);

  private:
    void initColors();
    std::vector<Tree> forestTrees;
    TreeParam param;
    std::vector<cv::Scalar> colors;
};

#endif // FOREST_H
