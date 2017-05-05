#ifndef TREE_UTILS_H
#define TREE_UTILS_H
#include "Sample.h"
// labels for classes
enum {VOID = 0, SHEEP = 1, WATER = 2, GRASS = 3};

/// structure to contain parameters for the tree
struct TreeParam{
int depth = 15;
int minPatchesAtLeaf = 20;
};

/**
 *   @brief  Generate a set of training samples for each color class
 *   by randomly extracting patches P_M of size 16 × 16 from each training
 *   image.
 *
 *   @param  trainingImgs input non-empty vector of training images
 *   @param  groundTruth input vector of ground truth images
 *   @param  samplePatchesPerClass output vector of samples.
 *   @param  nClasses number of classes
 *   @param  patchesPerClass number of patches to be sample for each class
 *   @param  patchSize side length of extracted patches
 *   @return void
 */
void generateTrainingSamples(std::vector<cv::Mat>& trainingImgs,
                              std::vector<cv::Mat>& groundTruth,
                              std::vector<Sample>& samplePatchesPerClass, int nClasses,
                              int patchesPerClass = 50, int patchSize = 16);


#endif // TREE_UTILS_H
