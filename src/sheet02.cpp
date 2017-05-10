#include <iostream>
#include<vector>
#include<fstream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Tree.h"
#include "Forest.h"
#include "tree_utils.h"
#include "Sample.h"

using namespace std;

int main(int argc, char** argv) {

    std::string trainImgFilePath;
    std::string testImgFilePath;
    std::string imgPath, segMapPath;
    std::vector<cv::Mat> trainImgs, testImgs;
    std::vector<cv::Mat> trainSegMaps, testSegMaps;
    int nImg = 0;
    int nClasses = 0;
    string imgDir = "images/";
    std::vector<std::string> testImgNames;

    trainImgFilePath = "images/train_images.txt";
    testImgFilePath = "images/test_images.txt";

    /// parsing input arguments
    if (argc == 3) {
        trainImgFilePath = argv[1];
        testImgFilePath = argv[2];
    }

    ///***********************************************///
    ///            LOAD TRAINING IMAGES               ///
    ///***********************************************///
    ifstream trainImgFile(trainImgFilePath.c_str());
    if (trainImgFile.is_open()) {

        trainImgFile >> nImg;
        trainImgFile >> nClasses;

        for (int idx = 0; idx < nImg; idx++) {
            trainImgFile >> imgPath;
            imgPath = imgDir + imgPath;
            cv::Mat img = cv::imread(imgPath);
            trainImgFile >> segMapPath;
            segMapPath = imgDir + segMapPath;
            cv::Mat segMap = cv::imread(segMapPath);

            trainImgs.push_back(img);
            trainSegMaps.push_back(segMap);
        }
        cout << trainImgs.size() << " training images loaded." << endl;
    } else {
        cout << "Unable to open input file:  " << trainImgFilePath << endl;
    }

    if (trainImgs.size() < 1) {
        cout << "No training image loaded." << endl;
        return 1;
    }

    ///***********************************************///
    ///               TREE TRAINING                   ///
    ///***********************************************///

    /// Define a structure TreeParam in the file tree_utils.h
    /// All the parameters required for random trees should be placed
    /// in TreeParam structure.
    TreeParam params;

    /// Implement a class Tree in the files Tree.h & Tree.cpp
    Tree tree;

    /// set the params and start training
    tree.setParam(params);

    /// train the tree
    int64 t0 = cv::getTickCount();
    tree.Train(trainImgs, trainSegMaps, nClasses);
    int64 t1 = cv::getTickCount();
    std::cerr << "Done training tree in " << (t1 - t0) / cv::getTickFrequency() << " secs" << std::endl;
    ///***********************************************///
    ///          LOADING TEST IMAGES                  ///
    ///***********************************************///

    ifstream testImgFile(testImgFilePath.c_str());
    if (testImgFile.is_open()) {

        testImgFile >> nImg;
        testImgFile >> nClasses;

        for (int idx = 0; idx < nImg; idx++) {
            testImgFile >> imgPath;
            testImgNames.push_back(imgPath);
            imgPath = imgDir + imgPath;
            cv::Mat img = cv::imread(imgPath);
            testImgFile >> segMapPath;
            segMapPath = imgDir + segMapPath;
            cv::Mat segMap = cv::imread(segMapPath);

            testImgs.push_back(img);
            testSegMaps.push_back(segMap);
        }
        cout << testImgs.size() << " test images loaded." << endl;

        ///***********************************************///
        ///          TESTING TRAINED TREE                 ///
        ///***********************************************///

        /// evaluating each test image
        for (unsigned int idx = 0; idx < testImgs.size(); idx++) {
            cv::Mat segMapTest;
            if (tree.isTrained()) {

                /// perform segmentation
                tree.testImage(testImgs[idx], segMapTest);

                cv::Mat outputImage;
                hconcat(testImgs[idx], segMapTest, outputImage);
                cv::imshow("Segmentation Result with a Tree", outputImage);
                cv::waitKey(0);
            }
        }
        std::cout
                << "Segmentation maps for test images saved at ../assign_data/tree_outputs/"
                << std::endl;
    } else {
        std::cout << "Unable to open input file:  " << trainImgFile << endl;
    }

    ///***********************************************///
    ///           TRAINING A RANDOM FOREST            ///
    ///***********************************************///
    const int NUM_TREES = 5;
    // Implement a class Forest in the files Forest.h & Forest.cpp
    Forest forest(NUM_TREES, params);
    forest.Train(trainImgs, trainSegMaps, nClasses);

    ///***********************************************///
    ///            TESTING FOREST                     ///
    ///***********************************************///

    std::cout << "Testing images using the trained Random Forest." << std::endl;
    for (unsigned int idx = 0; idx < testImgs.size(); idx++) {
        cv::Mat segMapTest;
        forest.testImage(testImgs[idx], segMapTest);
//        cv::medianBlur(segMapTest,segMapTest,3); // makes it better, but let's not use it
        cv::Mat outputImage;
        hconcat(testImgs[idx], segMapTest, outputImage);
        cv::imshow("Output Segmentation Map with Random Forest", outputImage);
        cv::waitKey(0);
    }
    std::cout << "Done!" << endl;
}
