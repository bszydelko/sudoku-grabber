#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>


using namespace cv;

class DigitRecognizer
{
public:
    DigitRecognizer();

    int classify(const Mat& img);
    bool loadDigits();
    bool prepareDigits();

private:
    Mat preprocessImage(const Mat& img);


private:
    std::vector<std::pair<Mat, std::vector<Point>>> digits;
};
