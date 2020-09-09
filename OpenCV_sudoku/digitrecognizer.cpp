#include "digitrecognizer.h"


DigitRecognizer::DigitRecognizer()
{
}

int DigitRecognizer::classify(const cv::Mat& img)
{
    Mat cellImg = preprocessImage(img); //chyba niepotrzebne
    img.copyTo(cellImg);

    Mat cellImg_float;
    Mat digit_float;

    cellImg.convertTo(cellImg_float, CV_32F);
    Mat resized;

    Scalar digit_mean, digit_std, cellImg_mean, cellImg_std;
    double covar;
    double correl;

    int pixels = cellImg_float.rows * cellImg_float.cols;

    std::vector<std::pair<double,int>> vecCorrel;
    std::vector<std::vector<Point>> cnts;

    int number = 1;

    findContours(cellImg, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (auto& digit : digits)
    {
        cnts.clear();
        resize(digit.first, resized, cellImg.size());
        resized.convertTo(digit_float, CV_32F);


        meanStdDev(cellImg_float, cellImg_mean, cellImg_std);
        meanStdDev(digit_float, digit_mean, digit_std);
        covar = (cellImg_float - cellImg_mean).dot(digit_float - digit_mean) / pixels;
        correl = covar / (cellImg_std[0] * digit_std[0]);

        vecCorrel.emplace_back(correl,number);
        number++;
    }

    std::sort(vecCorrel.begin(), vecCorrel.end(), 
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) -> bool {
        return a.first > b.first;
        });

    correl = vecCorrel.front().first;
    number = vecCorrel.front().second;

    if (correl > 0.5f)
    {
        std::cout << vecCorrel.front().first << std::endl;
        std::cout << number << std::endl;
        return number;
    }
    else return 0;
    
}

bool DigitRecognizer::loadDigits()
{
    std::vector<Point> cnt_empty;
    for (size_t i = 1; i <= 9; i++)
    { 
        digits.emplace_back(
            make_pair(imread(std::to_string(i) + ".jpg", IMREAD_GRAYSCALE), cnt_empty));
    }

    return true;
}

bool DigitRecognizer::prepareDigits()
{

    //wrzuc to jednej funkcji te wycinanie
    
    Rect rect;
    std::vector<std::vector<Point>> vecContour;
    std::vector<std::pair<Mat, std::vector<Point>>> prepared;

    for (auto& digit : digits)
    {
        GaussianBlur(digit.first, digit.first, Size(3, 3), 0);
        threshold(digit.first, digit.first, 200, 255, THRESH_BINARY_INV);
        findContours(digit.first, vecContour, RETR_TREE, CHAIN_APPROX_SIMPLE);
        rect = boundingRect(vecContour.front());

        prepared.emplace_back(
            make_pair(digit.first(rect), vecContour.front()));
        
    }

    digits = prepared;

    return false;
}

Mat DigitRecognizer::preprocessImage(const Mat& img)
{
    Rect rect;
    std::vector<std::vector<Point>> vecContour;
    Mat M;

    findContours(img, vecContour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    rect = boundingRect(vecContour.front());
    
    Mat retImg = img(rect);

    return retImg;
}