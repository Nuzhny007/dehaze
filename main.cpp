#include <opencv2/opencv.hpp>
#include <iostream>

#include "dehaze.h"

int main(int argc, char** argv)
{
    const char* keys =
    {
        "{@1          |images/hazy_scene.png  | picture file |}"
        "{path_size   |16                     | path_size |}"
        "{max_iter    |200                    | max_iter |}"
        "{eps         |10e-7                  | eps |}"
        "{lamda       |4                      | lamda |}"
        "{tmin        |0.2                    | tmin |}"
        "{dp          |0.7                    | dp |}"
        "{log         |0                      | Enable logging |}"
        "{out         |res.png                | save result image |}"
        "{show        |1                      | show result image |}"
    };
    cv::CommandLineParser parser(argc, argv, keys);
    parser.printMessage();

    std::string srcFile = parser.get<std::string>(0);
    std::string resFile = parser.get<std::string>("out");
    int path_size = parser.get<int>("path_size");
    int max_iter = parser.get<int>("max_iter");
    double eps = parser.get<double>("eps");
    int lamda = parser.get<int>("lamda");
    double tmin = parser.get<double>("tmin");
    double dp = parser.get<double>("dp");
    bool log = parser.get<int>("log") != 0;
    bool show = parser.get<int>("show") != 0;


    cv::Mat img = cv::imread(srcFile, cv::IMREAD_COLOR);

    if (img.empty())
    {
        std::cerr << "Can't read image: " << srcFile << std::endl;
        return -1;
    }
    else
    {
        std::cout << "Image readed successfully: " << srcFile << std::endl;
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    cv::Mat dehazed = dehaze(img, path_size, max_iter, eps, lamda, tmin, dp, log);
    auto stopTime = std::chrono::high_resolution_clock::now();
    auto workTime = std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime);
    std::cout << "work time = " << workTime.count() << std::endl;

    if (show)
    {
        cv::imshow("source", img);
        cv::imshow("dehazed", dehazed);
        cv::waitKey(0);
    }
    if (!resFile.empty())
        cv::imwrite(resFile, dehazed);
    return 0;
}
