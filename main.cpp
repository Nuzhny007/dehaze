#include <opencv2/opencv.hpp>
#include <iostream>

#include "dehaze.h"

int main(int argc, char** argv)
{
    std::string srcFile = "images/hazy_scene.png";
    if (argc > 1)
        srcFile = argv[1];

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

	cv::Mat dehazed = dehaze(img);

	cv::imshow("source", img);
	cv::imshow("dehazed", dehazed);
	cv::waitKey(0);
	return 0;
}
