#include <opencv2/opencv.hpp>
#include <iostream>

#include "dehaze.h"

int main(int argc, char** argv) {

	cv::Mat img = cv::imread("hazy_scene.png", cv::IMREAD_COLOR);
	cv::Mat dehazed = dehaze(img);

	cv::imshow("source", img);
	cv::imshow("dehazed", dehazed);
	cv::waitKey(0);
	return 0;
}