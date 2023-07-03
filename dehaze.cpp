#include "dehaze.h"
#include <ctime>

// Алгоритм удаления дымки с BGR изображения
cv::Mat dehaze(const cv::Mat& img, int patch_size, int max_iter, double eps, int lambda,
			   double tmin, double dp, bool log) {
	assert(img.channels() == 3 && "not bgr picture");

	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	gray = toDuble(gray);

	cv::Vec3b a = get_atm_light(img);
	cv::Scalar a_s(a[0], a[1], a[2]);

	cv::Mat i = im2col<uchar>(img, patch_size, patch_size);
	cv::Mat tmap = tmap_optimal(i, a_s, patch_size, max_iter, eps, log);


	//Удаление блочных артефактов с карты пропускания
	tmap = cv::repeat(tmap, patch_size * patch_size, 1);
	tmap = col2im<double>(tmap, patch_size, patch_size, ceil((double)img.cols / patch_size), ceil((double)img.rows / patch_size));
	tmap = tmap(cv::Range(0, img.rows), cv::Range(0, img.cols));
	tmap = guided_filter(gray, tmap, 30, 1e-4);

	cv::Mat adapt_light = adaptive_atm_light(tmap, img, gray, a_s, lambda);

	cv::Mat dimg = toDuble(img);
	cv::Mat temp;
	cv::pow(cv::max(tmap, tmin), dp, temp);
	cv::Mat res = (dimg - adapt_light);
	std::vector<cv::Mat> temp_3ch = { temp, temp, temp };
	cv::merge(temp_3ch, temp);
	res = res / temp + adapt_light;

	cv::threshold(res, res, 0.0, 0.0, cv::THRESH_TOZERO);
	cv::threshold(res, res, 1.0, 1.0, cv::THRESH_TRUNC);
    res.convertTo(res, CV_8UC3, 255.);

	return res;
}
