#include "atm_light.h"
#include "filtering.hpp"
#include "utils.h"

/* Делит изображение на 4 региона и выбирает тот, у кого больше средняя яркость
* @param img входное изображение
* @param chosen_reg выбранный регион
* @param x0 коордианата верхнего левого угла выбранного региона
* @param y0 коордианата верхнего левого угла выбранного региона
*/
void qtdiv(const cv::Mat& img, cv::Mat& chosen_reg, int& x0, int& y0) {
	int height = img.rows;
	int width = img.cols;

	cv::Mat first_sector = img(cv::Range(0, height / 2), cv::Range(0, width / 2));
	cv::Mat second_sector = img(cv::Range(0, height / 2), cv::Range(width / 2, width));
	cv::Mat third_sector = img(cv::Range(height / 2, height), cv::Range(0, width / 2));
	cv::Mat fourth_sector = img(cv::Range(height / 2, height), cv::Range(width / 2, width));

	std::vector<cv::Mat> sectors = { first_sector, second_sector, third_sector, fourth_sector };
	double max_bright = -1;
	int max_bright_idx{};
	for (int i = 0; i < 4; i++) {
		double bright = cv::mean(sectors[i])[0];
		if (bright > max_bright) {
			max_bright = bright;
			max_bright_idx = i;
		}
	}

	int offset_x = max_bright_idx % 2;
	int offset_y = max_bright_idx / 2;
	chosen_reg = sectors[max_bright_idx];
	x0 = x0 + offset_x * (width / 2);
	y0 = y0 + offset_y * (height / 2);
}

// Вычисляет точку атмосферного света методом QuadTree division
cv::Vec3b get_atm_light(const cv::Mat& img, double stopdiv_size, int minfilt_size) {
	assert(img.channels() == 3 && "not bgr picture");

	cv::Mat gray_img;
	cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
	min_filter(gray_img, gray_img, minfilt_size);

	int height = gray_img.rows, width = gray_img.cols;
	int div_height = height, div_width = width;
	int x0 = 0, y0 = 0;

	while ((double)div_height / height >= stopdiv_size && (double)div_width / width >= stopdiv_size) {
		qtdiv(gray_img, gray_img, x0, y0);
		div_height = gray_img.rows;
		div_width = gray_img.cols;
	}
	// Участок изображения с максимальной яркостью атмосферного света
	cv::Mat chosen_reg = img(cv::Range(y0, y0 + div_height), cv::Range(x0, x0 + div_width));
	cv::Mat deviation = white_deviation(chosen_reg);
	cv::Point atm_light;
	cv::minMaxLoc(deviation, NULL, NULL, &atm_light);
	cv::Vec3b res = chosen_reg.at<cv::Vec3b>(atm_light);

	return res;
}

//Адаптивный атмосферный свет
cv::Mat adaptive_atm_light(const cv::Mat& tmap, const cv::Mat& img, const cv::Mat& gray, cv::Scalar atm_light, int lambda) {
	atm_light = atm_light / 255;
	cv::Mat beta = 1 / (tmap - 1);
	std::vector<cv::Mat> channels(3);
	for (int i = 0; i < 3; i++) {
		channels[i] = guided_filter(gray, (beta.mul(beta).mul(gray) + lambda * atm_light[i]) / (beta.mul(beta) + lambda), 30, 1e-4);
	}

	cv::Mat adaptive_light;
	cv::merge(channels, adaptive_light);
	cv::threshold(adaptive_light, adaptive_light, 0.0, 0.0, cv::THRESH_TOZERO);
	cv::threshold(adaptive_light, adaptive_light, 1.0, 1.0, cv::THRESH_TRUNC);

	return adaptive_light;
}