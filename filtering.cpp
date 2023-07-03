#include "filtering.hpp"
#include "utils.h"
#include <iostream>


/* Симметричные отсупы
* @param img входное изображение
* @param size величина отсупа
*/
cv::Mat sympad(const cv::Mat& img, int size) {
	std::vector<cv::Mat> channels;
	std::vector<cv::Mat> padded_channels;
	cv::split(img, channels);

	for (auto ch : channels) {
		cv::Mat padded_ch;
		cv::Mat top_pad = ch.rowRange(1, (size - 1) / 2 + 1);
		cv::Mat bottom_pad = ch.rowRange(img.rows - (size - 1) / 2 - 1, img.rows - 1);
		cv::flip(top_pad, top_pad, 0);
		cv::flip(bottom_pad, bottom_pad, 0);
		std::vector<cv::Mat> concat_order = { top_pad, ch, bottom_pad };
		cv::vconcat(concat_order, padded_ch);

		cv::Mat left_pad = padded_ch.colRange(1, (size - 1) / 2 + 1);
		cv::Mat right_pad = padded_ch.colRange(padded_ch.cols - (size - 1) / 2 - 1, padded_ch.cols - 1);
		cv::flip(left_pad, left_pad, 1);
		cv::flip(right_pad, right_pad, 1);
		concat_order = { left_pad, padded_ch, right_pad };
		cv::hconcat(concat_order, padded_ch);

		padded_channels.push_back(padded_ch);
	}

	cv::Mat padded_img;
	cv::merge(padded_channels, padded_img);
	return padded_img;
}

/* Фильтр. На заданном радиусе выбирает пиксель с миниальной яркостью и заменяет им все остальные
* @param padded_img входное изображение с отступами
* @param size радиус
*/
template<typename T>
void custom_filt(const cv::Mat& padded_img, cv::Mat& out_img, int size) {
	int width = padded_img.cols - size + 1;
	int height = padded_img.rows - size + 1;
	std::vector<cv::Mat> channels;
	std::vector<cv::Mat> out_channels;
	cv::split(padded_img, channels);
	for (auto ch : channels) {
		cv::Mat res(height, width, padded_img.depth());
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				double cur_min;
				cv::minMaxIdx(ch(cv::Range(i, i + size), cv::Range(j, j + size)), &cur_min);
				res.at<T>(i, j) = (T)cur_min;
			}
		}
		out_channels.push_back(res);
	}
	
	cv::merge(out_channels, out_img);
}

// Минимальный фильтр. Уменьшает интенсивность белых объектов.
void min_filter(const cv::Mat& img, cv::Mat& out_img, int size = 5) {
	int padsize = (size - 1) / 2;
	cv::Mat padded_img = sympad(img, size);
	if (img.depth() == CV_8U) {
		custom_filt<uchar>(padded_img, out_img, size);
	}
	else if (img.depth() == CV_32F) {
		custom_filt<float>(padded_img, out_img, size);
	}
}

/* Box filter
*/
cv::Mat box_filter(const cv::Mat& img, int r) {
	cv::Mat out_img = cv::Mat::zeros(img.size(), img.depth());
	int h = img.rows;
	int w = img.cols;
	cv::Mat imCum(img.size(), img.depth());

	cumsum(img, imCum, 1);
	imCum.rowRange(r, 2 * r + 1).copyTo(out_img.rowRange(0, r + 1));
	cv::Mat tmp = imCum.rowRange(2 * r + 1, h) - imCum.rowRange(0, h - 2 * r - 1);
	tmp.copyTo(out_img.rowRange(r + 1, h - r));
	tmp = cv::repeat(imCum.rowRange(h - 1, h), r, 1) - imCum.rowRange(h - 2 * r - 1, h - r - 1);
	tmp.copyTo(out_img.rowRange(h - r, h));

	cumsum(out_img, imCum, 2);
	imCum.colRange(r, 2 * r + 1).copyTo(out_img.colRange(0, r + 1));
	tmp = imCum.colRange(2 * r + 1, w) - imCum.colRange(0, w - 2 * r - 1);
	tmp.copyTo(out_img.colRange(r + 1, w - r));
	tmp = cv::repeat(imCum.colRange(w - 1, w), 1, r) - imCum.colRange(w - 2*r - 1, w - r - 1);
	tmp.copyTo(out_img.colRange(w - r, w));

	return out_img;
}

/* Guided filter
*/
cv::Mat guided_filter(const cv::Mat& img, const cv::Mat& p, int r, double eps) {
	cv::Mat N = box_filter(cv::Mat::ones(img.size(), img.depth()), r);

	cv::Mat mean_I = box_filter(img, r) / N;
	cv::Mat mean_p = box_filter(p, r) / N;
	cv::Mat mean_Ip = box_filter(img.mul(p), r) / N;
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	cv::Mat mean_II = box_filter(img.mul(img), r) / N;
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);

	cv::Mat a = cov_Ip / (var_I + eps);
	cv::Mat b = mean_p - a.mul(mean_I);

	cv::Mat mean_a = box_filter(a, r) / N;
	cv::Mat mean_b = box_filter(b, r) / N;

	cv::Mat q = mean_a.mul(img) + mean_b;

	return q;
}