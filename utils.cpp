#include "utils.h"

//Создаёт барьер из граничных участков изображения
cv::Mat make_border(const cv::Mat& image, int border_size) {
	int size_top = border_size / 2;
	int size_bottom{};
	if (border_size % 2 == 0) {
		size_bottom = border_size / 2 - 1;
	}
	else {
		size_bottom = size_top;
	}

	cv::Mat border_top = image.rowRange(0, size_top);
	cv::Mat border_bottom = image.rowRange(image.rows - size_bottom - 1, image.rows);
	cv::Mat column_matx;
	std::vector<cv::Mat> concat_order = { border_top, image, border_bottom };
	cv::vconcat(concat_order, column_matx);

	cv::Mat border_left = column_matx.colRange(0, size_top);
	cv::Mat border_right = column_matx.colRange(column_matx.cols - size_bottom - 1, column_matx.cols);
	cv::Mat bordered_matx;
	concat_order = { border_left, column_matx, border_right };
	cv::hconcat(concat_order, bordered_matx);

	return bordered_matx;
}

// Убирает границы барьера
cv::Mat remove_border(const cv::Mat& image, int border_size) {
	int size_top = border_size / 2;
	int size_bottom{};
	if (border_size % 2 == 0) {
		size_bottom = border_size / 2 - 1;
	}
	else {
		size_bottom = size_top;
	}

	cv::Mat center_matx = image(cv::Range(size_top, image.cols - size_bottom - 1),
								cv::Range(size_top, image.rows - size_bottom - 1));

	return center_matx;
}

// Создаёт гистограмму распределения одного цветового канала
void calc_cnl_hist(const cv::Mat& cnl, cv::Mat& hist) {
	int ch[] = { 0 };
	int histSize[] = { 256 };
	float h_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges };
	cv::calcHist(&cnl, 1, ch, cv::noArray(), hist, 1, histSize, ranges, true);
}

// Считает отклонение от белого для каждого пикселя изображения, как евкликдову норму
cv::Mat white_deviation(const cv::Mat& img) {
	cv::Mat deviation = cv::Mat::zeros(img.rows, img.cols, CV_64F);
	cv::Mat conv_img;
	img.convertTo(conv_img, CV_64F);
	std::vector<cv::Mat> channels;
	cv::split(conv_img, channels);
	for (auto ch : channels) {
		cv::Mat powered_ch;
		cv::pow(ch - 255, 2, powered_ch);
		deviation = deviation + powered_ch;
	}
	cv::sqrt(deviation, deviation);

	return deviation;
}

//Куммулятивная сумма значений пикселей входного изображение
void cumsum(const cv::Mat& img, cv::Mat& out, int dim) {

	if (dim == 1) {
		img.rowRange(0, 1).copyTo(out.rowRange(0, 1));
		for (int i = 0; i < img.cols; i++) {
			for (int j = 1; j < img.rows; j++) {
				out.at<double>(j, i) = out.at<double>(j - 1, i) + img.at<double>(j, i);
			}
		}
	}
	else if (dim == 2) {
		img.colRange(0, 1).copyTo(out.colRange(0, 1));
		for (int i = 0; i < img.rows; i++) {
			for (int j = 1; j < img.cols; j++) {
				out.at<double>(i, j) = out.at<double>(i, j - 1) + img.at<double>(i, j);
			}
		}
	}
	
}

// Приводит bgr значения к диапазону[0, 1]
cv::Mat toDuble(const cv::Mat& img) {
	cv::Mat temp;
	img.convertTo(temp, CV_64F);
	temp /= 255;
	return temp;
}


