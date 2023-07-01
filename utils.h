#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

/* Создаёт барьер из граничных участков изображения
* @param image входное BGR изображение
* @param border_size размер границы
*/
cv::Mat make_border(const cv::Mat& image, int border_size);

/* Убирает границы барьера
* @param image входное BGR изображение
* @param border_size размер границы
*/
cv::Mat remove_border(const cv::Mat& image, int border_size);

/* Создаёт гистограмму распределения одного цветового канала
* @param cnl входной канал
* @param hist выходной массив для гистограммы
*/
void calc_cnl_hist(const cv::Mat& cnl, cv::Mat& hist);

/* Считает отклонение от белого для каждого пикселя изображения, как евкликдову норму
* @param img входное изображение
*/
cv::Mat white_deviation(const cv::Mat& img);

/* Аналогия одноименной ф - ии в matlab.Переводим участки изображения размером xsize на ysize в столбтсы
* @param img входное изображение
* @param xsize ширина блока, которые превращается в столбец
* @param ysize длина блока, которые превращается в столбец
*/
template<typename T>
cv::Mat im2col(const cv::Mat& img, int xsize, int ysize) {
	int n_xblock = (int)ceil(img.cols / (double)xsize);
	int n_yblock = (int)ceil(img.rows / (double)ysize);

	std::vector<cv::Mat> channels;
	cv::split(img, channels);

	for (int idx = 0; idx < channels.size(); idx++) {
		cv::Mat res = cv::Mat::zeros(xsize * ysize, n_xblock * n_yblock, channels[idx].depth());
		int col_idx = 0;

		for (int j = 0; j < n_xblock; j++) {
			for (int i = 0; i < n_yblock; i++) {
				int n_col{ 0 };
				int row_idx = n_col * ysize;
				int x_from = j * xsize;
				int y_from = i * ysize;
				int x_to = x_from + xsize > channels[idx].cols ? channels[idx].cols : x_from + xsize;
				int y_to = y_from + ysize > channels[idx].rows ? channels[idx].rows : y_from + ysize;
				for (int xx = x_from; xx < x_to; xx++) {
					row_idx = n_col * ysize;
					for (int yy = y_from; yy < y_to; yy++) {
						res.at<T>(row_idx, col_idx) = channels[idx].at<T>(yy, xx);
						row_idx++;
					}
					n_col++;
				}
				col_idx++;
			}
		}

		channels[idx] = res;
	}

	cv::Mat columned_img;
	cv::merge(channels, columned_img);

	return columned_img;
}

/* Аналогия одноименной ф - ии в matlab.Обратная к im2col
* @param img входное изображение
* @param xsize ширина блока, которые превращается в столбец
* @param ysize длина блока, которые превращается в столбец
* @param n_xblock количество блоков по горизонтали
* @param n_yblock количество блоков по вертикали
*/
template<typename T>
cv::Mat col2im(const cv::Mat& img, int xsize, int ysize, int n_xblock, int n_yblock) {
	std::vector<cv::Mat> channels;
	cv::split(img, channels);

	for (int i = 0; i < channels.size(); i++) {
		cv::Mat recovered_img(ysize * n_yblock, xsize * n_xblock, img.depth());

		int xblock_idx = 0, yblock_idx = 0;
		for (int j = 0; j < img.cols; j++) {
			int row_idx = 0;
			if (yblock_idx == n_yblock) {
				xblock_idx++;
				yblock_idx = 0;
			}
			int x_from = xblock_idx * xsize;
			int x_to = x_from + xsize;
			int y_from = yblock_idx * ysize;
			int y_to = y_from + ysize;
			for (int y = y_from; y < y_to; y++) {
				for (int x = x_from; x < x_to; x++) {
					recovered_img.at<T>(y, x) = img.at<T>(row_idx, j);
					row_idx++;
				}
			}
			yblock_idx++;
		}

		channels[i] = recovered_img;
	}

	cv::Mat res;
	cv::merge(channels, res);
	return res;
}

/* Приводит bgr значения к диапазону[0, 1]
* @param img входное изображение
*/
cv::Mat toDuble(const cv::Mat& img);

/* Куммулятивная сумма значений пикселей входного изображение
* @param img входное изображение
* @param out выходной массив
* @param dim направление, по которому суммировать. 1 - вертикально, 2 - горизнтально
*/
void cumsum(const cv::Mat& img, cv::Mat& out, int dim);

/* Печатает содержимое матритсы
* @param image входное изображение
*/
template<typename T>
void print_matrix(const cv::Mat& image) {
	std::vector<cv::Mat> channels;
	cv::split(image, channels);
	for (auto ch : channels) {
		for (int i = 0; i < ch.rows; i++) {
			for (int j = 0; j < ch.cols; j++) {
				std::cout << (float)ch.at<T>(i, j) << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}